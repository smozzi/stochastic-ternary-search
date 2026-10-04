"""Symmetric ternary, row-wise LSQ-inspired quantizer; no activation quantization."""

import math
import torch
from torch.nn import functional as F
from .baselines import BaselineMLP


class TernaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, step):
        u = weight / step[:, None]
        codes = u.clamp(-1, 1).round()
        ctx.save_for_backward(u, codes)
        ctx.fan_in = weight.shape[1]
        return codes * step[:, None]

    @staticmethod
    def backward(ctx, gradient):
        u, codes = ctx.saved_tensors
        inside = (u > -1) & (u < 1)
        grad_weight = gradient * inside
        grad_step = (gradient * torch.where(inside, codes - u, codes)).sum(
            1
        ) / math.sqrt(ctx.fan_in)
        return grad_weight, grad_step


class LearnedStepMLP(BaselineMLP):
    def __init__(self, mode="adam_ste", width=64, seed=0):
        if mode != "adam_ste":
            raise ValueError("LSQ baseline must use adam_ste")
        super().__init__(mode, width, seed)
        self.hidden_step = torch.nn.Parameter(
            2 * self.hidden.weight.detach().abs().mean(1)
        )
        self.output_step = torch.nn.Parameter(
            2 * self.output.weight.detach().abs().mean(1)
        )

    def forward(self, x):
        h = F.relu(
            F.linear(
                x,
                TernaryQuantizer.apply(self.hidden.weight, self.hidden_step),
                self.hidden.bias,
            )
        )
        return F.linear(
            h,
            TernaryQuantizer.apply(self.output.weight, self.output_step),
            self.output.bias,
        )

    @torch.no_grad()
    def export(self):
        result = dict(mode="adam_lsq")
        for name in ["hidden", "output"]:
            layer = getattr(self, name)
            step = getattr(self, name + "_step")
            codes = (layer.weight / step[:, None]).clamp(-1, 1).round().to(torch.int8)
            result[name] = dict(
                codes=codes,
                scale=step.detach().clone(),
                bias=layer.bias.detach().clone(),
            )
        return result

    @torch.no_grad()
    def project_steps(self):
        self.hidden_step.clamp_(min=1e-6)
        self.output_step.clamp_(min=1e-6)
