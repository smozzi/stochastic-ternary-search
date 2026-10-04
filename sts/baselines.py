"""ReLU Adam and TWN-like identity-STE comparators; separate from STS search."""

import math
import torch
from torch.nn import functional as F
from .batch import FastBatchStream


def learning_rate(initial, schedule, update, horizon):
    if schedule == "constant":
        return initial
    if schedule != "cosine" or horizon < 1:
        raise ValueError("invalid schedule")
    progress = min(max(update / horizon, 0.0), 1.0)
    return initial * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def ternarize(w):
    absolute = w.abs()
    mask = absolute > 0.7 * absolute.mean(1, keepdim=True)
    codes = torch.where(mask, w.sign(), torch.zeros_like(w)).to(torch.int8)
    scale = (absolute * mask).sum(1) / mask.sum(1).clamp_min(1)
    return codes, scale


class BaselineMLP(torch.nn.Module):
    def __init__(self, mode="adam_ste", width=64, seed=0):
        super().__init__()
        if mode not in ["adam", "adam_ste"] or width not in [32, 64, 128]:
            raise ValueError("unsupported model")
        self.mode = mode
        self.width = width
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.hidden = torch.nn.Linear(49, width)
            self.output = torch.nn.Linear(width, 10)
            torch.nn.init.kaiming_uniform_(self.hidden.weight, nonlinearity="relu")
            torch.nn.init.xavier_uniform_(self.output.weight)
            torch.nn.init.zeros_(self.hidden.bias)
            torch.nn.init.zeros_(self.output.bias)

    def weight(self, w):
        if self.mode == "adam":
            return w
        codes, scale = ternarize(w)
        return codes.to(w.dtype) * scale[:, None] + (w - w.detach())

    def forward(self, x):
        h = F.relu(F.linear(x, self.weight(self.hidden.weight), self.hidden.bias))
        return F.linear(h, self.weight(self.output.weight), self.output.bias)

    @torch.no_grad()
    def export(self):
        if self.mode == "adam":
            return dict(
                mode=self.mode,
                state={k: v.detach().clone() for k, v in self.state_dict().items()},
            )
        out = dict(mode=self.mode)
        for name in ["hidden", "output"]:
            layer = getattr(self, name)
            codes, scale = ternarize(layer.weight)
            out[name] = dict(codes=codes, scale=scale, bias=layer.bias.detach().clone())
        return out


@torch.no_grad()
def exported_forward(x, export):
    def layer(x, name):
        if export["mode"] == "adam":
            state = export["state"]
            return F.linear(x, state[name + ".weight"], state[name + ".bias"])
        row = export[name]
        return F.linear(x, row["codes"].float() * row["scale"][:, None], row["bias"])

    return layer(F.relu(layer(x, "hidden")), "output")


class AutogradBatchStream(FastBatchStream):
    """Allocate ordinary tensors before the sampler enters inference mode."""

    def __init__(self, size, batch, seed, threads=8):
        super().__init__(size, batch, seed, threads)
        self.allocate()

    def allocate(self):
        self.buffers = (
            torch.empty(self.batch, 49),
            torch.empty(self.batch, dtype=torch.int64),
            torch.empty(self.batch, dtype=torch.int64),
        )
        assert not self.buffers[0].is_inference()

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.allocate()


def optimizer(model, lr):
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        foreach=False,
    )


def train_batch(model, opt, x, y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    return float(loss.detach())


@torch.inference_mode()
def metrics(model, data):
    x, y = data
    correct = 0
    loss = 0.0
    for start in range(0, len(y), 1000):
        yy = y[start : start + 1000]
        logits = model(x[start : start + 1000])
        correct += int((logits.argmax(1) == yy).sum())
        loss += float(F.cross_entropy(logits, yy, reduction="sum"))
    return dict(loss=loss / len(y), accuracy=correct / len(y), score=-loss / len(y))
