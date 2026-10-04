"""Single-hidden-layer ternary MLP with fixed RMS calibration."""

import torch
from torch.nn import functional as F
from .controller import fraction_arms

ARCHITECTURES = {
    "single32": (49, 32, 10),
    "single64": (49, 64, 10),
    "single128": (49, 128, 10),
}


class Layout:
    def __init__(self, architecture):
        self.architecture = architecture
        self.sizes = ARCHITECTURES[architecture]
        self.layers = len(self.sizes) - 1
        self.weight_n = sum(a * b for a, b in zip(self.sizes, self.sizes[1:]))
        self.gain_n = sum(self.sizes[1:-1])
        self.bias_n = sum(self.sizes[1:])
        self.n = self.weight_n + self.gain_n + self.bias_n
        self.weights = []
        self.gains = []
        self.biases = []
        w = 0
        g = self.weight_n
        b = self.weight_n + self.gain_n
        for l, (ni, no) in enumerate(zip(self.sizes, self.sizes[1:])):
            self.weights.append(w)
            w += ni * no
            self.biases.append(b)
            b += no
            self.gains.append(g if l < self.layers - 1 else -1)
            if l < self.layers - 1:
                g += no
        self.arms = fraction_arms(self.weight_n)


class DepthMLP(torch.nn.Module):
    def __init__(self, architecture="single64"):
        super().__init__()
        self.layout = Layout(architecture)
        self.architecture = architecture
        self.calibration = None
        self.register_buffer("delta", torch.zeros(self.layout.bias_n))
        self.register_buffer("tau", torch.ones(self.layout.bias_n))
        self.register_buffer("gain_levels", torch.tensor([0.5, 1.0, 2.0]))

    def _check(self, v):
        if (
            v.dtype != torch.int8
            or v.shape[-1] != self.layout.n
            or not bool(((v >= -1) & (v <= 1)).all())
        ):
            raise ValueError("invalid flat ternary codes")

    def gains(self, g):
        return self.gain_levels[g.long() + 1]

    @torch.inference_mode()
    def calibrate(self, x, v):
        self._check(v)
        if (
            self.calibration is not None
            or v.ndim != 1
            or x.ndim != 2
            or x.shape[1] != 49
            or not len(x)
            or not torch.isfinite(x).all()
        ):
            raise ValueError("invalid one-time calibration")
        h = x
        details = []
        offset = 0
        for l, (ni, no) in enumerate(zip(self.layout.sizes, self.layout.sizes[1:])):
            wo, bo, go = (
                self.layout.weights[l],
                self.layout.biases[l],
                self.layout.gains[l],
            )
            w = v[wo : wo + ni * no].reshape(no, ni)
            b = v[bo : bo + no]
            d = h.double().square().mean().sqrt()
            z = F.linear(h.double(), w.double()) + d * b.double()
            t = z.square().mean(0).sqrt()
            self.delta[offset : offset + no].fill_(float(d))
            self.tau[offset : offset + no].copy_(torch.where(t > 0, t, 1.0))
            details.append(
                dict(delta=float(d), tau=t.tolist(), zero_tau=int((t == 0).sum()))
            )
            u = (
                F.linear(h, w.float()) + self.delta[offset : offset + no] * b.float()
            ) / self.tau[offset : offset + no]
            h = (u + u.square()) * self.gains(v[go : go + no]) if go >= 0 else u
            if not torch.isfinite(h).all():
                raise FloatingPointError("nonfinite initial activation")
            offset += no
        self.calibration = dict(examples=len(x), layers=details)

    @torch.inference_mode()
    def stages(self, x, v):
        self._check(v)
        if self.calibration is None or v.ndim != 1:
            raise ValueError("calibrated single model required")
        h = x
        offset = 0
        stages = []
        for l, (ni, no) in enumerate(zip(self.layout.sizes, self.layout.sizes[1:])):
            wo, bo, go = (
                self.layout.weights[l],
                self.layout.biases[l],
                self.layout.gains[l],
            )
            u = (
                F.linear(h, v[wo : wo + ni * no].reshape(no, ni).float())
                + self.delta[offset : offset + no] * v[bo : bo + no].float()
            ) / self.tau[offset : offset + no]
            if go >= 0:
                a = u + u.square()
                h = a * self.gains(v[go : go + no])
                stages.append(dict(u=u, activation=a, after_gain=h))
            else:
                h = u
                stages.append(dict(logits=h))
            offset += no
        return stages

    def features(self, x, v):
        s = self.stages(x, v)
        return s[-1]["logits"], s[-2]["after_gain"]

    def forward(self, x, v):
        return self.features(x, v)[0]

    @torch.inference_mode()
    def forward_many(self, x, v):
        self._check(v)
        if self.calibration is None or v.ndim != 2:
            raise ValueError("calibrated matrix required")
        h = x.unsqueeze(0).expand(len(v), -1, -1)
        offset = 0
        for l, (ni, no) in enumerate(zip(self.layout.sizes, self.layout.sizes[1:])):
            wo, bo, go = (
                self.layout.weights[l],
                self.layout.biases[l],
                self.layout.gains[l],
            )
            u = (
                torch.bmm(
                    h,
                    v[:, wo : wo + ni * no].reshape(-1, no, ni).float().transpose(1, 2),
                )
                + self.delta[offset : offset + no] * v[:, None, bo : bo + no].float()
            ) / self.tau[offset : offset + no]
            h = (
                (u + u.square()) * self.gains(v[:, go : go + no])[:, None, :]
                if go >= 0
                else u
            )
            offset += no
        return h


@torch.inference_mode()
def initial_codes(width, seed):
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(59 * width, generator=generator, dtype=torch.float64)
    weights = (u >= 1 / 3).to(torch.int8) + (u >= 2 / 3).to(torch.int8) - 1
    return torch.cat((weights, torch.zeros(2 * width + 10, dtype=torch.int8)))
