"""Same conditional 90/10 controller with an architecture-specific frozen grid."""

from collections import deque
import math
import torch

WINDOW, INTERVAL, POPULATIONS = 512, 64, (3, 6, 12, 24)


class GridController:
    def __init__(self, arms, adaptive=False):
        self.arms = tuple(arms)
        self.adaptive = bool(adaptive)
        self.k0 = 4
        self.p0 = 12
        self.steps = 0
        self.observations = deque(maxlen=WINDOW)
        self.last_decision = None

    def statistics(self, axis):
        if axis not in ("k", "p"):
            raise ValueError("unknown axis")
        arms = self.arms if axis == "k" else POPULATIONS
        groups = [
            [
                row
                for row in self.observations
                if (row[1] == self.p0 if axis == "k" else row[0] == self.k0)
                and row[0 if axis == "k" else 1] == arm
            ]
            for arm in arms
        ]
        rewards = [math.fsum(row[2] for row in group) for group in groups]
        costs = [math.fsum(row[3] for row in group) for group in groups]
        counts = [len(group) for group in groups]
        rates = [r / c if c else 0.0 for r, c in zip(rewards, costs)]
        return dict(arms=arms, rewards=rewards, costs=costs, counts=counts, rates=rates)

    def probabilities(self, axis):
        arms = self.arms if axis == "k" else POPULATIONS
        anchor = self.k0 if axis == "k" else self.p0
        if axis == "p" and not self.adaptive:
            return torch.tensor([float(p == 12) for p in arms], dtype=torch.float64)
        # Exclude the nominal action from exploration: actual nominal mass is exactly .9.
        q = torch.full((len(arms),), 0.1 / (len(arms) - 1), dtype=torch.float64)
        q[arms.index(anchor)] = 0.9
        return q

    def record(self, k, p, reward, cost):
        if (
            k not in self.arms
            or p not in POPULATIONS
            or not math.isfinite(reward)
            or reward < 0
            or not math.isfinite(cost)
            or cost <= 0
        ):
            raise ValueError("invalid observation")
        before = (self.k0, self.p0)
        self.observations.append((int(k), int(p), float(reward), float(cost)))
        self.steps += 1
        self.last_decision = None
        if self.steps % INTERVAL == 0:
            axis = "k" if (self.steps // INTERVAL) % 2 else "p"
            stats = self.statistics(axis)
            anchor = self.k0 if axis == "k" else self.p0
            best = max(stats["rates"])
            winner = anchor
            if best > 0 and not (
                stats["counts"][stats["arms"].index(anchor)]
                and stats["rates"][stats["arms"].index(anchor)] == best
            ):
                winner = stats["arms"][
                    next(
                        j
                        for j, r in enumerate(stats["rates"])
                        if stats["counts"][j] and r == best
                    )
                ]
            if axis == "k":
                self.k0 = winner
            elif self.adaptive:
                self.p0 = winner
            self.last_decision = dict(
                step=self.steps,
                axis=axis,
                before=before,
                after=(self.k0, self.p0),
                statistics=stats,
            )

    def state_dict(self):
        return dict(
            adaptive=self.adaptive,
            k0=self.k0,
            p0=self.p0,
            steps=self.steps,
            observations=tuple(self.observations),
            last_decision=self.last_decision,
        )

    def load_state_dict(self, st):
        if (
            st["adaptive"] != self.adaptive
            or st["k0"] not in self.arms
            or st["p0"] not in POPULATIONS
            or (not self.adaptive and st["p0"] != 12)
        ):
            raise ValueError("incompatible conditional controller")
        if (
            type(st["steps"]) is not int
            or st["steps"] < 0
            or len(st["observations"]) != min(WINDOW, st["steps"])
        ):
            raise ValueError("invalid window")
        for k, p, r, c in st["observations"]:
            if (
                k not in self.arms
                or p not in POPULATIONS
                or not math.isfinite(r)
                or r < 0
                or not math.isfinite(c)
                or c <= 0
            ):
                raise ValueError("invalid observation")
        self.k0, self.p0, self.steps = st["k0"], st["p0"], st["steps"]
        self.observations = deque(st["observations"], maxlen=WINDOW)
        self.last_decision = st["last_decision"]


def fraction_arms(n, fraction=0.05):
    """Nearest power of two by absolute distance; ties choose the smaller."""
    if (
        type(n) is not int
        or n < 1
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("invalid parameter count or fraction")
    powers = tuple(1 << j for j in range(n.bit_length()))
    cap = min(powers, key=lambda k: (abs(k - fraction * n), k))
    return tuple(k for k in powers if k <= cap)


import json
from pathlib import Path

_COST = json.loads(Path(__file__).with_name("cost_model.json").read_text())


def predict_batch_cost(count, extra, batch):
    if not 0 <= count <= 32 or extra not in (0, 1) or batch < 1:
        raise ValueError("invalid cost inputs")
    overhead, values = _COST["overhead"], _COST["batch"]
    cost = (
        overhead + values[count] + int(extra) * values[1]
        if count <= 12
        else overhead
        + values[12]
        + (count - 12) * (values[12] - values[8]) / 4
        + int(extra) * values[1]
    )
    return overhead + (cost - overhead + values[1]) * batch / 5000
