"""Regenerate archived V2 and current confidence figures from public CSV tables."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
for name, key, methods, end, title in [
    (
        "historical",
        "condition",
        {
            "adaptive": "STS historical V2",
            "adam_b128": "Adam FP32 · batch 128",
            "adam_ste_b128": "Adam + STE · batch 128",
            "adam_ste_b5000": "Adam + STE · batch 5000",
        },
        240,
        "Historical comparison · 3 seeds · 8 CPU threads",
    ),
    (
        "current",
        "method",
        {
            "native_int32": "STS · confidence enabled",
            "native_noconf": "STS · confidence disabled",
        },
        600,
        "Current reference · 5 seeds · 8 CPU threads",
    ),
]:
    rows = list(csv.DictReader((root / "results" / f"{name}_checkpoints.csv").open()))
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    for method, label in methods.items():
        times = sorted(
            {
                float(r["target"])
                for r in rows
                if r[key] == method and 0 < float(r["target"]) <= end
            }
        )
        values = [
            [
                float(r["validation_loss"])
                for r in rows
                if r[key] == method and float(r["target"]) == t
            ]
            for t in times
        ]
        assert all(len(v) == (3 if name == "historical" else 5) for v in values)
        means = np.array([np.mean(v) for v in values])
        sd = np.array([np.std(v, ddof=1) for v in values])
        (line,) = ax.plot(times, means, label=label, lw=2)
        ax.fill_between(
            times, means - sd, means + sd, color=line.get_color(), alpha=0.13
        )
    ax.set(
        xlabel="Training time (seconds)", ylabel="Validation cross-entropy", title=title
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    for ext in ["png", "svg"]:
        destination = (
            root
            / "docs"
            / "figures"
            / (
                f"archive/ste_constant_v2.{ext}"
                if name == "historical"
                else f"{name}.{ext}"
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
        if ext == "svg":
            destination.write_text(
                "\n".join(
                    line.rstrip() for line in destination.read_text().splitlines()
                )
                + "\n"
            )
    plt.close(fig)
