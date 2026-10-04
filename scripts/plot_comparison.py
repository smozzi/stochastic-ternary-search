"""Plot complete corrected confirmations; no smoothing or checkpoint selection."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / "results/comparison_checkpoints.csv").open()))
sets = {
    "comparison": {
        "sts_int32": "STS",
        "adam_b128": "Adam FP32 · cosine · batch 128",
        "adam_lsq_b128": "Adam + STE · learned steps · batch 128",
        "adam_lsq_b5000": "Adam + STE · learned steps · batch 5000",
    },
    "ste_diagnostic": {
        "adam_ste_b128": "STE · detached scales · batch 128",
        "adam_ste_b5000": "STE · detached scales · batch 5000",
        "adam_lsq_b128": "STE · learned steps · batch 128",
        "adam_lsq_b5000": "STE · learned steps · batch 5000",
    },
}
for name, methods in sets.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for method, label in methods.items():
        times = sorted(
            {
                float(r["target"])
                for r in rows
                if r["condition"] == method and 0 < float(r["target"]) <= 240
            }
        )
        assert times and times[-1] == 240
        for ax, metric in zip(axes, ["train_loss", "validation_loss"]):
            groups = [
                [
                    float(r[metric])
                    for r in rows
                    if r["condition"] == method and float(r["target"]) == t
                ]
                for t in times
            ]
            assert all(len(g) == 3 for g in groups)
            mean = np.array([np.mean(g) for g in groups])
            sd = np.array([np.std(g, ddof=1) for g in groups])
            (line,) = ax.plot(times, mean, label=label, lw=2)
            ax.fill_between(
                times, mean - sd, mean + sd, color=line.get_color(), alpha=0.12
            )
            ax.set(
                xlabel="Training time (seconds)",
                ylabel=metric.replace("_", " ")
                .replace("loss", "cross-entropy")
                .capitalize(),
            )
            ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "MNIST 7×7 · 49→64→10 · 3 seeds · mean ± sample SD · 8 CPU threads", fontsize=13
    )
    for ext in ["png", "svg"]:
        target = root / "docs/figures" / f"{name}.{ext}"
        fig.savefig(target, dpi=180)
        if ext == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
