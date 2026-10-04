"""Draw the reference STS architecture and the hidden-neuron computation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#24547a"
TEAL = "#197c72"
INK = "#20313e"
GRAY = "#667887"
ORANGE = "#b76832"


def main():
    fig, ax = plt.subplots(figsize=(14, 8.4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.025, top=0.98)
    ax.set(xlim=(0, 14), ylim=(0, 8.4), aspect="equal")
    ax.axis("off")

    def text(x, y, value, size=11, color=INK, **kwargs):
        return ax.text(x, y, value, fontsize=size, color=color, va="center", **kwargs)

    def arrow(start, end, color=GRAY, style="-", width=1.2):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                linewidth=width,
                linestyle=style,
            )
        )

    def box(x, y, w, h, label, edge=BLUE, size=12, fill="#f4f8fb"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.2,
            )
        )
        text(x + w / 2, y + h / 2, label, size=size, ha="center", color=edge)

    text(7, 8.12, "STS · 49 → 64 → 10", size=21, weight="bold", ha="center")
    text(2.3, 7.62, "49 inputs", size=13, weight="bold", ha="center")
    text(6.0, 7.62, "64 hidden neurons", size=13, weight="bold", ha="center")
    text(9.7, 7.62, "10 output logits", size=13, weight="bold", ha="center")
    text(
        4.15, 7.10, r"$W\in\{-1,0,+1\}^{64\times49}$", size=13, ha="center", color=TEAL
    )
    text(
        7.85, 7.10, r"$V\in\{-1,0,+1\}^{10\times64}$", size=13, ha="center", color=TEAL
    )

    inputs = [6.75, 6.32, 5.89, 5.05]
    hidden = [6.75, 6.25, 5.75, 5.05]
    outputs = [6.80 - 0.205 * i for i in range(10)]
    for y in inputs:
        for z in hidden:
            ax.plot([2.46, 5.82], [y, z], color="#d4dfe7", lw=0.7, zorder=0)
    for y in hidden:
        for z in outputs:
            ax.plot([6.18, 9.57], [y, z], color="#d4dfe7", lw=0.6, zorder=0)
    # Three schematic edge types, independent of any particular trained checkpoint.
    ax.plot([2.46, 5.82], [6.75, 6.75], color=TEAL, lw=1.8)
    ax.plot([2.46, 5.82], [6.32, 5.75], color=ORANGE, lw=1.8)
    ax.plot([2.46, 5.82], [5.89, 5.05], color=GRAY, lw=1.2, ls="--")
    for y, label in zip(inputs, [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_{49}$"]):
        ax.add_patch(Circle((2.3, y), 0.17, facecolor="#f0f5f9", edgecolor=BLUE))
        text(2.3, y, label, size=10, ha="center")
    text(2.3, 5.47, "⋮", size=20, ha="center")
    for y, label in zip(hidden, [r"$h_1$", r"$h_2$", r"$h_3$", r"$h_{64}$"]):
        ax.add_patch(Circle((6, y), 0.18, facecolor="#e8f4ef", edgecolor=TEAL))
        text(6, y, label, size=10, ha="center")
    text(6, 5.38, "⋮", size=20, ha="center")
    for k, y in enumerate(outputs):
        ax.add_patch(Circle((9.7, y), 0.085, facecolor="#edf3f9", edgecolor=BLUE))
        text(9.91, y, rf"$z_{k}$", size=9)

    for row in range(7):
        for col in range(7):
            ax.add_patch(
                Rectangle(
                    (0.30 + 0.15 * col, 5.85 + 0.15 * row),
                    0.15,
                    0.15,
                    facecolor="#edf2f6",
                    edgecolor="#a7b8c5",
                    linewidth=0.4,
                )
            )
    arrow((1.48, 6.37), (1.96, 6.37))
    text(0.83, 5.52, "MNIST 7×7", size=11, ha="center")
    text(2.3, 4.64, "Normalized pixels", size=10, ha="center", color=GRAY)
    text(
        6,
        4.64,
        "Quadratic activation + discrete gain",
        size=10,
        ha="center",
        color=GRAY,
    )
    text(
        4.5,
        4.28,
        "Input / hidden nodes shown in part; both layers are fully connected.",
        size=9,
        ha="center",
        color=GRAY,
    )

    arrow((10.32, 6.0), (10.87, 6.0))
    box(
        11.02,
        5.58,
        2.25,
        0.84,
        r"$\hat{y}=\arg\max_k z_k$" + "\nPredicted digit",
        size=12,
    )
    text(11.15, 5.08, "Cross-entropy for training", size=10)
    text(11.15, 4.76, "No output activation", size=10, color=GRAY)
    box(
        8.65,
        3.70,
        4.9,
        0.62,
        r"$z_k=\left(\sum_j v_{kj}h_j+\delta_{\rm out}b_k\right)/\tau_{{\rm out},k}$",
        size=13,
    )

    text(0.3, 3.90, "Hidden neuron j", size=14, weight="bold")
    text(
        0.3,
        3.49,
        r"$h_j=\lambda_j\left(u_j+u_j^2\right),\quad"
        r"u_j=\left(\sum_i w_{ji}x_i+\delta b_j\right)/\tau_j$",
        size=14,
    )

    items = [
        (0.40, 1.95, r"$\sum_i w_{ji}x_i$", "Ternary weights"),
        (2.90, 1.72, r"$+\,\delta b_j$", r"$b_j\in\{-1,0,+1\}$"),
        (5.17, 1.38, r"$\div\,\tau_j$", "Fixed scale"),
        (7.10, 2.02, r"$\phi(u)=u+u^2$", "Quadratic activation"),
        (9.67, 1.72, r"$\times\,\lambda_j$", r"$\lambda_j\in\{\frac{1}{2},1,2\}$"),
    ]
    for i, (x, w, label, caption) in enumerate(items):
        box(x, 2.34, w, 0.68, label, edge=TEAL if i in [0, 1, 4] else BLUE, size=15)
        text(x + w / 2, 2.02, caption, size=10, ha="center")
        if i:
            old_x, old_w, *_ = items[i - 1]
            arrow((old_x + old_w + 0.06, 2.68), (x - 0.10, 2.68))
    text(6.83, 3.01, r"$u_j$", size=11, ha="center")
    arrow((11.49, 2.68), (12.12, 2.68))
    text(12.40, 2.68, r"$h_j$", size=18, ha="center", color=TEAL)

    box(
        0.4,
        0.78,
        6.3,
        0.78,
        "Trained by STS · no gradients\nTernary weights and biases; discrete hidden gains",
        edge=TEAL,
        size=11,
    )
    box(
        7.12,
        0.78,
        6.3,
        0.78,
        "Calibrated once, then fixed\nδ: layer input RMS · τ: initial neuron preactivation RMS",
        size=11,
    )
    for x, label, color, style in [
        (3.0, "+1", TEAL, "-"),
        (5.4, "0", GRAY, "--"),
        (7.8, "−1", ORANGE, "-"),
    ]:
        ax.plot([x, x + 0.5], [0.33, 0.33], color=color, lw=1.8, ls=style)
        text(x + 0.64, 0.33, label, size=10, color=color)
    text(9.20, 0.33, "Schematic ternary connections", size=10, color=GRAY)

    for ext in ["png", "svg"]:
        target = ROOT / "docs/figures" / f"architecture.{ext}"
        fig.savefig(target, dpi=180)
        if ext == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
