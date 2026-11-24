"""Render the paired closed-loop results, keeping bilateral stress separate."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("figure", type=Path)
    args = ap.parse_args()
    s = json.loads(args.summary.read_text())["grid"]["all"]
    groups = [
        ("clean", "Clean"),
        ("dual_sensor", "Bilateral Sensor"),
        ("single_sensor", "Standard Sensor"),
        ("background", "Background"),
        ("lighting", "Lighting"),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.9),
        sharey=True,
        layout="constrained",
        width_ratios=[1.25, 1],
    )
    pos = np.arange(len(groups))[::-1]
    for ax in axes:
        ax.axvline(0, color="#555555", lw=1)
        ax.grid(axis="x", alpha=0.17)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for method, offset, marker, color, label in [
        ("dead050", 0.14, "o", "#11685e", "α = 0.50"),
        ("dead075", -0.14, "s", "#9b4d2e", "α = 0.75"),
    ]:
        vs = [s[g][method] for g, _ in groups]
        y = pos + offset
        for v, h in zip(vs, y):
            axes[0].plot(v["delta_ci"], [h, h], color=color, lw=1.8)
            axes[1].hlines(h, -v["harmed"], v["recovered"], color=color, lw=2)
            axes[1].plot(
                [-v["harmed"], v["recovered"]],
                [h, h],
                marker=marker,
                linestyle="none",
                ms=4,
                color=color,
            )
        axes[0].scatter(
            [v["delta_pp"] for v in vs],
            y,
            marker=marker,
            s=35,
            color=color,
            zorder=3,
            label=label,
        )
    axes[0].set_yticks(
        pos, [f"{label}  (n={s[g]['frozen']['n']})" for g, label in groups]
    )
    axes[0].set_xlabel("Success-rate change vs paired Frozen (pp)")
    axes[1].set_xlabel("Harmed (left)   /   recovered (right), episodes")
    axes[0].set_title("Paired difference and task-cluster 95% interval", fontsize=11)
    axes[1].set_title("Transitions pooled over three seeds", fontsize=11)
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("Block 3 bounded alignment | EGL closed-loop Dev", fontweight="bold")
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
