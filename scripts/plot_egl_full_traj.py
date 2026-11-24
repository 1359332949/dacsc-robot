"""Plot paired success differences and the reference-only Clean activation contrast."""

import argparse, json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(summary, out):
    d = json.loads(summary.read_text())
    g = d["grid"]["all"]
    groups = [
        ("clean", "Clean"),
        ("dual_sensor", "Bilateral Sensor"),
        ("single_sensor", "Standard Sensor"),
        ("table_texture", "Table texture"),
        ("scene_texture", "Scene texture"),
        ("lighting", "Lighting"),
        ("standard100", "Standard overall"),
    ]
    fig, ax = plt.subplots(
        1, 2, figsize=(10.8, 4.8), layout="constrained", width_ratios=[1.45, 1]
    )
    pos = np.arange(len(groups))[::-1]
    ax[0].axvline(0, color="#66717c", lw=0.9)
    for method, dy, color, label in [
        ("dead075_vs_frozen", 0.13, "#967041", "First-frame reference"),
        ("full_traj_dead075_vs_frozen", -0.13, "#11685e", "Full-trajectory reference"),
    ]:
        for i, (k, _) in enumerate(groups):
            v = g[k][method]
            ax[0].plot(v["delta_ci"], [pos[i] + dy] * 2, color=color, lw=1.7)
            ax[0].scatter(
                v["delta_pp"],
                pos[i] + dy,
                color=color,
                s=27,
                label=label if i == 0 else None,
                zorder=3,
            )
    ax[0].set_yticks(pos, [s for _, s in groups])
    ax[0].set_xlabel("Success-rate change vs Frozen (pp)")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].set_title("Paired Dev outcomes · 3 seeds", fontsize=12)
    focus = [r for r in d["clean_telemetry"] if r["focus"] and r["frozen"] is not None]
    ks = sorted({(r["seed"], r["id"]) for r in focus})
    for i, (seed, cid) in enumerate(ks):
        vs = [r for r in focus if r["seed"] == seed and r["id"] == cid]
        values = [
            100 * np.mean([r["frozen"][k][1] for r in vs]) for k in ["old", "full"]
        ]
        ax[1].plot(
            [0, 1], values, marker="o", lw=1.7, label=f"{cid[6:12]} / seed {seed}"
        )
    ax[1].set_xlim(-0.25, 1.25)
    ax[1].set_xticks([0, 1], ["First-frame", "Full-trajectory"])
    ax[1].set_ylabel("Active wrist channels (%)")
    ax[1].set_ylim(0, 100)
    ax[1].set_title("Same Frozen Clean images · chunks 5–7", fontsize=12)
    ax[1].legend(fontsize=8)
    fig.suptitle(
        "Does broader Clean coverage improve the controller?", fontweight="bold"
    )
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.12)
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.summary, a.out)
