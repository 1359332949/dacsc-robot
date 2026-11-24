"""Reference-only contrast on all ten held-out seed-0 Clean rollouts (not closed-loop gains)."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np


def audit(root, old_file):
    assert (
        hashlib.sha256(old_file.read_bytes()).hexdigest()
        == "9dceecd7ac5d343b49f2c2c7808becbd50ec6ed90b8077b7bd54c57e66f0f44f"
    )
    assert (
        hashlib.sha256((root / "full.npz").read_bytes()).hexdigest()
        == (root / "full.sha256").read_text()
    )
    rows = []
    ratios = []
    with np.load(old_file) as old, np.load(root / "full.npz") as new:
        for f in sorted((root / "source").glob("*.json")):
            v = json.loads(f.read_text())
            t = v["task"]
            x = np.load(root / "source" / (v["case_id"] + ".npy"))
            m = x[:, 0]
            sd = np.sqrt(np.maximum(0, x[:, 1] - m * m))
            vals = {}
            ratios.append(new[t + "__3__tau"] / np.maximum(old[t + "__3__tau"], 1e-12))
            for name, z in [("old", old), ("full", new)]:
                mu, ss, tau = [z[t + "__3__" + k] for k in ("mean", "std", "tau")]
                score = ((m - mu) ** 2 + (sd - ss) ** 2) / (ss**2 + 1e-6)
                vals[name] = 100 * (score > tau).mean(-1)[5:8].mean(0)
            rows.append(
                dict(
                    task=t,
                    id=v["case_id"],
                    chunks=len(x),
                    available_5_7=min(3, len(x) - 5),
                    old_pct=vals["old"].tolist(),
                    full_pct=vals["full"].tolist(),
                )
            )
    assert len(rows) == 10
    avg = lambda key: np.mean([r[key] for r in rows], axis=0).tolist()
    return dict(
        equal_task_mean_old=avg("old_pct"),
        equal_task_mean_full=avg("full_pct"),
        views=["agentview", "eye_in_hand"],
        tau_new_old_median=np.median(ratios, axis=(0, 2)).tolist(),
        cases=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("old", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(audit(a.root, a.old), indent=2) + "\n")
