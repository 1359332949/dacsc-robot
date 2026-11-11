"""Geometry of appearance/domain shift vs task/action directions for CoRP features.

Uses captures from scripts/representation_probe_capture.py:
  * clean LIBERO (source) and LIBERO-Plus (7 perturbation dimensions)
  * two feature spaces: representation tokens (reps) and cross-attention write (writes)

Reports, per feature space and dataset:
  * effective rank (participation ratio) of clean features
  * task subspace (between-task directions on clean data)
  * domain-shift subspace per perturbation dimension (base-task-paired differences)
  * action-sensitivity subspace (ridge probe action ~ features)
  * principal angles between those subspaces
  * how much of the task/action directions a source-support projection (PSC)
    would delete (null-space energy fraction)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    meta = list(data["meta"])
    return {"reps": data["reps"], "writes": data["writes"], "actions": data["actions"], "meta": meta}


def participation_ratio(cov: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    return float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2))


def centered_basis(x: np.ndarray, k: int) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    return vh[: min(k, vh.shape[0])]


def principal_angles(a: np.ndarray, b: np.ndarray) -> dict:
    cos = np.linalg.svd(a @ b.T, compute_uv=False)
    cos = np.clip(cos, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    return {"mean_deg": float(angles.mean()), "min_deg": float(angles.min()), "max_deg": float(angles.max())}


def mean_by_task(features: np.ndarray, meta: list) -> dict[tuple[str, str], np.ndarray]:
    groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for index, record in enumerate(meta):
        key = (record["dimension"], record["base_task"])
        groups[key].append(features[index])
    return {key: np.mean(np.stack(value), axis=0) for key, value in groups.items()}


def ridge_sensitivity(features: np.ndarray, actions: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    x = features - features.mean(axis=0, keepdims=True)
    y = actions - actions.mean(axis=0, keepdims=True)
    d = x.shape[1]
    w = np.linalg.solve(x.T @ x + lam * np.eye(d), x.T @ y)  # [D, A]
    return w.T  # [A, D]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--plus", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    clean = load(args.clean)
    plus = load(args.plus)
    report: dict = {
        "model": "official CoRP LIBERO checkpoint (corp_libero.pt)",
        "datasets": {"clean": str(args.clean), "plus": str(args.plus)},
        "rank": args.rank,
    }

    for space in ("reps", "writes"):
        clean_features = clean[space]
        report[space] = {}
        clean_mean = clean_features.mean(axis=0)
        clean_basis = centered_basis(clean_features, args.rank)  # source support directions
        clean_cov = np.cov(clean_features - clean_mean, rowvar=False)
        report[space]["clean_participation_ratio"] = participation_ratio(clean_cov)

        clean_means = mean_by_task(clean_features, clean["meta"])
        # clean meta uses dimension "clean" for every task
        clean_by_base = {base: mean for (dim, base), mean in clean_means.items()}
        task_stack = np.stack([clean_by_base[base] for base in sorted(clean_by_base)])
        task_basis = centered_basis(task_stack, min(args.rank, task_stack.shape[0] - 1))
        action_basis = centered_basis(ridge_sensitivity(clean_features, clean["actions"]), args.rank)
        report[space]["task_vs_action"] = principal_angles(task_basis, action_basis)

        # clean null space (what PSC would delete)
        task_captured = ((task_basis @ clean_basis.T) ** 2).sum(axis=1)
        action_captured = ((action_basis @ clean_basis.T) ** 2).sum(axis=1)
        report[space]["psc_deletes_task_energy"] = float(1.0 - task_captured.mean())
        report[space]["psc_deletes_action_energy"] = float(1.0 - action_captured.mean())

        plus_means = mean_by_task(plus[space], plus["meta"])
        dims = sorted({dim for (dim, _) in plus_means})
        report[space]["dimensions"] = {}
        for dim in dims:
            shifts = []
            for (dim_name, base), value in plus_means.items():
                if dim_name != dim or base not in clean_by_base:
                    continue
                shifts.append(value - clean_by_base[base])
            if len(shifts) < 1:
                continue
            shift_basis = centered_basis(np.stack(shifts), min(args.rank, len(shifts)))
            angles = principal_angles(shift_basis, task_basis)
            action_energy = float(np.sum((shift_basis @ action_basis.T) ** 2) / shift_basis.shape[0])
            deletion = float(1.0 - np.sum((shift_basis @ clean_basis.T) ** 2) / shift_basis.shape[0])
            report[space]["dimensions"][dim] = {
                "n_base_tasks": len(shifts),
                "shift_vs_task": angles,
                "shift_energy_in_action_subspace": action_energy,
                "psc_deletes_shift_energy": deletion,
            }

    args.out.write_text(json.dumps(report, indent=2))
    for space in ("reps", "writes"):
        entry = report[space]
        print(f"== {space}: clean participation ratio {entry['clean_participation_ratio']:.2f}")
        print(f"   task vs action angles: {entry['task_vs_action']}")
        print(f"   PSC would delete task energy {entry['psc_deletes_task_energy']:.2f}, "
              f"action energy {entry['psc_deletes_action_energy']:.2f}")
        for dim, stats in entry.get("dimensions", {}).items():
            print(f"   {dim:22s} n={stats['n_base_tasks']} "
                  f"shift-vs-task mean {stats['shift_vs_task']['mean_deg']:6.1f}deg "
                  f"action-energy {stats['shift_energy_in_action_subspace']:.2f} "
                  f"psc-deletes {stats['psc_deletes_shift_energy']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
