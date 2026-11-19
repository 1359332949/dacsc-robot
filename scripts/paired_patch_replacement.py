"""Offline paired-patch replacement diagnostic (B0-B3).

B0: matched Clean patches + Clean context.
B1: matched Shift patches + Shift context.
B2: same-state Clean patches substituted into Shift context.
B3: different-state Clean patches, matched by role/window/chunk, into Shift context.

B2/B3 use privileged Clean patches and are offline causal diagnostics only.
They are not deployable TTA methods and must never be run on target-test data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/CoRP"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))

from corp import load_policy  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402
from target_only_action_validation import _one_action_metrics  # noqa: E402
from dacsc.differentiable_policy import sample_from_z0  # noqa: E402
from dacsc.patch_diagnostics import random_clean_donor  # noqa: E402


def _tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(array), device=device, dtype=dtype).unsqueeze(0)


def _run_case(policy, patches, language_tokens, pooled_language, state_token, z0, device):
    extractor = policy.representation_extractor
    patch_dict = {
        view: _tensor(value, device=device)
        for view, value in patches.items()
    }
    pooled = _tensor(pooled_language, device=device)
    lang = _tensor(language_tokens, device=device)
    state = _tensor(state_token, device=device)
    visual = extractor.cross_attention_resampler(patch_dict, pooled)
    rep = Representation(
        tokens=torch.cat([visual, lang.to(visual.dtype), state.to(visual.dtype)], dim=1),
        pooled_language=pooled,
        action_dtype=visual.dtype,
    )
    z = _tensor(z0, device=device, dtype=visual.dtype)
    action_norm = sample_from_z0(policy, rep, z)
    return {
        "visual": visual,
        "action_norm": action_norm,
    }


def _rms(a, b):
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item())


def _recovery(base_error, intervention_error):
    if base_error is None or base_error <= 1e-12:
        return None
    return float(1.0 - intervention_error / base_error)


def _aggregate(records):
    groups = {}
    for r in records:
        key = (r["role"], r["window"])
        groups.setdefault(key, []).append(r)
    out = {}
    for (role, window), values in groups.items():
        row = {}
        metric_names = (
            "B1_vs_B0", "B2_vs_B0", "B3_vs_B0",
        )
        for name in metric_names:
            items = [v[name] for v in values if v.get(name) is not None]
            if not items:
                row[name] = None
                continue
            row[name] = {
                key: float(np.mean([x[key] for x in items]))
                for key in items[0]
            }
        row["B2_action_recovery_fraction"] = _recovery(
            row["B1_vs_B0"]["normalized_mae"] if row["B1_vs_B0"] else None,
            row["B2_vs_B0"]["normalized_mae"] if row["B2_vs_B0"] else None,
        )
        row["B3_action_recovery_fraction"] = _recovery(
            row["B1_vs_B0"]["normalized_mae"] if row["B1_vs_B0"] else None,
            row["B3_vs_B0"]["normalized_mae"] if row["B3_vs_B0"] else None,
        )
        row["B2_beats_B3_on_normalized_mae"] = bool(
            row["B2_vs_B0"] is not None and row["B3_vs_B0"] is not None
            and row["B2_vs_B0"]["normalized_mae"] < row["B3_vs_B0"]["normalized_mae"]
        )
        out[f"{role}/{window}"] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--random-seed", type=int, default=20261119)
    args = ap.parse_args()
    manifest_path = args.manifest or args.pairs.with_suffix(".manifest.json")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "corp_paired_dino_patches_v1":
        raise ValueError("unexpected patch-pair manifest schema")
    records = manifest.get("records", [])
    if any(r.get("role") not in ("source_fit", "source_validation") for r in records):
        raise ValueError("B0-B3 accepts source_fit/source_validation only")
    valid = [i for i, r in enumerate(records) if r.get("pairing_pass")]
    if not valid:
        raise RuntimeError("no pairing-pass pairs available")
    npz = np.load(args.pairs, allow_pickle=False)
    views = list(manifest["view_names"])

    matrix = json.loads(args.matrix.read_text())
    task = next(x for x in matrix["base_tasks"] if x["suite"] == manifest["suite"])
    variant = next(
        v for v in task["variants"]["lighting"]
        if int(v["classification_id"]) == int(manifest["lighting_id"])
    )
    import libero.libero.envs.bddl_utils as BDDLUtils
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]

    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    results = []
    for record_index in valid:
        record = records[record_index]
        row = int(record["feature_row"])
        if row != record_index:
            raise ValueError("manifest feature_row must equal record index for donor mapping")
        donor_index = random_clean_donor(records, record_index, args.random_seed)
        donor = records[donor_index] if donor_index is not None else None

        clean_patches = {
            view: npz[f"clean_{view}"][row] for view in views
        }
        shift_patches = {
            view: npz[f"shift_{view}"][row] for view in views
        }
        random_patches = None if donor is None else {
            view: npz[f"clean_{view}"][int(donor["feature_row"])] for view in views
        }

        with torch.no_grad():
            clean_case = _run_case(
                policy, clean_patches,
                npz["clean_language_tokens"][row],
                npz["clean_pooled_language"][row],
                npz["clean_state_token"][row],
                npz["z0"][row], device,
            )
            shift_case = _run_case(
                policy, shift_patches,
                npz["shift_language_tokens"][row],
                npz["shift_pooled_language"][row],
                npz["shift_state_token"][row],
                npz["z0"][row], device,
            )
            paired_replacement = _run_case(
                policy, clean_patches,
                npz["shift_language_tokens"][row],
                npz["shift_pooled_language"][row],
                npz["shift_state_token"][row],
                npz["z0"][row], device,
            )
            random_replacement = None
            if random_patches is not None:
                random_replacement = _run_case(
                    policy, random_patches,
                    npz["shift_language_tokens"][row],
                    npz["shift_pooled_language"][row],
                    npz["shift_state_token"][row],
                    npz["z0"][row], device,
                )

        a0 = clean_case["action_norm"]
        a1 = shift_case["action_norm"]
        a2 = paired_replacement["action_norm"]
        a3 = None if random_replacement is None else random_replacement["action_norm"]
        horizon = int(a0.shape[1])
        metrics_b1 = _one_action_metrics(policy, a1, a0, horizon)
        metrics_b2 = _one_action_metrics(policy, a2, a0, horizon)
        metrics_b3 = None if a3 is None else _one_action_metrics(policy, a3, a0, horizon)

        r0 = clean_case["visual"]
        r1 = shift_case["visual"]
        r2 = paired_replacement["visual"]
        r3 = None if random_replacement is None else random_replacement["visual"]
        resampler = {
            "B1_rms_to_B0": _rms(r1, r0),
            "B2_rms_to_B0": _rms(r2, r0),
            "B3_rms_to_B0": None if r3 is None else _rms(r3, r0),
            "B1_rms_to_B0_recovery": None,
            "B2_rms_to_B0_recovery": None,
            "B3_rms_to_B0_recovery": None,
        }
        resampler["B2_rms_to_B0_recovery"] = _recovery(
            resampler["B1_rms_to_B0"], resampler["B2_rms_to_B0"]
        )
        resampler["B3_rms_to_B0_recovery"] = (
            None if r3 is None else _recovery(
                resampler["B1_rms_to_B0"], resampler["B3_rms_to_B0"]
            )
        )
        results.append({
            "feature_row": row,
            "state_id": int(record["state_id"]),
            "role": record["role"],
            "window": record["window"],
            "chunk_index": int(record["chunk_index"]),
            "step": int(record["step"]),
            "pairing_pass": True,
            "random_donor_state_id": None if donor is None else int(donor["state_id"]),
            "random_donor_role": None if donor is None else donor["role"],
            "B1_vs_B0": metrics_b1,
            "B2_vs_B0": metrics_b2,
            "B3_vs_B0": metrics_b3,
            "resampler": resampler,
            "same_z0": True,
            "intervention_semantics": (
                "B2 replaces only the Shift visual patch tensors with the same-state "
                "Clean patch tensors, retaining Shift language/proprio state tokens. "
                "B3 uses a different-state Clean donor matched by role/window/chunk."
            ),
        })

    report = {
        "schema": "corp_paired_patch_replacement_v1",
        "pair_manifest": str(manifest_path),
        "pair_feature_file": str(args.pairs),
        "checkpoint": str(args.checkpoint),
        "suite": manifest["suite"],
        "lighting_id": int(manifest["lighting_id"]),
        "records_total": len(records),
        "records_pairing_pass": len(valid),
        "views": views,
        "per_record": results,
        "aggregates": _aggregate(results),
        "claim_boundary": (
            "B2/B3 are privileged offline mediator interventions. B2 recovery "
            "supports visual patch mediation but is not a deployable TTA signal. "
            "B3 tests whether same-phase random Clean content produces a similar "
            "change. No closed-loop success claim is made."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "records": len(results),
        "aggregates": report["aggregates"],
    }, indent=2))


if __name__ == "__main__":
    main()
