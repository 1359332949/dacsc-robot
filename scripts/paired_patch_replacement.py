"""Offline B0-B3 intervention under known target lighting on source states.

The pair set renders the known target lighting configuration on source_fit/
source_validation initial states. Target-dev/Test initial states and outcomes
are forbidden. This is not source-free or unknown-target TTA.

B0: matched Clean patches + Clean context.
B1: matched Shift patches + Shift context.
B2: same-state Clean patches substituted into Shift context.
B3: different-state Clean patches, matched by role/window/chunk, into Shift context.

B2/B3 use privileged Clean patches and are offline causal diagnostics only.
They are not deployable TTA methods and must never be run on target-test data.
"""
from __future__ import annotations

import argparse
import hashlib
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
from target_only_action_validation import _one_action_metrics, configure  # noqa: E402
from dacsc.differentiable_policy import sample_from_z0  # noqa: E402
from dacsc.patch_diagnostics import random_clean_donor  # noqa: E402


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(array), device=device, dtype=dtype).unsqueeze(0)


def _max_abs_diff(a, b):
    x = torch.as_tensor(np.asarray(a), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(b), dtype=torch.float32)
    if x.shape != y.shape:
        return float("inf")
    return float((x - y).abs().max().item())


def _run_case(policy, patches, language_tokens, pooled_language, state_token, z0, device):
    extractor = policy.representation_extractor
    resampler_param = next(extractor.cross_attention_resampler.parameters())
    visual_input_dtype = resampler_param.dtype
    patch_dict = {
        view: _tensor(value, device=device, dtype=visual_input_dtype)
        for view, value in patches.items()
    }
    pooled = _tensor(pooled_language, device=device, dtype=visual_input_dtype)
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


def _state_balanced_mean(values, metric_key, nested_key=None):
    """Average chunks within each state, then average states equally."""
    by_state = {}
    for record in values:
        obj = record.get(metric_key)
        if obj is None:
            continue
        if nested_key is not None:
            obj = obj.get(nested_key)
        if obj is None:
            continue
        by_state.setdefault(int(record["state_id"]), []).append(obj)
    if not by_state:
        return None
    metric_names = tuple(next(iter(by_state.values()))[0].keys())
    state_means = {
        state: {
            key: float(np.mean([item[key] for item in items]))
            for key in metric_names
        }
        for state, items in by_state.items()
    }
    return {
        "n_states": len(state_means),
        "n_records": sum(len(v) for v in by_state.values()),
        "state_balanced_mean": {
            key: float(np.mean([metrics[key] for metrics in state_means.values()]))
            for key in metric_names
        },
        "per_state": state_means,
    }


def _aggregate(records):
    groups = {}
    for record in records:
        groups.setdefault((record["role"], record["window"]), []).append(record)
    out = {}
    for (role, window), values in groups.items():
        row = {
            "n_records": len(values),
            "n_states": len({int(v["state_id"]) for v in values}),
            "aggregation_unit": "state-balanced: average chunks within each state, then states",
        }
        for name in ("B1_vs_B0", "B2_vs_B0", "B3_vs_B0"):
            row[name] = _state_balanced_mean(values, name)
        row["resampler"] = {
            name: _state_balanced_mean(values, "resampler", name)
            for name in ("B1_rms_to_B0", "B2_rms_to_B0", "B3_rms_to_B0")
        }
        action_b1, action_b2, action_b3 = (
            row["B1_vs_B0"], row["B2_vs_B0"], row["B3_vs_B0"]
        )
        b1_mae = None if action_b1 is None else action_b1["state_balanced_mean"]["normalized_mae"]
        b2_mae = None if action_b2 is None else action_b2["state_balanced_mean"]["normalized_mae"]
        b3_mae = None if action_b3 is None else action_b3["state_balanced_mean"]["normalized_mae"]
        row["B2_clean_policy_action_consistency_recovery_fraction"] = _recovery(b1_mae, b2_mae)
        row["B3_clean_policy_action_consistency_recovery_fraction"] = _recovery(b1_mae, b3_mae)
        row["B2_beats_B3_on_clean_policy_action_consistency"] = bool(
            b2_mae is not None and b3_mae is not None and b2_mae < b3_mae
        )

        def resampler_mean(name):
            item = row["resampler"][name]
            return None if item is None else item["state_balanced_mean"]["value"]

        b1_res = resampler_mean("B1_rms_to_B0")
        b2_res = resampler_mean("B2_rms_to_B0")
        b3_res = resampler_mean("B3_rms_to_B0")
        row["B2_resampler_recovery_fraction"] = _recovery(b1_res, b2_res)
        row["B3_resampler_recovery_fraction"] = _recovery(b1_res, b3_res)
        out[f"{role}/{window}"] = row
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--phase1-report", type=Path, required=True,
                    help="Phase 1 report for this exact pair manifest and NPZ")
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--random-seed", type=int, default=20261119)
    ap.add_argument("--action-parity-tol", type=float, default=1e-5)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    manifest_path = args.manifest or args.pairs.with_suffix(".manifest.json")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    configure(args)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("feature_file_sha256") and _sha256_file(args.pairs) != manifest["feature_file_sha256"]:
        raise ValueError("NPZ feature hash does not match its manifest")
    if manifest.get("checkpoint_sha256") and _sha256_file(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint hash does not match the pair-capture manifest")
    if manifest.get("matrix_sha256") and _sha256_file(args.matrix) != manifest["matrix_sha256"]:
        raise ValueError("matrix hash does not match the pair-capture manifest")
    if manifest.get("schema") != "corp_paired_dino_patches_v1":
        raise ValueError("unexpected patch-pair manifest schema")
    phase1 = json.loads(args.phase1_report.read_text())
    if phase1.get("schema") != "corp_patch_phase1_report_v1":
        raise ValueError("unexpected Phase 1 report schema")
    if phase1.get("pair_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("Phase 1 report does not describe this pair manifest")
    if phase1.get("pair_feature_sha256") != _sha256_file(args.pairs):
        raise ValueError("Phase 1 report does not describe this NPZ feature file")
    if phase1.get("views") != manifest.get("view_names"):
        raise ValueError("Phase 1 report view list differs from pair manifest")
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
        clean_collection_parity = _max_abs_diff(
            a0[0].detach().cpu().numpy(), npz["clean_normalized_actions"][row]
        )
        shift_collection_parity = _max_abs_diff(
            a1[0].detach().cpu().numpy(), npz["shift_normalized_actions"][row]
        )
        if max(clean_collection_parity, shift_collection_parity) > args.action_parity_tol:
            raise RuntimeError(
                "B0/B1 do not reproduce collector actions: "
                f"state={record['state_id']} chunk={record['chunk_index']} "
                f"clean_max_abs={clean_collection_parity:.8g} "
                f"shift_max_abs={shift_collection_parity:.8g} "
                f"tol={args.action_parity_tol:.8g}"
            )
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
            "collector_action_parity": {
                "clean_b0_max_abs": clean_collection_parity,
                "shift_b1_max_abs": shift_collection_parity,
                "tolerance": args.action_parity_tol,
                "pass": True,
            },
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
        "pair_manifest_sha256": _sha256_file(manifest_path),
        "phase1_report": str(args.phase1_report),
        "phase1_report_sha256": _sha256_file(args.phase1_report),
        "pair_feature_file": str(args.pairs),
        "pair_feature_sha256": _sha256_file(args.pairs),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "matrix_sha256": _sha256_file(args.matrix),
        "analysis_script_sha256": _sha256_file(Path(__file__)),
        "suite": manifest["suite"],
        "lighting_id": int(manifest["lighting_id"]),
        "shift_domain_semantics": manifest.get("shift_domain_semantics"),
        "target_dev_test_initial_states_used": False,
        "target_success_labels_or_rewards_used": False,
        "action_reference_semantics": (
            "B0 is the frozen CoRP action on the paired Clean observation with "
            "the same z0. B2/B3 action metrics measure consistency with this "
            "Clean Frozen behavior, not expert correctness or task success."
        ),
        "records_total": len(records),
        "records_pairing_pass": len(valid),
        "action_parity_tolerance": args.action_parity_tol,
        "collector_action_parity_all_pass": True,
        "views": views,
        "per_record": results,
        "aggregates": _aggregate(results),
        "claim_boundary": (
            "B2/B3 are privileged offline mediator interventions under the known "
            "target lighting configuration rendered on source initial states. "
            "B2 recovery supports same-state patch mediation but is not a deployable "
            "TTA signal or unknown-target generalization result. B3 tests whether "
            "different-state Clean content produces a similar change. Action "
            "recovery means consistency with B0 Clean Frozen actions, not expert "
            "correctness. Aggregates are state-balanced; no target-dev/test "
            "states, success labels, or rewards are used, and no closed-loop "
            "success claim is made."
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
