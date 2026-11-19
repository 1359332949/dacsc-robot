"""Phase 1 analysis of paired DINO patches (Source Fit -> Source Validation).

Reports paired Clean/Shift sensitivity, Clean cross-state content variability,
self-similarity matrices, and held-out-state domain AUROC. The Shift samples
are from the known target lighting configuration rendered on source Fit/Val
initial states; Target-dev/Test initial states and outcomes are not accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


from dacsc.patch_diagnostics import (
    cosine_self_similarity,
    normalized_distance,
    token_descriptor,
)


def _aggregate_by_state(records, descriptors, role, window, domain):
    grouped = {}
    for record in records:
        if record["role"] != role or record["window"] != window:
            continue
        key = int(record["state_id"])
        row = int(record["feature_row"])
        grouped.setdefault(key, []).append(descriptors[row][domain])
    return {state: np.mean(values, axis=0) for state, values in grouped.items()}


def _bootstrap_auc_by_state(state_scores, seed=20261119, n_boot=2000):
    states = sorted(state_scores)
    if len(states) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(states, size=len(states), replace=True)
        y, score = [], []
        for state in sampled:
            clean_score, shift_score = state_scores[int(state)]
            y.extend([0, 1])
            score.extend([clean_score, shift_score])
        if len(set(y)) == 2:
            values.append(float(roc_auc_score(y, score)))
    if not values:
        return None
    return {
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
        "bootstrap_samples": len(values),
        "unit": "initial_state_cluster",
    }


def _mean_self_similarity(npz, records, views, window, role):
    result = {}
    for view in views:
        for domain in ("clean", "shift"):
            matrices = []
            for record in records:
                if record["role"] != role or record["window"] != window:
                    continue
                row = int(record["feature_row"])
                matrices.append(cosine_self_similarity(npz[f"{domain}_{view}"][row]))
            if matrices:
                shapes = {m.shape for m in matrices}
                if len(shapes) != 1:
                    raise ValueError(f"self-similarity shape changes for {view}/{window}/{domain}")
                result[f"selfsim_{domain}_{view}_{window}"] = np.mean(matrices, axis=0).astype(np.float32)
    return result


def analyze_view_window(npz, records, view, window, seed):
    # feature_row is an index into the original NPZ. Pairing-failed rows are
    # excluded from records, so never assume list position == feature_row.
    descriptors = {}
    for record in records:
        row = int(record["feature_row"])
        descriptors[row] = {
            "clean": token_descriptor(npz[f"clean_{view}"][row]),
            "shift": token_descriptor(npz[f"shift_{view}"][row]),
        }

    fit = _aggregate_by_state(records, descriptors, "source_fit", window, "clean")
    fit_shift = _aggregate_by_state(records, descriptors, "source_fit", window, "shift")
    val = _aggregate_by_state(records, descriptors, "source_validation", window, "clean")
    val_shift = _aggregate_by_state(records, descriptors, "source_validation", window, "shift")
    fit_ids = sorted(set(fit) & set(fit_shift))
    val_ids = sorted(set(val) & set(val_shift))
    if not fit_ids:
        return {"status": "insufficient_source_fit", "source_fit_states": 0}
    if not val_ids:
        return {"status": "insufficient_source_validation", "source_fit_states": len(fit_ids)}

    fit_records = [
        r for r in records
        if r["role"] == "source_fit" and r["window"] == window
    ]
    val_records = [
        r for r in records
        if r["role"] == "source_validation" and r["window"] == window
    ]
    # The scale/prototype is estimated from one aggregated Clean descriptor
    # per Source Fit initial state, so states with more chunks do not dominate.
    fit_clean = np.stack([fit[s] for s in fit_ids])
    scale = np.maximum(fit_clean.std(axis=0, ddof=0), 1e-3)

    fit_pair_by_state = {state: [] for state in fit_ids}
    for record in fit_records:
        row = int(record["feature_row"])
        fit_pair_by_state[int(record["state_id"])].append(
            normalized_distance(
                descriptors[row]["clean"], descriptors[row]["shift"], scale
            )
        )
    val_pair_by_state = {state: [] for state in val_ids}
    for record in val_records:
        row = int(record["feature_row"])
        val_pair_by_state[int(record["state_id"])].append(
            normalized_distance(
                descriptors[row]["clean"], descriptors[row]["shift"], scale
            )
        )

    # Equal weight per initial state; chunks are repeated measures within a
    # state and are averaged before the state-level mean.
    pair_dists_fit = {
        state: float(np.mean(values))
        for state, values in fit_pair_by_state.items() if values
    }
    pair_dists_val = {
        state: float(np.mean(values))
        for state, values in val_pair_by_state.items() if values
    }

    # Clean-Clean content distance is computed between different initial
    # states at the same chunk index, so both distances share the same
    # record/chunk-level manipulation phase.
    clean_by_chunk = {}
    for record in fit_records:
        row = int(record["feature_row"])
        clean_by_chunk.setdefault(int(record["chunk_index"]), {})[
            int(record["state_id"])
        ] = descriptors[row]["clean"]
    content_by_chunk = {}
    for chunk, state_map in clean_by_chunk.items():
        states = sorted(state_map)
        chunk_dists = []
        for i, state_a in enumerate(states):
            for state_b in states[i + 1:]:
                chunk_dists.append(
                    normalized_distance(state_map[state_a], state_map[state_b], scale)
                )
        if chunk_dists:
            content_by_chunk[chunk] = float(np.mean(chunk_dists))
    content_dists = list(content_by_chunk.values())


    x_train, y_train = [], []
    x_val, y_val = [], []
    state_to_val_scores = {}
    for state in fit_ids:
        x_train.extend([fit[state], fit_shift[state]])
        y_train.extend([0, 1])
    for state in val_ids:
        x_val.extend([val[state], val_shift[state]])
        y_val.extend([0, 1])
    x_train = np.stack(x_train)
    x_val = np.stack(x_val)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=seed),
    )
    classifier.fit(x_train, y_train)
    val_prob = classifier.predict_proba(x_val)[:, 1]
    grouped_auc = float(roc_auc_score(y_val, val_prob))
    for i, state in enumerate(val_ids):
        state_to_val_scores[state] = (float(val_prob[2 * i]), float(val_prob[2 * i + 1]))

    # Source Fit Clean prototype: distance-to-prototype is a separate,
    # transparent baseline; it is not substituted for the held-out classifier.
    prototype = fit_clean.mean(axis=0)
    proto_scores, proto_labels = [], []
    for state in val_ids:
        proto_scores.extend([
            normalized_distance(val[state], prototype, scale),
            normalized_distance(val_shift[state], prototype, scale),
        ])
        proto_labels.extend([0, 1])
    prototype_auc = float(roc_auc_score(proto_labels, proto_scores))

    d_style = float(np.mean(list(pair_dists_fit.values())))
    d_style_val = float(np.mean(list(pair_dists_val.values())))
    d_content = float(np.mean(content_dists)) if content_dists else None
    r_sep = float(d_style / d_content) if d_content is not None and d_content > 1e-12 else None
    return {
        "status": "ok",
        "source_fit_states": len(fit_ids),
        "source_validation_states": len(val_ids),
        "source_fit_records": sum(
            r["role"] == "source_fit" and r["window"] == window for r in records
        ),
        "source_validation_records": sum(
            r["role"] == "source_validation" and r["window"] == window for r in records
        ),
        "paired_shift_sensitivity": {
            "D_style_fit_state_balanced_whitened_rms": d_style,
            "D_style_validation_state_balanced_whitened_rms": d_style_val,
            "source_fit_states": len(pair_dists_fit),
            "source_validation_states": len(pair_dists_val),
            "source_fit_records": len(fit_records),
            "source_validation_records": len(val_records),
            "aggregation": "average chunks within initial state, then average states",
        },
        "clean_content_variability": {
            "D_content_fit_chunk_balanced_whitened_rms": d_content,
            "matched_chunk_state_pairs_total": int(sum(
                len(state_map) * (len(state_map) - 1) // 2
                for state_map in clean_by_chunk.values()
            )),
            "matched_chunk_indices": len(content_dists),
            "aggregation": "average cross-state pairs within chunk, then average chunks",
        },
        "R_sep": r_sep,
        "heldout_state_domain_separation": {
            "logistic_regression_AUROC": grouped_auc,
            "logistic_regression_AUROC_CI95_state_bootstrap":
                _bootstrap_auc_by_state(state_to_val_scores, seed=seed),
            "source_clean_prototype_distance_AUROC": prototype_auc,
            "classifier_train_unit": "paired observations grouped by source_fit state",
            "classifier_test_unit": "paired observations grouped by source_validation state",
            "source_fit_validation_state_overlap": False,
        },
        "interpretation_guard": (
            "AUROC measures domain information/generalization, not causal "
            "responsibility or action benefit."
        ),
        "patch_tokens_per_sample": int(npz[f"clean_{view}"].shape[1]),
        "embedding_dim": int(npz[f"clean_{view}"].shape[2]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True, help="NPZ from collect_patch_pairs.py")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True, help="JSON report path")
    ap.add_argument("--self-similarity-out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args()
    manifest_path = args.manifest or args.pairs.with_suffix(".manifest.json")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "corp_paired_dino_patches_v1":
        raise ValueError("unexpected patch-pair manifest schema")
    expected_feature_hash = manifest.get("feature_file_sha256")
    if expected_feature_hash and _sha256_file(args.pairs) != expected_feature_hash:
        raise ValueError("NPZ feature hash does not match its manifest")
    records = manifest.get("records", [])
    roles = {r.get("role") for r in records}
    if roles - {"source_fit", "source_validation"}:
        raise ValueError(f"non-source records forbidden in Phase 1: {sorted(roles)}")
    fit_states = {int(r["state_id"]) for r in records if r["role"] == "source_fit"}
    val_states = {int(r["state_id"]) for r in records if r["role"] == "source_validation"}
    if fit_states & val_states:
        raise ValueError("source_fit/source_validation state overlap")
    valid_records = [r for r in records if r.get("pairing_pass")]
    if not valid_records:
        raise RuntimeError("no pairing-pass records; refusing to report Phase 1 metrics")

    npz = np.load(args.pairs, allow_pickle=False)
    views = list(manifest["view_names"])
    windows = sorted({r["window"] for r in valid_records})
    summaries = {}
    self_similarity = {}
    for view in views:
        summaries[view] = {}
        for window in windows:
            summaries[view][window] = analyze_view_window(
                npz, valid_records, view, window, args.seed
            )
            self_similarity.update(
                _mean_self_similarity(npz, valid_records, [view], window, "source_fit")
            )

    ss_path = args.self_similarity_out or args.out.with_suffix(".self_similarity.npz")
    if ss_path.exists():
        raise FileExistsError(f"refusing to overwrite {ss_path}")
    if self_similarity:
        np.savez_compressed(ss_path, **self_similarity)

    report = {
        "schema": "corp_patch_phase1_report_v1",
        "pair_manifest": str(manifest_path),
        "pair_manifest_sha256": _sha256_file(manifest_path),
        "pair_feature_file": str(args.pairs),
        "pair_feature_sha256": _sha256_file(args.pairs),
        "analyzer_script_sha256": _sha256_file(Path(__file__)),
        "records_total": len(records),
        "records_pairing_pass": len(valid_records),
        "source_fit_state_ids": sorted(fit_states),
        "source_validation_state_ids": sorted(val_states),
        "views": views,
        "token_layout": manifest.get("token_layout"),
        "patch_grid_hw": manifest.get("patch_grid_hw"),
        "patch_token_count_per_view": manifest.get("patch_token_count_per_view"),
        "prefix_token_count": manifest.get("prefix_token_count"),
        "visual_encoder_token_count_per_view": manifest.get("visual_encoder_token_count_per_view"),
        "windows": windows,
        "per_view": summaries,
        "self_similarity_file": str(ss_path) if self_similarity else None,
        "state_role_semantics": (
            "source_fit/source_validation identify disjoint initial-state partitions"
        ),
        "shift_domain_semantics": manifest.get(
            "shift_domain_semantics",
            "known target lighting rendered on source initial states",
        ),
        "target_dev_test_initial_states_used": False,
        "target_success_labels_or_rewards_used": False,
        "decision_boundary": (
            "Known-perturbation offline diagnosis only. High AUROC supports "
            "separable lighting information that generalizes across held-out "
            "source initial states; it does not establish unknown-target "
            "generalization, causal responsibility, or closed-loop benefit. "
            "Low AUROC does not prove a downstream bottleneck."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
