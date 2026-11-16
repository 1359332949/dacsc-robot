"""Build the frozen multi-domain experiment matrix from LIBERO-Plus metadata.

No Gate code, no rollout.  This only selects planned pairs and records roles.
"""

from __future__ import annotations
import argparse, json, sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.task_pairing import canonical_base_name

CATEGORIES = {
    "background": "Background Textures",
    "lighting": "Light Conditions",
    "sensor": "Sensor Noise",
    "camera": "Camera Viewpoints",
}
BASE_TASKS = [
    ("libero_spatial", "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate", "dev_task"),
    ("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket", "dev_task"),
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet", "heldout_task"),
    ("libero_10", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "heldout_task"),
]
ROLE = {
    "background": ("dev_domain", "heldout_same_domain"),
    "lighting": ("dev_domain", "heldout_same_domain"),
    "sensor": ("heldout_unseen_category", "heldout_unseen_category"),
    "camera": ("heldout_unseen_category", "heldout_unseen_category"),
}


def variant_params(domain, name):
    import re
    if domain == "camera":
        m = re.search(r"_view_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_initstate_", name)
        return tuple(int(x) for x in m.groups()) if m else None
    if domain == "sensor":
        m = re.search(r"_noise_(\d+)", name)
        return int(m.group(1)) if m else None
    m = re.search(r"_(?:table|light)_(\d+)", name)
    return int(m.group(1)) if m else None


def select_variants(domain, cand):
    """Pick two variants with maximal parameter separation, not first IDs."""
    if len(cand) <= 2:
        return cand, "all_available"
    best = None
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            pi, pj = variant_params(domain, str(cand[i].get("name", ""))), variant_params(domain, str(cand[j].get("name", "")))
            if pi is None or pj is None:
                dist = 0
            elif isinstance(pi, tuple):
                dist = sum(abs(a - b) for a, b in zip(pi, pj))
            else:
                dist = abs(int(pi) - int(pj))
            key = (dist, str(cand[i].get("name")), str(cand[j].get("name")))
            if best is None or key > best[0]:
                best = (key, i, j)
    if best is None:
        return cand[:2], "first_two"
    return [cand[best[1]], cand[best[2]]], "max_parameter_separation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--bddl-root", type=Path, required=True)
    ap.add_argument("--init-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--state-seed", type=int, default=1234)
    ap.add_argument("--dev-count", type=int, default=30)
    ap.add_argument("--heldout-count", type=int, default=20)
    args = ap.parse_args()
    data = json.loads(args.classification.read_text())
    matrix = {"base_tasks": [], "domains": CATEGORIES,
              "state_split": {"seed": args.state_seed, "dev": args.dev_count,
                              "heldout": args.heldout_count},
              "notes": ["Clean is paired per state; no Clean/Shift labels are used by any proxy.",
                        "Episode/state split must be fixed before any adaptation.",
                        "Roles are frozen before running."]}
    for suite, base, base_role in BASE_TASKS:
        entries = data.get(suite, [])
        base_entry = {"suite": suite, "base_task": base, "base_role": base_role,
                      "variants": {}, "state_split": None}
        for domain, category in CATEGORIES.items():
            cand = [e for e in entries if e.get("category") == category and
                    canonical_base_name(str(e.get("name", ""))) == base]
            cand = sorted(cand, key=lambda e: int(e.get("id", 0)))
            if domain == "sensor":
                motion = []
                for e in cand:
                    import re
                    m = re.search(r"_noise_(\d+)", str(e.get("name", "")))
                    if m and int(m.group(1)) <= 10:
                        motion.append(e)
                if len(motion) >= 2:
                    cand = motion
            chosen = []
            picked, selection_rule = select_variants(domain, cand)
            for i, e in enumerate(picked):
                name = str(e["name"])
                base_bddl = args.bddl_root / suite / f"{base}.bddl"
                init = args.init_root / suite / f"{base}.pruned_init"
                chosen.append({
                    "classification_id": e.get("id"), "name": name,
                    "composite_key": f"{suite}:{e.get('id')}:{name}",
                    "category": category, "difficulty": e.get("difficulty_level"),
                    "role": ROLE[domain][min(i, 1)],
                    "base_bddl": str(base_bddl), "base_bddl_exists": base_bddl.is_file(),
                    "base_init": str(init), "base_init_exists": init.is_file(),
                    "env_bddl_name": str(args.bddl_root / suite / f"{name}.bddl"),
                    "variant_kind": ("virtual_noise" if domain == "sensor" else
                                     "virtual_view" if domain == "camera" else "bddl"),
                    "actual_bddl_exists": (args.bddl_root / suite / f"{name}.bddl").is_file(),
                    "note": ("sensor noise is applied in ControlEnv.step; raw sim.render bypasses it"
                             if domain == "sensor" else
                             "Camera _view_ BDDL is virtual; env derives base BDDL"
                             if domain == "camera" else "actual variant BDDL file"),
                })
            base_entry["variants"][domain] = chosen
            base_entry.setdefault("selection_rule", {})[domain] = selection_rule
        init = args.init_root / suite / f"{base}.pruned_init"
        if init.is_file():
            states = torch.load(init, weights_only=False)
            n = len(states)
            rng = __import__("numpy").random.RandomState(args.state_seed)
            perm = rng.permutation(n)
            dev = sorted(perm[:args.dev_count].tolist())
            held = sorted(perm[args.dev_count:args.dev_count + args.heldout_count].tolist())
            base_entry["state_split"] = {
                "source_file": str(init), "source_sha256": __import__("hashlib").sha256(init.read_bytes()).hexdigest(),
                "n_total": n, "seed": args.state_seed, "dev_indices": dev,
                "heldout_indices": held, "episode_groups": "one state = one episode-init unit for Phase A"}
        matrix["base_tasks"].append(base_entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(matrix, indent=2))
    counts = {}
    for b in matrix["base_tasks"]:
        for d, vs in b["variants"].items():
            for v in vs:
                counts[v["role"]] = counts.get(v["role"], 0) + 1
    print(json.dumps({"pairs": sum(len(v) for b in matrix["base_tasks"] for v in b["variants"].values()),
                      "roles": counts, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
