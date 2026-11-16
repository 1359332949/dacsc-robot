"""Build the frozen multi-domain experiment matrix from LIBERO-Plus metadata.

No Gate code, no rollout.  This only selects planned pairs and records roles.
"""

from __future__ import annotations
import argparse, json, sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--bddl-root", type=Path, required=True)
    ap.add_argument("--init-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.classification.read_text())
    matrix = {"base_tasks": [], "domains": CATEGORIES, "state_split": {"dev": 30, "heldout": 20},
              "notes": ["Clean is paired per state; no Clean/Shift labels are used by any proxy.",
                        "Episode/state split must be fixed before any adaptation.",
                        "Roles are frozen before running."]}
    for suite, base, base_role in BASE_TASKS:
        entries = data.get(suite, [])
        base_entry = {"suite": suite, "base_task": base, "base_role": base_role, "variants": {}}
        for domain, category in CATEGORIES.items():
            cand = [e for e in entries if e.get("category") == category and
                    canonical_base_name(str(e.get("name", ""))) == base]
            cand = sorted(cand, key=lambda e: int(e.get("id", 0)))
            chosen = []
            for i, e in enumerate(cand[:2]):
                name = str(e["name"])
                base_bddl = args.bddl_root / suite / f"{base}.bddl"
                init = args.init_root / suite / f"{base}.pruned_init"
                chosen.append({
                    "classification_id": e.get("id"), "name": name,
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
