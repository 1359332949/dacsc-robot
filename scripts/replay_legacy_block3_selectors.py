"""Replay historical *selection*, never policies, on immutable LIBERO-Plus metadata.

Source: project session 72603ae9, records below. Reimplementation, not execution of
archived shell commands. Outputs are conservative exclusions, never scoreable IDs.
"""

import argparse, ast, hashlib, json, random
from pathlib import Path

S = "libero_spatial"
O = "libero_object"
NO = "Sensor Noise"
CAM = "Camera Viewpoints"
BG = "Background Textures"
LI = "Light Conditions"
FP = {
    "task_classification.json": "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6",
    "libero_suite_task_map.py": "41640d2f542169b400be7e94249e3b1f862e7500745dc318bab49b19576b31ac",
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def replay(assets):
    for k, v in FP.items():
        if sha((assets / k).read_bytes()) != v:
            raise ValueError("asset version mismatch: " + k)
    cls = json.loads((assets / "task_classification.json").read_text())
    mp = ast.literal_eval(
        ast.parse((assets / "libero_suite_task_map.py").read_text()).body[0].value
    )
    for t in [S, O]:
        assert len(set(mp[t])) == len(mp[t]) == len(cls[t]) and all(
            mp[t][r["id"] - 1] == r["name"] for r in cls[t]
        )
    batches = {}

    def draw(t, cat, level, n, rng, *, skip=(), exact=False, shuffle=True):
        ids = [
            r["id"] - 1
            for r in cls[t]
            if r["category"] == cat
            and (
                (r.get("difficulty_level") or 0) == level
                if exact
                else (r.get("difficulty_level") or 0) >= level
            )
            and r["id"] - 1 not in skip
        ]
        if shuffle:
            rng.shuffle(ids)
            return ids[:n]
        return rng.sample(ids, min(n, len(ids)))

    def add(seq, name, t, ids):
        assert len(ids) == len(set(ids))
        batches[name] = dict(seq=seq, suite=t, ids=ids)
        return ids

    for t in [S, O]:
        rng = random.Random(7)
        add(
            9619,
            "hard_tasks_" + t,
            t,
            [
                i
                for cat in sorted((LI, BG, NO))
                for i in draw(t, cat, 4, 15, rng, shuffle=False)
            ],
        )
        add(
            9708,
            "hard_noise_" + t,
            t,
            draw(t, NO, 4, 20, random.Random(11), shuffle=False),
        )
    rng = random.Random(13)
    m3 = add(
        10097,
        "m3_tasks_spatial",
        S,
        [
            i
            for cat, lv in ((CAM, 5), (NO, 4))
            for i in draw(S, cat, lv, 20, rng, shuffle=False)
        ],
    )
    ns = add(
        10215,
        "noise60_spatial",
        S,
        draw(S, NO, 4, 60, random.Random(21), shuffle=False),
    )
    no = add(
        10338, "noise60_object", O, draw(O, NO, 4, 60, random.Random(31), shuffle=False)
    )
    rng = random.Random(41)
    c41 = {cat: draw(S, cat, lv, 8, rng, skip=m3) for cat, lv in ((CAM, 5), (NO, 4))}
    # Independent recorded stdout (10467), not values used to generate the draw.
    assert c41 == {
        CAM: [875, 968],
        NO: [1497, 1692, 1412, 1417, 1715, 1571, 1714, 1426],
    }, c41
    add(10463, "calib_tasks_spatial", S, c41[CAM] + c41[NO])
    rng = random.Random(43)
    add(10472, "calib_tasks_spatial_v2_noise", S, draw(S, NO, 4, 8, rng, skip=m3))
    add(10472, "calib_tasks_spatial_v2_camera", O, draw(O, CAM, 5, 8, rng))
    add(
        10480,
        "calib_noise_spatial",
        S,
        draw(S, NO, 4, 8, random.Random(47), skip=set(m3 + ns)),
    )
    add(10668, "calib_noise_object", O, draw(O, NO, 4, 8, random.Random(53), skip=no))
    for t, cat, seqs, seeds, sizes, names in [
        (
            S,
            NO,
            [11219, 11819, 11861, 12507],
            [61, 71, 81, 161],
            [20] * 4,
            ["noiseL3_spatial", "noiseL3_extra", "noiseL3_extra2", "noiseL3_extra3"],
        ),
        (
            O,
            CAM,
            [12173, 12215, 12507, 12541],
            [121, 131, 151, 171],
            [20, 20, 20, 60],
            ["camera_obj", "camera_obj2", "camera_obj3", "camera_obj_big"],
        ),
    ]:
        used = set()
        for q, seed, n, name in zip(seqs, seeds, sizes, names):
            used.update(
                add(
                    q,
                    name,
                    t,
                    draw(
                        t,
                        cat,
                        3 if t == S else 4,
                        n,
                        random.Random(seed),
                        skip=used,
                        exact=t == S and seed != 161,
                    ),
                )
            )
    # Two more complete ordered historical printouts (not the generation inputs).
    assert batches["calib_noise_spatial"]["ids"] == [1678, 1418, 1701, 1572, 1413, 1531]
    assert batches["calib_noise_object"]["ids"] == [
        1435,
        1751,
        1814,
        1398,
        1791,
        1498,
        1640,
        1693,
    ]
    # Missing salt-derived seed: each task might have been chosen. Test the
    # union over effective seeds, never pretend all possible IDs actually ran.
    sweep = {}
    for t in [S, O]:
        for cat in [BG, LI, NO]:
            ids = [r["id"] - 1 for r in cls[t] if r["category"] == cat]
            possible = set()
            for seed in range(10000):
                v = ids.copy()
                random.Random(seed).shuffle(v)
                possible.update(v[:30])
                if len(possible) == len(ids):
                    break
            sweep[t + "/" + cat] = dict(
                pool=len(ids), possible=len(possible), effective_seeds_examined=seed + 1
            )
    return cls, batches, sweep


def main(a):
    cls, batches, sweep = replay(a.assets)
    legacy = json.loads(a.audit.read_text())
    reg = json.loads(a.registry.read_text())
    assert sha(a.registry.read_bytes()) == legacy["registry_sha256"]
    exclude = set(legacy["excluded_group_ids"])
    by = {
        c["evaluator_index"]: c
        for c in reg["registry"]
        if c["suite"] == S and c["factor"] in ("lighting", "background", "sensor_noise")
    }
    assert len(by) == 901
    chosen = {i for v in batches.values() if v["suite"] == S for i in v["ids"]}
    assert all(
        by[i]["logical_task_name"] == cls[S][i]["name"] for i in chosen & set(by)
    )
    counts = {}
    for fac in ("lighting", "background", "sensor_noise"):
        rr = [(i, c["equivalence_group"]) for i, c in by.items() if c["factor"] == fac]
        before = {g for i, g in rr if g in exclude}
        new = {g for i, g in rr if i in chosen}
        counts[fac] = dict(
            groups=len(rr),
            previous=len(before),
            selector_union=len(new),
            new_exclusions=len(new - before),
            union=len(new | before),
            still_unresolved=len(rr) - len(new | before),
            certified=0,
        )
    out = dict(
        kind="reconstructed_selection_not_outcomes",
        source_session="72603ae9-4cd6-420f-a92c-2917ff9a1a6b",
        assets=FP,
        id_convention="0-based evaluator; classification_id = evaluator_id + 1",
        ordered_id_witnesses={"10467": 10, "10483": 6, "10671": 8},
        batches=batches,
        spatial_audit=counts,
        unresolved_sweep=dict(
            selector_seq=13721,
            counts_seq=13724,
            launch_seq=13729,
            result_summary_seq=13782,
            per_suite_per_factor=30,
            seed="hash((tag,s)) % 10000",
            possible_scope=sweep,
        ),
        scoreable=False,
    )
    a.out.parent.mkdir(exist_ok=True, parents=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    if a.export:
        a.export.mkdir(exist_ok=True, parents=True)
        for name, v in batches.items():
            rr = [
                dict(
                    suite=v["suite"],
                    task_id=i,
                    classification_id=i + 1,
                    logical_task_name=cls[v["suite"]][i]["name"],
                    category=cls[v["suite"]][i]["category"],
                    difficulty=cls[v["suite"]][i]["difficulty_level"],
                )
                for i in v["ids"]
            ]
            (a.export / (name + ".json")).write_text(json.dumps(rr, indent=2) + "\n")
    print(json.dumps(counts))
    print(
        "reconstructed lists",
        len(batches),
        "24 ordered IDs from 3 historical printouts matched",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("assets", type=Path)
    p.add_argument("registry", type=Path)
    p.add_argument("audit", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--export", type=Path)
    main(p.parse_args())
