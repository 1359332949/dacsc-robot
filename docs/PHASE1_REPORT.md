# Phase 1 report: libero_10 Clean smoke + latency + baseline self-check

Status: **PASS-ish.** libero_10 runs: 9/10 Clean; latency table measured;
external baseline proxies finite. True ActMAD is **not** implemented yet; the
current arm is an ActMAD-style scale-adaptation proxy, clearly labeled.

## 1. libero_10 Clean smoke

10 official base tasks, `init_index=0`, one deterministic environment seed,
EGL, `env.noise=0`, settle 10, max_steps=520, FP32, H=E=12.

| # | Task | Success | Steps |
|---:|---|---:|---:|
| 0 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | ✅ | 221 |
| 1 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer... | ✅ | 214 |
| 2 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave... | ❌ | 520 |
| 3 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | ✅ | 359 |
| 4 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box... | ✅ | 223 |
| 5 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce... | ✅ | 288 |
| 6 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter... | ✅ | 232 |
| 7 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate... | ✅ | 211 |
| 8 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate... | ✅ | 217 |
| 9 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment... | ✅ | 167 |

**9/10 = 90%.** All actions finite. Initial-state, initial-image, and action
hashes are in `docs/data/phase1/libero10_clean_smoke.json`.
The one failure is task 2 (microwave mug), maxed at 520. Per Phase 1 rule this
is reported, not tuned.

## 2. Latency / memory (RTX 3090, batch 1, 30 timed iterations)

| Arm | Forward ms | x Frozen | Backward | Activation peak MB |
|---|---:|---:|---:|---:|
| Frozen | 32.78 | 1.00x | 0 | 205.7 |
| V1 block3 | 32.19 | 0.98x | 0 | 205.7 |
| Pixel align | 31.38 | 0.96x | 0 | 205.7 |
| Deep block12 | 23.82 | 0.73x | 0 | 205.7 |
| ActMAD-style proxy | 23.07 | 0.70x | 1 update | 205.7 |

Caveats:
- Peak is **activation peak after model load**, not total model memory.
- Deep/ActMAD numbers are faster than Frozen in this single run, which is
  counter-intuitive; this is likely warmup/clock variance, so do not use the
  xFrozen column until it is remeasured with repeated trials.
- ActMAD-style proxy update: 34.65 ms, one backward, scale change 0.131.
  It is a scale-adapter proxy, **not** the official ActMAD method.

## 3. External baseline self-check

| Arm | Finite action | Notes |
|---|---|---|
| Frozen | yes | action mean 0.3425 |
| V1 block3 | yes | mean 0.3425 |
| Pixel align | yes | image mean abs diff 0.711 |
| Deep block12 | yes | mean 0.3453 |
| ActMAD-style proxy | yes | one finite gradient update, scale changed |

## Decision

- libero_10 is usable: **9/10** Clean; no language/init/render blocker.
- Latency harness works, but the numbers need a multi-trial rerun before
  publication; current xFrozen values are not trustworthy.
- **True ActMAD must still be implemented** before the Table 4 external-baseline
  claim; the current arm is a proxy self-check only.
- Pixel and deep-block12 code paths execute without NaN.

Next: fix latency measurement (3 repeats, median), implement real ActMAD, then
Phase 2 main table.
