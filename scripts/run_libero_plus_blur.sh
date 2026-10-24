#!/usr/bin/env bash
# 4 suites 顺序跑，3 组 blur 由外部并行启动。
set -euo pipefail
BLUR="${BLUR:-0}"
PER="${PER_CATEGORY:-20}"
OUT="${OUT:-/hy-tmp/runs/libero_blur_${BLUR}_full}"
PY="${PY:-/hy-tmp/models/corp/CoRP/.venv/bin/python}"
mkdir -p "$OUT"
for suite in libero_spatial libero_object libero_goal libero_10; do
  echo "[blur=$BLUR] suite=$suite"
  MUJOCO_GL=egl "$PY" scripts/eval_corp_libero_plus.py \
    --suite "$suite" --per-category "$PER" --episodes-per-task 1 \
    --blur-sigma "$BLUR" --output-dir "$OUT/$suite" < /dev/null
done
"$PY" - "$OUT" "$BLUR" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
suites = {}
tot_n = tot_s = 0
for f in sorted(out.glob("*/summary.json")):
    d = json.loads(f.read_text())
    suites[d["suite"]] = d
    tot_n += d["episodes"]; tot_s += d["successes"]
(out / "summary.json").write_text(json.dumps({
    "blur_sigma": float(sys.argv[2]),
    "episodes": tot_n, "successes": tot_s,
    "success_rate": tot_s / tot_n if tot_n else 0.0,
    "suites": suites,
}, indent=2))
print(f"TOTAL {tot_s}/{tot_n} = {tot_s/tot_n if tot_n else 0:.3f}")
PY
echo "BLUR_${BLUR}_DONE"
