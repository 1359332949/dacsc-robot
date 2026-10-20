#!/usr/bin/env bash
# Mechanism seed matrix: random / D / F × seeds 42,43,44.
# Each run is 51 steps so the step50 gate on/off probe is logged.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/hy-tmp/runs/seed_matrix}"
mkdir -p "$OUT"
cd "$ROOT"

COMMON=(
  dataloader.batch_size=30 dataloader.num_workers=4
  val_dataloader.batch_size=8 val_dataloader.num_workers=2
  training.num_epochs=1 training.max_train_steps=51
  training.lr_warmup_steps=20 training.checkpoint_every=1000
  training.sample_every=1000 training.val_every=null checkpoint.topk.k=1
  task.dataset.domain_batch_size=30
  policy.hard_mask_mode=gumbel_ste
  policy.hard_domain_mode=dgcd
  policy.action_importance_impl=rsc
  "policy.hard_layer_indices=[11]"
)

run_one() {
  local group="$1" seed="$2"; shift 2
  local dir="$OUT/${group}_s${seed}"
  mkdir -p "$dir"
  if grep -q '"[[:space:]]*step"[[:space:]]*:[[:space:]]*50' "$dir/train.log" 2>/dev/null; then
    echo "[skip] $group seed=$seed"
    return 0
  fi
  echo "=== $group seed=$seed -> $dir ==="
  WANDB_MODE=disabled bash scripts/run_real_train.sh \
    "${COMMON[@]}" training.seed="$seed" task.dataset.seed="$seed" \
    hydra.run.dir="$dir" "$@" > "$dir/train.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[FAIL] $group seed=$seed rc=$rc"; fi
  grep -o 'GATE_METRICS {[^}]*}' "$dir/train.log" > "$dir/gate_metrics.jsonl" || true
  return $rc
}

for seed in 42 43 44; do
  run_one random "$seed" \
    policy.use_action_importance=false policy.hard_mask_domain_shared=false \
    policy.dgcd_group_smooth=0.0 policy.dgcd_domain_center=false \
    policy.hard_selection_mode=linear policy.hard_domain_mode=random
done

for seed in 42 43 44; do
  run_one D "$seed" \
    policy.use_action_importance=false policy.hard_mask_domain_shared=true \
    policy.dgcd_group_smooth=1.0 policy.dgcd_domain_center=true \
    policy.hard_selection_mode=linear
done

for seed in 42 43 44; do
  run_one F "$seed" \
    policy.use_action_importance=true policy.hard_mask_domain_shared=true \
    policy.dgcd_group_smooth=1.0 policy.dgcd_domain_center=true \
    policy.hard_selection_mode=two_stage policy.hard_candidate_ratio=0.6 \
    policy.hard_action_protect_ratio=0.2 policy.hard_clean_importance_every=4
done

echo MATRIX_DONE
