#!/usr/bin/env bash
# Strict-split, unified-center-crop flow baseline: no-mask / random mask.
# DomainDrop is opt-in only after these baselines are stable.
set -uo pipefail
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export OTEL_SDK_DISABLED=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/hy-tmp/runs/strict_flow_centercrop_baseline_51}"
GROUP_LIST="${GROUP_LIST:-flow_no_mask flow_random}"
SEEDS="${SEEDS:-42 43 44 45 46}"
mkdir -p "$OUT"
cd "$ROOT"

COMMON=(
  dataloader.batch_size=30 dataloader.num_workers=4
  val_dataloader.batch_size=8 val_dataloader.num_workers=2
  training.num_epochs=1 training.max_train_steps=51
  training.lr_warmup_steps=20 training.checkpoint_every=1000
  training.sample_every=1000 training.val_every=50 training.val_every_steps=0 training.max_val_steps=16 training.max_test_steps=16
  checkpoint.save_last_ckpt=false checkpoint.topk.k=0
  task.dataset.domain_batch_size=30 logging.mode=disabled
  policy.hard_mask_mode=gumbel_ste
  policy.action_importance_impl=rsc
  "policy.hard_layer_indices=[11]"
)

run_one() {
  local group="$1" seed="$2"; shift 2
  local dir="$OUT/${group}_s${seed}"
  mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then
    echo "[skip] $group seed=$seed"
    return 0
  fi
  echo "=== $group seed=$seed -> $dir ==="
  DACSC_CONFIG_NAME=train_dinov2_flow WANDB_MODE=disabled \
    timeout 1800 bash scripts/run_real_train.sh "${COMMON[@]}" \
    training.seed="$seed" task.dataset.seed="$seed" \
    hydra.run.dir="$dir" "$@" > "$dir/train.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$dir/.done"; else echo "[FAIL] $group seed=$seed rc=$rc"; fi
  grep -o 'GATE_METRICS {[^}]*}' "$dir/train.log" > "$dir/gate_metrics.jsonl" || true
  return $rc
}

for group in $GROUP_LIST; do
  case "$group" in
    flow_no_mask)
      for seed in $SEEDS; do
        run_one flow_no_mask "$seed" \
          policy.enable_gating=false policy.use_action_importance=false \
          policy.hard_domain_mode=random policy.hard_mask_domain_shared=false \
          policy.dgcd_group_smooth=0.0 policy.dgcd_domain_center=false \
          policy.hard_selection_mode=linear
      done
      ;;
    flow_random)
      for seed in $SEEDS; do
        run_one flow_random "$seed" \
          policy.enable_gating=true policy.use_action_importance=false \
          policy.hard_domain_mode=random policy.hard_mask_domain_shared=false \
          policy.dgcd_group_smooth=0.0 policy.dgcd_domain_center=false \
          policy.hard_selection_mode=linear
      done
      ;;
    *)
      echo "[unknown group] $group" >&2
      exit 2
      ;;
  esac
done
echo STRICT_FLOW_BASELINE_DONE
