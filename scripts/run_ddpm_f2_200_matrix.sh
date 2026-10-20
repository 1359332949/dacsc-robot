#!/usr/bin/env bash
# DDPM phase-1 mechanism matrix at 200 steps:
#   no_mask / random 33% / D / F2  ×  seeds 42,43,44
# Uses the controlled DDPM baseline (train_dinov2_bs192), not flow.
set -uo pipefail
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export OTEL_SDK_DISABLED=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/hy-tmp/runs/ddpm_f2_200}"
STEPS="${STEPS:-200}"
SEEDS="${SEEDS:-42 43 44}"
GROUPS="${GROUPS:-no_mask random D F2}"
mkdir -p "$OUT"
cd "$ROOT"

COMMON=(
  dataloader.batch_size=30 dataloader.num_workers=4
  val_dataloader.batch_size=8 val_dataloader.num_workers=2
  training.num_epochs=1 training.max_train_steps="$STEPS"
  training.lr_warmup_steps=20 training.checkpoint_every=1000
  training.sample_every=1000 training.val_every=null
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
  echo "=== $group seed=$seed steps=$STEPS -> $dir ==="
  DACSC_CONFIG_NAME=train_dinov2_bs192 timeout 3600 \
    bash scripts/run_real_train.sh "${COMMON[@]}" \
    training.seed="$seed" task.dataset.seed="$seed" \
    hydra.run.dir="$dir" "$@" > "$dir/train.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$dir/.done"
  else
    echo "[FAIL] $group seed=$seed rc=$rc"
  fi
  grep -o 'GATE_METRICS {[^}]*}' "$dir/train.log" > "$dir/gate_metrics.jsonl" || true
  return $rc
}

for group in $GROUPS; do
  case "$group" in
    no_mask)
      for seed in $SEEDS; do
        run_one no_mask "$seed" policy.enable_gating=false
      done
      ;;
    random)
      for seed in $SEEDS; do
        run_one random "$seed" \
          policy.enable_gating=true policy.use_action_importance=false \
          policy.hard_domain_mode=random policy.hard_mask_domain_shared=false \
          policy.dgcd_group_smooth=0.0 policy.dgcd_domain_center=false \
          policy.hard_selection_mode=linear
      done
      ;;
    D)
      for seed in $SEEDS; do
        run_one D "$seed" \
          policy.enable_gating=true policy.use_action_importance=false \
          policy.hard_domain_mode=dgcd policy.hard_mask_domain_shared=true \
          policy.dgcd_group_smooth=1.0 policy.dgcd_domain_center=true \
          policy.layout_residual=true policy.hard_selection_mode=linear
      done
      ;;
    F2)
      for seed in $SEEDS; do
        run_one F2 "$seed" \
          policy.enable_gating=true policy.use_action_importance=true \
          policy.hard_domain_mode=dgcd policy.hard_mask_domain_shared=true \
          policy.dgcd_group_smooth=1.0 policy.dgcd_domain_center=true \
          policy.layout_residual=true policy.hard_selection_mode=two_stage \
          policy.hard_candidate_ratio=0.6 policy.hard_action_protect_ratio=0.2 \
          policy.hard_clean_importance_every=4
      done
      ;;
    *)
      echo "[unknown group] $group" >&2
      exit 2
      ;;
  esac
done
echo DDPM_F2_200_DONE
