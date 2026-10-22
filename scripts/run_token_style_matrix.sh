#!/usr/bin/env bash
# Token-level style-statistics matrix on flow K48:
#   none / random_stats / cross_style_stats / action_aware × seeds
# Hard gating is disabled so this isolates the token-style modulator.
set -uo pipefail
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export OTEL_SDK_DISABLED=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/hy-tmp/runs/token_style_matrix}"
STEPS="${STEPS:-201}"
SEEDS="${SEEDS:-42 43 44}"
GROUP_LIST="${GROUP_LIST:-none random cross action}"
mkdir -p "$OUT"
cd "$ROOT"

COMMON=(
  dataloader.batch_size=30 dataloader.num_workers=4
  val_dataloader.batch_size=8 val_dataloader.num_workers=2
  training.num_epochs=1 training.max_train_steps="$STEPS"
  training.lr_warmup_steps=20 training.checkpoint_every=1000
  training.sample_every=1000 training.val_every=50 training.val_every_steps=50 training.max_val_steps=4 training.max_test_steps=4
  checkpoint.save_last_ckpt=false checkpoint.topk.k=0
  task.dataset.domain_batch_size=30 logging.mode=disabled
  policy.enable_gating=false
  policy.token_style_diagnostics_every=50
  policy.token_style_alpha=0.5
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
  DACSC_CONFIG_NAME=train_dinov2_flow timeout 3600 \
    bash scripts/run_real_train.sh "${COMMON[@]}" \
    training.seed="$seed" task.dataset.seed="$seed" \
    hydra.run.dir="$dir" "$@" > "$dir/train.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$dir/.done"
  else
    echo "[FAIL] $group seed=$seed rc=$rc"
  fi
  grep -o 'TOKEN_STYLE {[^}]*}' "$dir/train.log" > "$dir/token_style.jsonl" || true
  return $rc
}

for group in $GROUP_LIST; do
  case "$group" in
    none)
      for seed in $SEEDS; do
        run_one none "$seed" policy.token_style_mode=none
      done
      ;;
    random)
      for seed in $SEEDS; do
        run_one random "$seed" policy.token_style_mode=random_stats
      done
      ;;
    cross)
      for seed in $SEEDS; do
        run_one cross "$seed" policy.token_style_mode=cross_style_stats
      done
      ;;
    action)
      for seed in $SEEDS; do
        run_one action "$seed" \
          policy.token_style_mode=cross_style_stats_action_aware \
          policy.token_style_action_weight=1.0 \
          policy.token_style_protect_ratio=0.2 \
          policy.token_style_importance_every=4 \
          policy.token_style_momentum=0.9
      done
      ;;
    action30)
      for seed in $SEEDS; do
        run_one action30 "$seed" \
          policy.token_style_mode=cross_style_stats_action_aware \
          policy.token_style_action_weight=1.0 \
          policy.token_style_protect_ratio=0.3 \
          policy.token_style_importance_every=4 \
          policy.token_style_momentum=0.9
      done
      ;;
    mean)
      for seed in $SEEDS; do
        run_one mean "$seed" \
          policy.token_style_mode=cross_style_stats \
          policy.token_style_stats=mean
      done
      ;;
    patch25)
      for seed in $SEEDS; do
        run_one patch25 "$seed" \
          policy.token_style_mode=cross_style_stats \
          policy.token_style_stage=patch policy.token_style_alpha=0.25
      done
      ;;
    patch50)
      for seed in $SEEDS; do
        run_one patch50 "$seed" \
          policy.token_style_mode=cross_style_stats \
          policy.token_style_stage=patch policy.token_style_alpha=0.5
      done
      ;;
    patchmean)
      for seed in $SEEDS; do
        run_one patchmean "$seed" \
          policy.token_style_mode=cross_style_stats \
          policy.token_style_stage=patch policy.token_style_stats=mean \
          policy.token_style_alpha=0.25
      done
      ;;
    norm_res)
      for seed in $SEEDS; do
        run_one norm_res "$seed" policy.token_style_mode=normalize
      done
      ;;
    norm_patch)
      for seed in $SEEDS; do
        run_one norm_patch "$seed" \
          policy.token_style_mode=normalize policy.token_style_stage=patch
      done
      ;;
    norm_action)
      for seed in $SEEDS; do
        run_one norm_action "$seed" \
          policy.token_style_mode=normalize_action_aware \
          policy.token_style_action_weight=1.0 \
          policy.token_style_protect_ratio=0.2 \
          policy.token_style_importance_every=4
      done
      ;;
    *)
      echo "[unknown group] $group" >&2
      exit 2
      ;;
  esac
done
echo TOKEN_STYLE_MATRIX_DONE
