#!/usr/bin/env bash
# Baselines before any long training: no-mask ERM and DFR-like frozen-encoder head retraining.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/hy-tmp/runs/baseline_matrix}"
mkdir -p "$OUT"
cd "$ROOT"

COMMON=(
  dataloader.batch_size=30 dataloader.num_workers=4
  val_dataloader.batch_size=8 val_dataloader.num_workers=2
  training.num_epochs=1 training.max_train_steps=51
  training.lr_warmup_steps=20 training.checkpoint_every=1000
  training.sample_every=1000 training.val_every=null checkpoint.topk.k=1
  training.seed=42 task.dataset.seed=42
)

run_one() {
  local name="$1"; shift
  local dir="$OUT/$name"
  mkdir -p "$dir"
  echo "=== $name -> $dir ==="
  WANDB_MODE=disabled bash scripts/run_real_train.sh "${COMMON[@]}" \
    hydra.run.dir="$dir" "$@" > "$dir/train.log" 2>&1
  local rc=$?
  echo "$name rc=$rc"
  return $rc
}

run_one no_mask policy.enable_gating=false
run_one dfr_head policy.enable_gating=false policy.freeze_obs_encoder=true
echo BASELINE_MATRIX_DONE
