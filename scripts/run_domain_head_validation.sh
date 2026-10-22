#!/usr/bin/env bash
# Validate a warm-started frozen right-agentview DINOv2 domain head.
# No mask, no GRL, no policy gradients into the domain branch.
set -uo pipefail
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true OTEL_SDK_DISABLED=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/hy-tmp/runs/domain_head_validation}"
STEPS="${STEPS:-51}"
SEEDS="${SEEDS:-42}"
CHECKPOINT="${CHECKPOINT:-/hy-tmp/runs/domain_head_warmstart.pt}"
mkdir -p "$OUT"
cd "$ROOT"

COMMON=(
  dataloader.batch_size=96 dataloader.num_workers=4
  val_dataloader.batch_size=16 val_dataloader.num_workers=2
  training.num_epochs=1 training.max_train_steps="$STEPS"
  training.lr_warmup_steps=20 training.checkpoint_every=1000
  training.sample_every=1000 training.val_every=null
  checkpoint.save_last_ckpt=false checkpoint.topk.k=0
  task.dataset.domain_balance=true
  task.dataset.domain_batch_size=96
  task.dataset.domain_group_size=20
  task.dataset.domain_samples_per_group=32
  task.dataset.domain_max_groups_per_batch=3
  logging.mode=disabled
  policy.enable_gating=false
  policy.token_style_mode=none
  policy.domain_head_checkpoint="$CHECKPOINT"
  policy.domain_head_feature=patch_mean
  policy.domain_head_log_every=5
  policy.domain_head_aux_weight=0.0
)

run_one() {
  local seed="$1"
  local dir="$OUT/domainhead_s${seed}"
  mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then echo "[skip] seed=$seed"; return 0; fi
  echo "=== domain-head validation seed=$seed steps=$STEPS -> $dir ==="
  DACSC_CONFIG_NAME=train_dinov2_flow timeout 3600 bash scripts/run_real_train.sh \
    "${COMMON[@]}" training.seed="$seed" task.dataset.seed="$seed" \
    hydra.run.dir="$dir" > "$dir/train.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$dir/.done"; else echo "[FAIL] seed=$seed rc=$rc"; fi
  grep -o 'DOMAIN_HEAD {[^}]*}' "$dir/train.log" > "$dir/domain_head.jsonl" || true
  return $rc
}

for seed in $SEEDS; do run_one "$seed"; done
echo DOMAIN_HEAD_VALIDATION_DONE
