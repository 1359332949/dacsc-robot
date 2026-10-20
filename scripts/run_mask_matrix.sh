#!/usr/bin/env bash
# Mask-generation comparison on flow K48. Default seed=42; override SEEDS.
set -uo pipefail
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export OTEL_SDK_DISABLED=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/hy-tmp/runs/mask_matrix}"
SEEDS="${SEEDS:-42}"
mkdir -p "$OUT"
cd "$ROOT"

COMMON=(
  dataloader.batch_size=30 dataloader.num_workers=4
  val_dataloader.batch_size=8 val_dataloader.num_workers=2
  training.num_epochs=1 training.max_train_steps=51
  training.lr_warmup_steps=20 training.checkpoint_every=1000
  training.sample_every=1000 training.val_every=null
  checkpoint.save_last_ckpt=false checkpoint.topk.k=0
  task.dataset.domain_batch_size=30 logging.mode=disabled
  policy.use_action_importance=false
  policy.hard_selection_mode=linear
  policy.hard_mask_domain_shared=false
  policy.hard_layer_indices=[11]
)

run_one() {
  local name="$1" seed="$2"; shift 2
  local dir="$OUT/${name}_s${seed}"
  mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then echo "[skip] $name seed=$seed"; return 0; fi
  echo "=== $name seed=$seed -> $dir ==="
  DACSC_CONFIG_NAME=train_dinov2_flow timeout 1800 bash scripts/run_real_train.sh \
    "${COMMON[@]}" training.seed="$seed" task.dataset.seed="$seed" \
    hydra.run.dir="$dir" "$@" > "$dir/train.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && touch "$dir/.done" || echo "[FAIL] $name seed=$seed rc=$rc"
  grep -o 'GATE_METRICS {[^}]*}' "$dir/train.log" > "$dir/gate_metrics.jsonl" || true
  return $rc
}

for seed in $SEEDS; do
  run_one topk "$seed" policy.hard_mask_mode=topk_ste     policy.dgcd_group_smooth=1.0 policy.dgcd_domain_center=true
  run_one gumbel "$seed" policy.hard_mask_mode=gumbel_ste     policy.dgcd_group_smooth=1.0 policy.dgcd_domain_center=true
  # Closer to official DgCD: per-sample discrepancy, no group smoothing/center.
  run_one dgcd_wrs_raw "$seed" policy.hard_mask_mode=dgcd_wrs     policy.hard_domain_mode=dgcd policy.dgcd_group_smooth=0.0     policy.dgcd_domain_center=false policy.layout_residual=false     policy.hard_wrs_recover=true policy.hard_wrs_normalize=global
  # Our adapted score pipeline.
  run_one dgcd_wrs_adapted "$seed" policy.hard_mask_mode=dgcd_wrs     policy.hard_domain_mode=dgcd policy.dgcd_group_smooth=1.0     policy.dgcd_domain_center=true policy.layout_residual=true     policy.hard_wrs_recover=true policy.hard_wrs_normalize=global
  # DomainDrop-inspired classifier WRS; name reflects that it is not exact official code.
  run_one domaindrop_inspired_wrs "$seed" policy.hard_mask_mode=domaindrop_wrs     policy.hard_domain_mode=classifier policy.dgcd_group_smooth=1.0     policy.dgcd_domain_center=true policy.layout_residual=true     policy.hard_wrs_recover=false policy.hard_wrs_normalize=none
  # PLACE/DomainDrop-style random single layer.
  run_one gumbel_randomlayer "$seed" policy.hard_mask_mode=gumbel_ste     policy.dgcd_group_smooth=1.0 policy.dgcd_domain_center=true     policy.hard_layer_indices=null
  run_one dgcd_wrs_randomlayer "$seed" policy.hard_mask_mode=dgcd_wrs     policy.hard_domain_mode=dgcd policy.dgcd_group_smooth=1.0     policy.dgcd_domain_center=true policy.hard_wrs_recover=true     policy.hard_wrs_normalize=global policy.hard_layer_indices=null
done
echo MASK_MATRIX_DONE
