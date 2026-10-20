#!/usr/bin/env bash
# Faithful CNN dropout matrix on the official ResNet18ConvFiLM baseline.
# Groups: no dropout / random channel / DomainDrop / PLACE / DgCD.
set -uo pipefail
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export OTEL_SDK_DISABLED=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/hy-tmp/runs/cnn_dropout_matrix}"
STEPS="${STEPS:-51}"
SEEDS="${SEEDS:-42 43 44}"
GROUP_LIST="${GROUP_LIST:-none random domaindrop place dgcd}"
mkdir -p "$OUT"
cd "$ROOT"

COMMON=(
  dataloader.batch_size=192 dataloader.num_workers=4
  val_dataloader.batch_size=32 val_dataloader.num_workers=4
  training.num_epochs=1 training.max_train_steps="$STEPS"
  training.lr_warmup_steps=20 training.checkpoint_every=1000
  training.sample_every=1000 training.val_every=null
  checkpoint.save_last_ckpt=false checkpoint.topk.k=0
  task.dataset.domain_balance=true
  task.dataset.domain_batch_size=192
  task.dataset.domain_group_size=5
  task.dataset.domain_samples_per_group=32
  task.dataset.domain_max_groups_per_batch=6
  logging.mode=disabled
  policy.cnn_diagnostics_every=10
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
  DACSC_CONFIG_NAME=train_official_cnn_dropout timeout 3600 \
    bash scripts/run_real_train.sh "${COMMON[@]}" \
    training.seed="$seed" task.dataset.seed="$seed" \
    hydra.run.dir="$dir" "$@" > "$dir/train.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$dir/.done"
  else
    echo "[FAIL] $group seed=$seed rc=$rc"
  fi
  grep -o 'CNN_DROPOUT {[^}]*}' "$dir/train.log" > "$dir/cnn_dropout.jsonl" || true
  return $rc
}

for group in $GROUP_LIST; do
  case "$group" in
    none)
      for seed in $SEEDS; do
        run_one none "$seed" policy.cnn_dropout_mode=none
      done
      ;;
    random)
      for seed in $SEEDS; do
        run_one random "$seed" \
          policy.cnn_dropout_mode=place policy.cnn_layer_selection=fixed \
          "policy.cnn_layer_indices=[7]" policy.cnn_progressive=false \
          policy.cnn_drop_ratio=0.33 policy.cnn_recover=true
      done
      ;;
    domaindrop)
      for seed in $SEEDS; do
        run_one domaindrop "$seed" \
          policy.cnn_dropout_mode=domaindrop policy.cnn_layer_selection=fixed \
          "policy.cnn_layer_indices=[7]" policy.cnn_drop_ratio=0.33 \
          policy.cnn_num_domains=10 policy.cnn_domain_group_size=5 \
          policy.cnn_domain_lambda=1.0 policy.cnn_domain_grl=0.25
      done
      ;;
    place)
      for seed in $SEEDS; do
        run_one place "$seed" \
          policy.cnn_dropout_mode=place policy.cnn_layer_selection=random \
          policy.cnn_progressive=true policy.cnn_velocity=20.0 \
          policy.cnn_drop_ratio=0.33 policy.cnn_recover=true
      done
      ;;
    dgcd)
      for seed in $SEEDS; do
        run_one dgcd "$seed" \
          policy.cnn_dropout_mode=dgcd policy.cnn_layer_selection=fixed \
          "policy.cnn_layer_indices=[7]" policy.cnn_drop_ratio=0.33 \
          policy.cnn_recover=true
      done
      ;;
    *)
      echo "[unknown group] $group" >&2
      exit 2
      ;;
  esac
done
echo CNN_DROPOUT_MATRIX_DONE
