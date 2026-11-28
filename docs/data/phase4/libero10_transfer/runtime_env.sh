cd /hy-tmp/patch-diag
export SOH_EVAL_OUT=/hy-tmp/patch-diag/runs/phase4_soh_v2_3seed
export LIBERO_CONFIG_PATH=/hy-tmp/vanilla_libero_config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYGLFW_LIBRARY=/hy-tmp/corp_venv/lib/python3.10/site-packages/glfw/x11/libglfw.so
export HF_HOME=/hy-tmp/upstream_repro/hf-home HF_HUB_CACHE=/hy-tmp/upstream_repro/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/hy-tmp/upstream_repro/harness_pkg:/hy-tmp/patch-diag/runs/repro_baseline_v2_20261126/author_audit/mujoco327:/hy-tmp/lerobot_community_revision:/hy-tmp/LIBERO-vanilla:/hy-tmp/smolvla_venv/lib/python3.10/site-packages:/hy-tmp/corp_venv/lib/python3.10/site-packages:/hy-tmp/patch-diag:/hy-tmp/patch-diag/scripts
