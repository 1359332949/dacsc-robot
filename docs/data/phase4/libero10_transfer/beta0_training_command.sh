cd /hy-tmp/patch-diag
tar -xzf /hy-tmp/beta0.tar.gz
rm -f runs/phase4_soh/soh_v2_beta0_ckpt.pt runs/phase4_soh/soh_v2_beta0_ckpt.json
export LIBERO_CONFIG_PATH=/hy-tmp/vanilla_libero_config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYGLFW_LIBRARY=/hy-tmp/corp_venv/lib/python3.10/site-packages/glfw/x11/libglfw.so
export HF_HOME=/hy-tmp/upstream_repro/hf-home HF_HUB_CACHE=/hy-tmp/upstream_repro/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/hy-tmp/upstream_repro/harness_pkg:/hy-tmp/patch-diag/runs/repro_baseline_v2_20261126/author_audit/mujoco327:/hy-tmp/lerobot_community_revision:/hy-tmp/LIBERO-vanilla:/hy-tmp/smolvla_venv/lib/python3.10/site-packages:/hy-tmp/corp_venv/lib/python3.10/site-packages:/hy-tmp/patch-diag:/hy-tmp/patch-diag/scripts
nohup /hy-tmp/smolvla_venv/bin/python -u scripts/phase4_soh_train.py --v2 --damping-beta 0.0 --subspace runs/phase4_soh/subspace_7dof_init0.npz --out runs/phase4_soh/soh_v2_beta0_ckpt.pt --tasks 10 --frames-per-task 32 --epochs 30 --steps 0 --lr 1e-3 --act-weight 1.0 --id-weight 0.5 > /tmp/beta0_train.log 2>&1 &
echo PID:$!
