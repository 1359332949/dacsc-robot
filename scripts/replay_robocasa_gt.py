"""用数据集 ground-truth action 复现 RoboCasa episode，验证 success-rate 管线。"""
from __future__ import annotations
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/dacsc-robot")
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
from robocasa.utils.env_utils import convert_action
from robocasa.scripts.dataset_scripts.label_subtasks import reset_to


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/hy-tmp/datasets/robocasa365/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--task", default="CloseFridge")
    ap.add_argument("--style", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    ds = Path(args.dataset)
    meta = json.loads((ds / "extras" / f"episode_{args.episode:06d}" / "ep_meta.json").read_text())
    states = np.load(ds / "extras" / f"episode_{args.episode:06d}" / "states.npz")["states"]
    df = pd.read_parquet(ds / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet")
    actions = np.stack(df["action"].to_numpy())  # modality order
    print(f"episode={args.episode} steps={len(actions)} states={states.shape} lang={meta.get('lang')}")

    style = args.style if args.style is not None else int(meta["style_id"])
    env = RoboCasaGymEnv(env_name=args.task, split=None, obj_instance_split="pretrain",
                         style_ids=[style], camera_widths=256, camera_heights=256)
    try:
        model_xml = gzip.open(ds / "extras" / f"episode_{args.episode:06d}" / "model.xml.gz", "rt").read()
        reset_to(env.env, {"model": model_xml, "states": states[0],
                           "ep_meta": json.dumps(meta)})
        print("sim state dim after model reset:", env.env.sim.get_state().flatten().shape,
              "recorded:", states[0].shape)
        success, steps = False, 0
        n = args.max_steps or len(actions)
        for i in range(n):
            a = actions[i]
            # modality order -> convert_action concat order
            reordered = np.concatenate([a[5:8], a[8:11], a[11:12], a[0:4], a[4:5]])
            obs, reward, done, trunc, info = env.step(convert_action(reordered.astype(np.float32)))
            steps += 1
            success = bool(info.get("success", False))
            if success or done:
                break
        print(f"GT replay success={int(success)} steps={steps} / {len(actions)}")
        return 0 if success else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
