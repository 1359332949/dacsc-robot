"""Load and run a fail-closed ensemble of failure monitors."""

from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch

from dacsc.failure_monitor import CausalFailureMonitor


class MonitorEnsemble:
    def __init__(self, artifact_path: Path, base_dir: Path | None = None):
        self.artifact_path = Path(artifact_path)
        self.artifact = json.loads(self.artifact_path.read_text())
        if self.artifact.get("schema") != "monitor_artifact_v2":
            raise ValueError("unsupported monitor artifact")
        members = self.artifact.get("members", [])
        if len(members) < int(self.artifact.get("min_members", 2)):
            raise ValueError("not enough monitor members")
        base = base_dir or Path.cwd()
        self.members = []
        for m in members:
            p = Path(m["checkpoint"])
            if not p.exists(): p = base / p
            if hashlib.sha256(p.read_bytes()).hexdigest() != m["sha256"]:
                raise ValueError(f"member hash mismatch: {p}")
            ck = torch.load(p, map_location="cpu")
            model = CausalFailureMonitor(window_size=ck["window"], hidden_dim=ck["hidden"],
                                         use_visual=ck.get("use_visual", True))
            model.load_state_dict(ck["model"]); model.eval()
            mu = {k: np.asarray(v[0]) for k, v in ck["normalization"].items()}
            sd = {k: np.asarray(v[1]) for k, v in ck["normalization"].items()}
            self.members.append({"model": model, "mu": mu, "sd": sd,
                                 "window": int(ck["window"]), "name": m.get("name", p.stem)})

    def predict(self, state, action, visual):
        probs = []
        for m in self.members:
            s = ((state - m["mu"]["state"]) / m["sd"]["state"]).astype(np.float32)
            a = ((action - m["mu"]["action"]) / m["sd"]["action"]).astype(np.float32)
            v = ((visual - m["mu"]["visual_delta"]) / m["sd"]["visual_delta"]).astype(np.float32)
            with torch.no_grad():
                p = m["model"](torch.from_numpy(s), torch.from_numpy(a), torch.from_numpy(v)).numpy()
            probs.append(p)
        return np.mean(np.stack(probs), axis=0)
