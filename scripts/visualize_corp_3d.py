"""Generate an interactive Three.js 3D view of the adapted CoRP graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dacsc.corp_robocasa import (  # noqa: E402
    ROBOCASA_CORP_CONFIG,
    TaskAuxConfig,
    load_corp_robocasa,
)
from scripts.visualize_corp import _action_velocity, _tensor_parts  # noqa: E402


class ModelvizCoRP(nn.Module):
    """Single-image tensor wrapper for modelviz's hook-based PyTorch parser."""

    def __init__(self, model: nn.Module, instruction: str) -> None:
        super().__init__()
        self.model = model
        self.instruction = instruction

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        images = {name: image for name in self.model.view_names}
        state = image.new_zeros(image.shape[0], self.model.state_dim)
        tokens, pooled_language = _tensor_parts(
            self.model, images, state, self.instruction
        )
        action = image.new_zeros(
            image.shape[0], self.model.action_horizon, self.model.action_dim
        )
        tau = image.new_zeros(image.shape[0])
        return _action_velocity(self.model, tokens, pooled_language, action, tau)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "repos/CoRP/checkpoints/corp_libero.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "audit/corp_model_visualization_3d",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--instruction", default="open the cabinet door")
    args = parser.parse_args()
    try:
        from modelviz import visualize_threejs
    except ImportError as exc:
        raise SystemExit(
            "install visualization dependencies: "
            "pip install -r requirements.visualization.txt"
        ) from exc

    torch.set_num_threads(4)
    model, load_report = load_corp_robocasa(
        args.checkpoint,
        config=ROBOCASA_CORP_CONFIG,
        device="cpu",
        task_aux=TaskAuxConfig(num_classes=61),
    )
    model.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    html_path = args.out / "corp_3d.html"
    visualize_threejs(
        ModelvizCoRP(model, args.instruction),
        input_shape=(1, 3, args.image_size, args.image_size),
        title="CoRP RoboCasa v5 - 3D",
        show_shapes=True,
        show_params=True,
        group_blocks=False,
        save_path=str(html_path),
    )
    metadata = {
        "html": str(html_path.resolve()),
        "bytes": html_path.stat().st_size,
        "load_report": load_report,
        "renderer": "modelviz-ai 0.1.0 / Three.js 0.160",
        "note": "Three.js modules are loaded from unpkg on first browser open.",
    }
    (args.out / "three_d_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
