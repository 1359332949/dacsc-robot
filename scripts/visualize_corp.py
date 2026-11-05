"""Visualize the adapted CoRP PyTorch graph with Torchview and torchinfo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torchinfo import summary
from torchview import draw_graph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dacsc.corp_robocasa import (  # noqa: E402
    ROBOCASA_CORP_CONFIG,
    TaskAuxConfig,
    load_corp_robocasa,
)


def _tensor_parts(
    model: nn.Module,
    images: dict[str, torch.Tensor],
    state: torch.Tensor,
    instruction: str,
):
    """Execute the official extractor children without exposing its dataclass output."""
    extractor = model.representation_extractor
    batch = state.shape[0]
    patches = extractor.visual_encoder(
        torch.cat([images[name] for name in extractor.view_names], dim=0)
    )
    patch_tokens = {
        name: patches[index * batch : (index + 1) * batch]
        for index, name in enumerate(extractor.view_names)
    }
    language_tokens, pooled_language = extractor.language_encoder(instruction)
    visual_tokens = extractor.cross_attention_resampler(patch_tokens, pooled_language)
    visual_language = torch.cat(
        [visual_tokens, language_tokens.to(visual_tokens.dtype)], dim=1
    )
    state_token = extractor.state_encoder(state)
    tokens = torch.cat([visual_language, state_token], dim=1)
    return tokens, pooled_language


def _action_velocity(
    model: nn.Module,
    tokens: torch.Tensor,
    pooled_language: torch.Tensor,
    action: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Unroll the official DiT children without exposing PrefixCache to graph tools."""
    generator = model.action_generator
    caches = []
    prefix_mask = torch.ones(
        tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device
    )[None, None]
    for block in generator.blocks:
        adaln = block.adaln(pooled_language)
        query, key, value = block.attention.project_qkv(block.attention_norm(tokens))
        tokens = tokens + block.attention.attend(query, key, value, prefix_mask)
        tokens = tokens + block.mlp(block.mlp_norm(tokens))
        caches.append((key, value, adaln))

    action_tokens = (
        generator.action_projection(action)
        + generator.action_position[:, : action.shape[1]]
        + generator.time_embedding(tau).unsqueeze(1)
    )
    for block, (prefix_key, prefix_value, adaln) in zip(
        generator.blocks, caches, strict=True
    ):
        scale_attn, shift_attn, scale_mlp, shift_mlp = adaln
        attention_input = block.adaln.modulate(
            block.attention_norm(action_tokens), scale_attn, shift_attn
        )
        query, key, value = block.attention.project_qkv(attention_input)
        key = torch.cat([prefix_key, key], dim=2)
        value = torch.cat([prefix_value, value], dim=2)
        action_tokens = action_tokens + block.attention.attend(query, key, value)
        mlp_input = block.adaln.modulate(
            block.mlp_norm(action_tokens), scale_mlp, shift_mlp
        )
        action_tokens = action_tokens + block.mlp(mlp_input)
    return generator.velocity_projection(generator.output_norm(action_tokens))


class TensorOnlyCorpGraph(nn.Module):
    """Keep text fixed outside the traced tensor inputs without changing CoRP."""

    def __init__(self, model: nn.Module, instruction: str) -> None:
        super().__init__()
        self.model = model
        self.instruction = instruction

    def forward(
        self, images: dict[str, torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        tokens, pooled_language = _tensor_parts(
            self.model, images, state, self.instruction
        )
        batch = state.shape[0]
        action = state.new_zeros(
            batch, self.model.action_horizon, self.model.action_dim
        )
        tau = state.new_zeros(batch)
        return _action_velocity(self.model, tokens, pooled_language, action, tau)


class RepresentationGraph(nn.Module):
    def __init__(self, model: nn.Module, instruction: str) -> None:
        super().__init__()
        self.model = model
        self.instruction = instruction

    def forward(self, images, state):
        return _tensor_parts(self.model, images, state, self.instruction)


class ActionGeneratorGraph(nn.Module):
    def __init__(self, model: nn.Module, instruction: str) -> None:
        super().__init__()
        self.model = model
        self.instruction = instruction

    def forward(self, images, state):
        tokens, pooled_language = _tensor_parts(
            self.model, images, state, self.instruction
        )
        batch = state.shape[0]
        action = state.new_zeros(
            batch, self.model.action_horizon, self.model.action_dim
        )
        tau = state.new_zeros(batch)
        return _action_velocity(self.model, tokens, pooled_language, action, tau)


def _render(graph, output_dir: Path, name: str) -> list[str]:
    rendered = []
    for fmt in ("svg", "pdf", "png"):
        path = graph.visual_graph.render(
            filename=name,
            directory=str(output_dir),
            format=fmt,
            cleanup=True,
        )
        rendered.append(str(Path(path).resolve()))
    return rendered


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
        default=ROOT / "audit/corp_model_visualization",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--instruction", default="open the cabinet door")
    args = parser.parse_args()
    if args.depth < 1:
        parser.error("--depth must be positive")

    for dot in (Path("/opt/homebrew/bin/dot"), ROOT / ".cache/graphviz-env/bin/dot"):
        if dot.is_file():
            os.environ["PATH"] = f"{dot.parent}:{os.environ.get('PATH', '')}"
            break

    torch.set_num_threads(4)
    model, load_report = load_corp_robocasa(
        args.checkpoint,
        config=ROBOCASA_CORP_CONFIG,
        device="cpu",
        task_aux=TaskAuxConfig(num_classes=61),
    )
    model.eval()
    images = {
        view: torch.zeros(1, 3, args.image_size, args.image_size)
        for view in model.view_names
    }
    state = torch.zeros(1, model.state_dim)
    graph_models = {
        "corp_full": TensorOnlyCorpGraph(model, args.instruction),
        "corp_representation": RepresentationGraph(model, args.instruction),
        "corp_action_generator": ActionGeneratorGraph(model, args.instruction),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "load_report": load_report,
        "image_size": args.image_size,
        "views": list(model.view_names),
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "action_horizon": model.action_horizon,
        "instruction": args.instruction,
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "representation_extractor": sum(
                p.numel() for p in model.representation_extractor.parameters()
            ),
            "visual_encoder": sum(
                p.numel()
                for p in model.representation_extractor.visual_encoder.parameters()
            ),
            "language_encoder": sum(
                p.numel()
                for p in model.representation_extractor.language_encoder.parameters()
            ),
            "resampler": sum(
                p.numel()
                for p in model.representation_extractor.cross_attention_resampler.parameters()
            ),
            "state_encoder": sum(
                p.numel()
                for p in model.representation_extractor.state_encoder.parameters()
            ),
            "action_generator": sum(
                p.numel() for p in model.action_generator.parameters()
            ),
            "task_aux_head": (
                0
                if model.task_aux_head is None
                else sum(p.numel() for p in model.task_aux_head.parameters())
            ),
        },
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    with torch.no_grad():
        info = summary(
            graph_models["corp_full"],
            input_data=(images, state),
            depth=args.depth,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "trainable",
            ),
            row_settings=("depth", "var_names"),
            verbose=0,
        )
    (args.out / "torchinfo.txt").write_text(str(info) + "\n")

    rendered: dict[str, list[str]] = {}
    with torch.no_grad():
        for name, graph_model in graph_models.items():
            graph = draw_graph(
                graph_model,
                input_data=(images, state),
                graph_name=name,
                depth=args.depth,
                device="cpu",
                dtypes=[torch.float32],
                mode="train",
                roll=True,
                show_shapes=True,
                expand_nested=False,
                hide_module_functions=True,
                hide_inner_tensors=False,
                strict=False,
                directory=str(args.out),
                save_graph=True,
            )
            rendered[name] = _render(graph, args.out, name)
    metadata["rendered"] = rendered
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
