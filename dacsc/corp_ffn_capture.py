"""Read-only CoRP FFN/block capture and a pinned Registers analysis adapter.

No gates, model replacement, extra tokens, gradients or training integration.
Upstream logging and neuron ranking are called from verified original files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_registers_reference(root: str | Path) -> SimpleNamespace:
    """Load original analysis functions in an isolated namespace, without networking."""
    root = Path(root).expanduser().resolve()
    contract = json.loads((ROOT / "analysis_upstream.lock.json").read_text())[
        "Registers"
    ]
    for rel, expected in contract["files"].items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"prepare the pinned Registers source first: {path}"
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Registers source changed: {rel}")
    # Do not shadow another project's 'shared' package or alter sys.path.
    prefix = "_dacsc_registers_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    for name, directory in ((prefix, root), (prefix + ".shared", root / "shared")):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules[name] = package
    hooks = importlib.import_module(prefix + ".shared.hook_fn")
    manager = importlib.import_module(prefix + ".shared.hook_manager")
    algorithms = importlib.import_module(prefix + ".shared.algorithms")
    return SimpleNamespace(
        log_internal=hooks.log_internal,
        find_register_neurons=algorithms.find_register_neurons,
        HookMode=manager.HookMode,
        source=root,
        contract=contract,
    )


@dataclass(frozen=True)
class FFNFrame:
    """Independent CPU FP32 snapshots in [sample,view,CLS+patch,feature] order."""

    view_names: tuple[str, ...]
    ffn: dict[int, torch.Tensor]
    blocks: dict[int, torch.Tensor]
    source_dtypes: dict[str, str]

    def upstream_arrays(self, sample: int, view: str) -> tuple[np.ndarray, np.ndarray]:
        """Registers expects [layer,token,feature] for one image, including CLS."""
        index = self.view_names.index(view)
        if not 0 <= sample < next(iter(self.ffn.values())).shape[0]:
            raise IndexError("sample outside the captured batch")
        return (
            np.stack([x[sample, index].numpy() for x in self.ffn.values()]),
            np.stack([x[sample, index].numpy() for x in self.blocks.values()]),
        )


class CoRPFFNCapture:
    """Context-scoped observers. Keeps only the latest completed backbone forward.

    The caller supplies eval/no_grad or inference_mode. Hooks return None and
    never replace an output; clone before calling the original NumPy logger.
    """

    def __init__(self, policy, reference, *, view_names: tuple[str, ...] | None = None):
        if getattr(policy, "token_style", None) is not None:
            raise ValueError("capture the no-intervention CoRP, not a styled policy")
        self.policy = policy
        self.reference = reference
        self.backbone = policy.representation_extractor.visual_encoder.backbone
        self.view_names = tuple(policy.view_names if view_names is None else view_names)
        if not self.view_names or len(set(self.view_names)) != len(self.view_names):
            raise ValueError("capture needs unique view names")
        if not set(self.view_names) <= set(policy.view_names):
            raise ValueError("capture view is outside the policy camera contract")
        if tuple(policy.representation_extractor.view_names) != tuple(
            policy.view_names
        ):
            raise ValueError("policy and official extractor camera orders differ")
        self._single_view_calibration = self.view_names != tuple(policy.view_names)
        if self._single_view_calibration and len(self.view_names) != 1:
            raise ValueError("use full policy order or one explicit calibration view")
        self.layers = tuple(range(len(self.backbone.blocks)))
        if not self.layers:
            raise ValueError("backbone has no blocks")
        self.embedding_dim = self.backbone.blocks[0].mlp.fc1.in_features
        self.ffn_dim = self.backbone.blocks[0].mlp.fc1.out_features
        for block in self.backbone.blocks:
            if (
                not isinstance(block.mlp.act, torch.nn.GELU)
                or block.mlp.fc1.in_features != self.embedding_dim
                or block.mlp.fc1.out_features != self.ffn_dim
                or block.mlp.fc2.in_features != self.ffn_dim
            ):
                raise ValueError("capture requires the plain CoRP GELU FFN contract")
        self._handles = []
        self.clear()

    def clear(self):
        self._logs = {
            kind: {layer: [] for layer in self.layers} for kind in ("ffn", "blocks")
        }
        self._dtypes = {}
        self.last: FFNFrame | None = None
        self._rows = 0

    def __enter__(self):
        if self._handles:
            raise RuntimeError("capture is already active")
        try:
            self._handles.append(
                self.policy.representation_extractor.register_forward_pre_hook(
                    self._check_policy_views
                )
            )
            self._handles.append(self.backbone.register_forward_pre_hook(self._begin))
            for layer, block in enumerate(self.backbone.blocks):
                self._handles.append(
                    block.mlp.act.register_forward_hook(self._logger("ffn", layer))
                )
                self._handles.append(
                    block.register_forward_hook(self._logger("blocks", layer))
                )
            self._handles.append(
                self.backbone.register_forward_hook(self._finish, always_call=True)
            )
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, *args):
        self.close()

    def _check_policy_views(self, module, inputs):
        self.clear()
        if self._single_view_calibration:
            raise ValueError(
                "single-view calibration cannot label a multi-view policy forward"
            )
        if (
            tuple(module.view_names) != self.view_names
            or tuple(self.policy.view_names) != self.view_names
        ):
            raise ValueError("camera order changed after observer construction")

    def _begin(self, module, inputs):
        self.clear()
        if torch.is_grad_enabled() or any(m.training for m in self.policy.modules()):
            raise RuntimeError(
                "FFN collection requires eval and no_grad/inference_mode"
            )
        if getattr(self.policy, "token_style", None) is not None:
            raise ValueError("styled policy is outside read-only baseline capture")
        image = inputs[0]
        if image.ndim != 4 or image.shape[1:] != (3, 224, 224):
            raise ValueError("capture requires the CoRP224 RGB image contract")
        if not image.shape[0] or image.shape[0] % len(self.view_names):
            raise ValueError("flat visual batch is not divisible by the view count")
        self._rows = image.shape[0]

    def _logger(self, kind, layer):
        def observe(module, inputs, output):
            width = self.ffn_dim if kind == "ffn" else self.embedding_dim
            if output.shape != (self._rows, 257, width) or self._logs[kind][layer]:
                raise ValueError("unexpected or repeated CLS/patch activation")
            self._dtypes[f"{kind}.{layer}"] = str(output.dtype)
            # Preserve upstream log_internal, but avoid aliasing CPU model output
            # and explicitly adapt BF16 to NumPy's supported FP32 log format.
            snapshot = output.detach().to(device="cpu", dtype=torch.float32, copy=True)
            self.reference.log_internal(
                module, inputs, snapshot, self._logs[kind][layer]
            )
            # Returning None is essential: this observer does not modify forward.

        return observe

    def _finish(self, module, inputs, output):
        if output is None:
            self.clear()
            return
        if output.shape != (self._rows, 256, self.embedding_dim):
            self.clear()
            raise ValueError(
                "backbone output is not exactly the original patch sequence"
            )
        if any(
            len(log) != 1 for values in self._logs.values() for log in values.values()
        ):
            self.clear()
            raise RuntimeError("incomplete FFN capture")
        batch = self._rows // len(self.view_names)

        def unpack(kind):
            return {
                layer: torch.from_numpy(values[0])
                .reshape(len(self.view_names), batch, 257, -1)
                .transpose(0, 1)
                for layer, values in self._logs[kind].items()
            }

        self.last = FFNFrame(
            self.view_names, unpack("ffn"), unpack("blocks"), dict(self._dtypes)
        )


def rank_register_images(
    policy,
    reference,
    image_path: str | Path,
    *,
    view_name: str,
    register_norm_threshold: float,
    detect_outliers_layer: int,
    processed_image_cnt: int = 100,
):
    """Optional single-view calibration: call the original detector unchanged.

    No images/no outliers retains the original failure; no fallback ranking.
    This is register-neuron analysis, not style scoring or action attribution.
    """
    from dacsc.corp_images import prepare_image

    if not np.isfinite(register_norm_threshold) or register_norm_threshold <= 0:
        raise ValueError("declare a positive finite outlier threshold")
    if type(processed_image_cnt) is not int or processed_image_cnt <= 0:
        raise ValueError("declare a positive calibration image count")
    capture = CoRPFFNCapture(policy, reference, view_names=(view_name,))
    if type(detect_outliers_layer) is not int or not -len(
        capture.layers
    ) <= detect_outliers_layer < len(capture.layers):
        raise ValueError("outlier detection layer outside current backbone")

    class Manager:
        def reinit(self, mode):
            if mode != reference.HookMode.ANALYSIS:
                raise ValueError("intervention mode is unsupported")
            capture.clear()

        def finalize(self):
            if not capture._handles:
                raise RuntimeError("read-only observer is not active")

        def get_neuron_activations(self):
            return capture.last.upstream_arrays(0, view_name)[0]

        def get_layer_outputs(self):
            return capture.last.upstream_arrays(0, view_name)[1]

    def run_model(backbone, image):
        with torch.inference_mode():
            return backbone(image)

    state = dict(
        model=capture.backbone,
        preprocess=lambda image: prepare_image(np.asarray(image.convert("RGB"))),
        run_model=run_model,
        hook_manager=Manager(),
        num_layers=len(capture.layers),
        num_neurons_per_layer=capture.ffn_dim,
    )
    device = str(next(capture.backbone.parameters()).device)
    # The original load_images uses random.sample. Preserve its algorithm and
    # caller's starting state, but do not perturb later data/analysis sampling.
    rng = random.getstate()
    try:
        with capture:
            return reference.find_register_neurons(
                state,
                str(image_path),
                register_norm_threshold=register_norm_threshold,
                detect_outliers_layer=detect_outliers_layer,
                device=device,
                processed_image_cnt=processed_image_cnt,
            )
    finally:
        random.setstate(rng)
