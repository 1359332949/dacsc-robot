"""Shared parser for held-out metrics in DACSC run directories."""
from __future__ import annotations

import json
from pathlib import Path


def read_run_metrics(run_dir: Path) -> dict:
    """Return train diagnostics, held-out validation, and final test metrics.

    ``train_action_mse_error`` is retained only as a diagnostic of train-set
    fitting. It is never promoted to ``val_action_mse``.
    """
    out = {
        "val_action_mse": None,
        "val_samples": None,
        "val_batches": None,
        "test_action_mse": None,
        "test_samples": None,
        "test_batches": None,
        "train_action_mse": None,
        "train_loss": None,
    }
    log = run_dir / "logs.json.txt"
    if log.exists():
        for line in log.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("val_action_mse") is not None:
                out["val_action_mse"] = float(row["val_action_mse"])
                out["val_samples"] = row.get("val_samples")
                out["val_batches"] = row.get("val_batches")
            if row.get("train_action_mse_error") is not None:
                out["train_action_mse"] = float(row["train_action_mse_error"])
            if row.get("train_loss") is not None:
                out["train_loss"] = float(row["train_loss"])

    test_file = run_dir / "test_metrics.json"
    if test_file.exists():
        test = json.loads(test_file.read_text())
        out["test_action_mse"] = test.get("test_action_mse")
        out["test_samples"] = test.get("test_samples")
        out["test_batches"] = test.get("test_batches")
    return out
