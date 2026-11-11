"""Closed-loop success-rate evaluation with the PSC source-support projection.

Thin wrapper around scripts/eval_corp_robocasa_sr.py: it monkey-patches
CorpRoboCasa.sample_action so that the cross-attention write is projected onto
the source support (dacsc/psc.py) for the duration of each policy call.
Install path is passed with the PSC_STATS environment variable; without it the
wrapper is a no-op (identical to the upstream evaluator).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_robocasa import CorpRoboCasa  # noqa: E402
from dacsc.psc import install, load_stats  # noqa: E402

_original_sample_action = CorpRoboCasa.sample_action
_stats_path = os.environ.get("PSC_STATS", "")
_stats = load_stats(_stats_path) if _stats_path else None


def _sample_action_with_psc(self, images, state, instruction, num_steps=None):
    if _stats is None:
        return _original_sample_action(self, images, state, instruction, num_steps=num_steps)
    projector = install(self, _stats)
    try:
        return _original_sample_action(self, images, state, instruction, num_steps=num_steps)
    finally:
        projector.close()


if _stats is not None:
    CorpRoboCasa.sample_action = _sample_action_with_psc

from scripts.eval_corp_robocasa_sr import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
