import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("analyse_b2_v3", Path(__file__).resolve().parents[1] / "scripts/analyse_b2_v3.py")
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


def row(**kw):
    base = {"domain": "Camera", "suite": "libero_spatial", "task_id": 705, "episode": 0,
            "branch": 8, "candidate": "prod_step_005"}
    base.update(kw); return base


def test_complete_row_is_neither_success():
    rows = [row(tau_prox=1e-6, action_change=1e-7, applied_norm=1e-5, alpha=None)]
    s = mod.summarise(rows)
    assert s["success_outcomes"]["neither"] == 1
    assert s["candidates"]["prod_step_005"]["n_complete"] == 1


def test_success_classification_and_stage_skip():
    rows = [
        {"domain": "Camera", "suite": "libero_spatial", "task_id": 705, "episode": 0,
         "branch": 8, "censor": "stage_or_success", "contact": 0.5, "success_before": True},
        row(censor="termination_or_success", success_control=False, success_update=True),
        row(censor="termination_or_success", success_control=True, success_update=False),
        row(censor="termination_or_success", success_control=True, success_update=True),
        row(censor="action_cap", action_changes=[0.2], steps_control=6, steps_update=6),
    ]
    s = mod.summarise(rows)
    c = s["candidates"]["prod_step_005"]
    assert s["stage_skips"] == 1
    assert c["success_outcomes"]["update_only"] == 1
    assert c["success_outcomes"]["control_only"] == 1
    assert c["success_outcomes"]["both"] == 1
    assert c["success_outcomes"]["not_observed"] == 1
    assert c["censor_reasons"]["action_cap"] == 1
