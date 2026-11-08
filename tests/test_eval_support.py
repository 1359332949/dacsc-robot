"""Task-level source panel support checks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dacsc.eval_support import validate_source_panel


def write_task(
    root: Path, task: str, rows: list[tuple[int, int, int, str]]
) -> None:
    dataset = root / "v1.0/pretrain/atomic" / task / "date/lerobot"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "extras").mkdir()
    lines = []
    for episode, style, layout, drawer in rows:
        lines.append(
            json.dumps(
                {
                    "episode_index": episode,
                    "length": 4,
                    "tasks": [f"{task} open"],
                }
            )
        )
        extra = dataset / "extras" / f"episode_{episode:06d}"
        extra.mkdir()
        (extra / "ep_meta.json").write_text(
            json.dumps(
                {
                    "style_id": style,
                    "layout_id": layout,
                    "lang": f"{task} open",
                    "fixture_refs": {"drawer": drawer},
                }
            )
        )
    (dataset / "meta/episodes.jsonl").write_text("\n".join(lines) + "\n")


class EvalSupportTests(unittest.TestCase):
    def test_seen_layout_and_style_are_separate_from_pair_sparsity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_task(
                root,
                "TaskA",
                [
                    (0, 12, 31, "stack_1"),
                    (9, 13, 31, "stack_1"),
                    (10, 14, 32, "stack_2"),
                ],
            )
            report = validate_source_panel(
                root,
                tasks=["TaskA"],
                styles=[12, 14],
                layout_id=31,
                train_style_range=(11, 50),
                source_holdout_residue=9,
            )
            task = report["tasks"]["TaskA"]
            self.assertTrue(report["passed"])
            self.assertTrue(task["layout_supported"])
            self.assertTrue(task["styles_supported"])
            self.assertFalse(task["pairs_supported"])

            strict = validate_source_panel(
                root,
                tasks=["TaskA"],
                styles=[12, 14],
                layout_id=31,
                train_style_range=(11, 50),
                source_holdout_residue=9,
                require_pairs=True,
            )
            self.assertFalse(strict["passed"])

            fixture_seen = validate_source_panel(
                root,
                tasks=["TaskA"],
                styles=[12],
                layout_id=31,
                train_style_range=(11, 50),
                source_holdout_residue=9,
                requested_fixture_refs_by_task={
                    "TaskA": {"drawer": "stack_1"}
                },
            )
            self.assertTrue(fixture_seen["passed"])
            task = fixture_seen["tasks"]["TaskA"]
            self.assertEqual(task["requested_fixture_train_support"], {"12": 1})
            self.assertEqual(
                task["requested_style_layout_fixture_support"], {"12": 1}
            )

            fixture_unseen = validate_source_panel(
                root,
                tasks=["TaskA"],
                styles=[12, 14],
                layout_id=31,
                train_style_range=(11, 50),
                source_holdout_residue=9,
                requested_fixture_refs_by_task={
                    "TaskA": {"drawer": "stack_1"}
                },
            )
            self.assertFalse(fixture_unseen["passed"])
            self.assertEqual(
                fixture_unseen["tasks"]["TaskA"]["missing_fixture_train_styles"],
                [14],
            )

    def test_missing_layout_or_source_only_style_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_task(
                root,
                "TaskA",
                [
                    (0, 12, 31, "stack_1"),
                    (9, 13, 31, "stack_1"),
                    (10, 14, 32, "stack_2"),
                ],
            )
            missing_style = validate_source_panel(
                root,
                tasks=["TaskA"],
                styles=[12, 13],
                layout_id=31,
                train_style_range=(11, 50),
                source_holdout_residue=9,
            )
            self.assertFalse(missing_style["passed"])
            self.assertEqual(
                missing_style["tasks"]["TaskA"]["missing_train_styles"], [13]
            )

            missing_layout = validate_source_panel(
                root,
                tasks=["TaskA"],
                styles=[12],
                layout_id=11,
                train_style_range=(11, 50),
                source_holdout_residue=9,
            )
            self.assertFalse(missing_layout["passed"])
            self.assertFalse(
                missing_layout["tasks"]["TaskA"]["layout_supported"]
            )


if __name__ == "__main__":
    unittest.main()
