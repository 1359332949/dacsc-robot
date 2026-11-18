# Failure-event annotation v1

Motivated by VLA-Tug (arXiv:2611.08425): failure is not the episode timeout;
it is the first observable point after which the task can no longer be
recovered by the current policy.

## Schema

For each failed trajectory:

| field | meaning |
|---|---|
| `failure_event_step` | first step where the causal error becomes observable |
| `failure_mode` | grasp_miss / late_close / slip / no_lift / wrong_target / stuck / other |
| `task_phase` | approach / alignment / grasp / lift / transport / place |
| `irrecoverable` | whether recovery was already impossible at this step |
| `event_confidence` | annotator confidence |
| `event_source` | `human_annotation` |

## Workflow

1. Generate a template:

```bash
python scripts/make_failure_annotation_template.py \
  --trajectory-dir /path/to/target_dev \
  --out /tmp/failure_annotation.csv
```

2. Fill `failure_event_step` and the other columns from video + proprio/action
   trace. Do not infer from the final timeout alone.

3. Validate and emit annotations:

```bash
python scripts/annotate_failure_events.py \
  --csv /tmp/failure_annotation.csv \
  --out docs/data/failure_annotations_v1.json
```

4. Build monitor windows with annotations:

```bash
python scripts/build_monitor_dataset.py \
  --trajectory-dir /path/to/target_dev \
  --annotations docs/data/failure_annotations_v1.json \
  --label-mode within_horizon --horizon 10 \
  --out /tmp/target_annotated.npz
```

## Important

The current trajectory data has **no failure-event annotations**. Existing
lead-time numbers are remaining-to-episode-end, not lead-to-failure. Do not
report them as failure warning lead time.
