# Data layout

RoboCasa datasets and scene assets are external and are not tracked by Git.

The active pipeline expects RoboCasa365 atomic training data under:

```text
<data-root>/v1.0/pretrain/atomic/<Task>/<date>/lerobot
```

Use one of the checked provisioning paths:

```bash
python3 scripts/fetch_robocasa365.py --dryrun --preset probe
python3 scripts/fetch_robocasa365.py --preset style-check \
  --root /path/to/robocasa365 --audit
python3 scripts/fetch_robocasa365_modelscope.py --preset atomic-all \
  --root /path/to/robocasa365 --audit
```

Scene assets can be fetched with:

```bash
bash scripts/fetch_robocasa_assets_hf.sh /path/to/robocasa_assets
```

After download, validate the layout:

```bash
python3 scripts/check_data_layout.py --root /path/to/robocasa365
python3 scripts/audit_dataset.py \
  --dataset-root /path/to/robocasa365/v1.0/pretrain \
  --out audit/pretrain
```

`--data-root`, `DACSC_DATA_ROOT`, or the default host-side dataset location is
selected by the training/evaluation entry point. Do not commit datasets,
checkpoints, videos, or absolute-path symlinks.
