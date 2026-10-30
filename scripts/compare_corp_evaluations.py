"""Compare completed evaluations using standard seed-level NumPy/SciPy statistics."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.evaluation_statistics import compare_evaluations
from dacsc.run_output import entry_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    # Relative case paths are relative to the manifest, not the process cwd.
    for run in manifest["runs"]:
        for side in ("left", "right"):
            path = Path(run[side]).expanduser()
            run[side] = str(
                (args.manifest.parent / path).resolve()
                if not path.is_absolute()
                else path.resolve()
            )
    with entry_file(args.out):
        result = compare_evaluations(manifest, confidence=args.confidence)
        tmp = args.out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
        tmp.replace(args.out)
    print(json.dumps(result["paired_difference"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
