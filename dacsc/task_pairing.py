"""Task identity helpers for LIBERO-Plus V1.1.

The canonicalization here is deliberately strict: only known parameterized
variant suffixes are stripped.  We no longer strip arbitrary trailing digits.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

# Real Camera names look like:
#   ..._view_46_0_100_0_0_initstate_0
# Table/light/language variants use similar parameterized suffixes.
_VARIANT_SUFFIX = re.compile(
    r"_(?:view|views|camera|cameras|table|tb|light|lighting|background|bg|"
    r"language|lang|noise|sensor|layout)_\d[^/]*$"
)
_INITSTATE_SUFFIX = re.compile(r"_initstate_\d+$")
_PARAM_SUFFIX = re.compile(r"_(?:view|views|camera|cameras|table|tb|light|lighting|"
                           r"background|bg|language|lang|noise|sensor|layout)_\d[^ ]*$")


def canonical_base_name(name: str) -> str:
    out = name.strip()
    out = _INITSTATE_SUFFIX.sub("", out)
    out = _VARIANT_SUFFIX.sub("", out)
    return out.strip("_-")


def canonical_instruction(text: str) -> str:
    out = re.sub(r"\s+", " ", text.strip())
    out = _PARAM_SUFFIX.sub("", out)
    return out.strip()


def _section(text: str, keyword: str) -> str:
    start = text.find(f"(:{keyword}")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def parse_objects(text: str) -> list[dict[str, str]]:
    """Return [{'name': ..., 'type': ...}] from the BDDL objects section."""
    block = _section(text, "objects")
    if not block:
        return []
    body = block[len("(:objects"):].rstrip(")")
    body = re.sub(r";.*", "", body)
    tokens = body.replace("(", " ").replace(")", " ").split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"-", "and", "or", "not", "default"} or tok.startswith(":"):
            i += 1; continue
        if i + 2 < len(tokens) and tokens[i + 1] == "-":
            out.append({"name": tok, "type": tokens[i + 2]})
            i += 3
        else:
            out.append({"name": tok, "type": "unknown"})
            i += 1
    # stable unique by (name, type)
    seen = set(); unique = []
    for obj in out:
        key = (obj["name"], obj["type"])
        if key not in seen:
            seen.add(key); unique.append(obj)
    return sorted(unique, key=lambda x: (x["name"], x["type"]))


def goal_text(text: str) -> str:
    block = _section(text, "goal")
    return re.sub(r"\s+", " ", block).strip() if block else ""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_fingerprint(task, bddl_path: Path) -> dict:
    text = Path(bddl_path).read_text(errors="replace")
    return {
        "task_name": str(getattr(task, "name", "")),
        "language": str(getattr(task, "language", "")),
        "canonical_language": canonical_instruction(str(getattr(task, "language", ""))),
        "base_name": canonical_base_name(str(getattr(task, "name", ""))),
        "bddl_path": str(bddl_path),
        "bddl_sha256": sha256_file(bddl_path),
        "goal_text": goal_text(text),
        "objects": parse_objects(text),
    }


def task_pair_compatible(clean: dict, shift: dict) -> dict:
    checks = {
        "language": bool(clean["canonical_language"]) and
                    clean["canonical_language"] == shift["canonical_language"],
        "base_name": bool(clean["base_name"]) and clean["base_name"] == shift["base_name"],
        "goal_text": bool(clean["goal_text"]) and clean["goal_text"] == shift["goal_text"],
        "objects": bool(clean["objects"]) and clean["objects"] == shift["objects"],
    }
    return {"checks": checks, "pass": all(checks.values())}


def load_task_map(path: Path) -> dict[str, list[str]]:
    """Load libero_suite_task_map.py (or JSON) as {suite: [task names]}."""
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
    else:
        spec = importlib.util.spec_from_file_location("_libero_task_map", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = vars(module)
    out: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                out[str(key)] = list(value)
            elif isinstance(value, dict) and all(isinstance(x, str) for x in value.values()):
                out[str(key)] = list(value.values())
    return out


def resolve_task_name(task_map: dict[str, list[str]], suite: str, index: int) -> tuple[str, str]:
    if suite not in task_map:
        raise KeyError(f"suite {suite!r} not in task map")
    names = task_map[suite]
    if not (0 <= index < len(names)):
        raise IndexError(f"task {index} out of range for {suite} (n={len(names)})")
    name = names[index]
    return name, canonical_base_name(name)


def classification_entries(data, suite: str, task_id: int) -> list[dict]:
    """Explicit (suite, task_id) lookup; never mix ids across suites."""
    found = []

    def walk(obj, current_suite=None):
        if isinstance(obj, dict):
            s = obj.get("suite", current_suite)
            for key in ("task_id", "task_index", "index", "id"):
                if key in obj:
                    try:
                        tid = int(obj[key])
                    except Exception:
                        tid = None
                    if tid == task_id and (s is None or str(s) == suite):
                        found.append(obj)
            for k, v in obj.items():
                walk(v, str(k) if str(k) in {"libero_spatial", "libero_object", "libero_goal", "libero_10"} else s)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, current_suite)
    walk(data)
    return found


def validate_classification(entries: list[dict], expected_category: str | None = None) -> dict:
    checks = {"present": bool(entries)}
    if not entries:
        return {"checks": checks, "pass": False, "categories": []}
    cats = []
    for e in entries:
        cat = None
        for key in ("category", "perturbation", "perturbation_type", "type"):
            if key in e:
                cat = str(e[key]); break
        cats.append(cat)
    checks["category"] = True
    if expected_category:
        checks["category"] = any(c is not None and expected_category.lower() in c.lower()
                                 for c in cats)
    return {"checks": checks, "pass": all(checks.values()), "categories": cats}
