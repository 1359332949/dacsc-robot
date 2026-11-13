"""Task-identity fingerprints for V1.1 Clean/Shift pairing preflight."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SUFFIX_PATTERNS = (
    r"_(?:view|views|camera|cameras)(?:_?\d+)?$",
    r"_(?:light|lighting)(?:_?\d+)?$",
    r"_(?:background|bg|table|tb)(?:_?\d+)?$",
    r"_(?:noise|sensor)(?:_?\d+)?$",
    r"_(?:language|lang)(?:_?\d+)?$",
    r"_(?:layout)(?:_?\d+)?$",
    r"_(?:robot|init|initial)(?:_?\d+)?$",
    r"_\d+$",
)


def canonical_base_name(name: str) -> str:
    out = name.strip().lower()
    changed = True
    while changed:
        changed = False
        for pat in _SUFFIX_PATTERNS:
            new = re.sub(pat, "", out)
            if new != out:
                out = new; changed = True
    return out.strip("_-")


def _section(text: str, keyword: str) -> str:
    """Return the parenthesized section starting with (:keyword ...)."""
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


def parse_objects(text: str) -> list[str]:
    block = _section(text, "objects")
    if not block:
        return []
    body = block[len("(:objects"):].rstrip(")")
    body = re.sub(r";.*", "", body)
    tokens = body.replace("(", " ").replace(")", " ").split()
    names = []
    skip = {"-", "and", "or", "not", "default"}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in skip or tok.startswith(":") or tok in {"-"}:
            i += 1; continue
        # "name - type": keep name, skip type token after '-'
        names.append(tok)
        if i + 1 < len(tokens) and tokens[i + 1] == "-":
            i += 3  # name, '-', type
        else:
            i += 1
    return sorted(set(names))


def goal_text(text: str) -> str:
    block = _section(text, "goal")
    if not block:
        return ""
    return re.sub(r"\s+", " ", block).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_fingerprint(task, bddl_path: Path) -> dict:
    text = Path(bddl_path).read_text(errors="replace")
    return {
        "task_name": str(getattr(task, "name", "")),
        "language": str(getattr(task, "language", "")),
        "base_name": canonical_base_name(str(getattr(task, "name", ""))),
        "bddl_path": str(bddl_path),
        "bddl_sha256": sha256_file(bddl_path),
        "goal_text": goal_text(text),
        "objects": parse_objects(text),
    }


def task_pair_compatible(clean: dict, shift: dict) -> dict:
    checks = {
        "language": bool(clean["language"]) and clean["language"] == shift["language"],
        "base_name": bool(clean["base_name"]) and clean["base_name"] == shift["base_name"],
        "goal_text": bool(clean["goal_text"]) and clean["goal_text"] == shift["goal_text"],
        "objects": bool(clean["objects"]) and clean["objects"] == shift["objects"],
    }
    return {"checks": checks, "pass": all(checks.values())}
