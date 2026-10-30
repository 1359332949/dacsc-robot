"""Exclusive output ownership using upstream filelock, including log creation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import runpy
import sys

from filelock import FileLock

_ACTIVE_OUTPUT: Path | None = None


@contextmanager
def output_directory(path: Path, *, resume: bool = False):
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock beside the directory so a failed contender cannot mutate its contents.
    with FileLock(str(path) + ".lock", timeout=0):
        if path.exists() and not path.is_dir():
            raise FileExistsError(path)
        if not resume and path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Refusing to reuse nonempty run directory: {path}")
        path.mkdir(exist_ok=True)
        yield path


@contextmanager
def output_file(path: Path):
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=0):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite result: {path}")
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", default="run.log")
    parser.add_argument(
        "--file", action="store_true", help="Own one result file instead of a directory"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or Path(args.log).name != args.log:
        parser.error("a command and a plain log filename are required")
    owner = output_file(args.out) if args.file else output_directory(args.out)
    with owner as path:
        # Run the Python entry in this process so the upstream lock also covers
        # imports, log opening, exceptions and SystemExit without orphan children.
        import os

        global _ACTIVE_OUTPUT
        _ACTIVE_OUTPUT = path
        original_argv = sys.argv
        stdout_fd, stderr_fd = os.dup(1), os.dup(2)
        try:
            log_path = (path.parent if args.file else path) / args.log
            with log_path.open("x") as stream:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stream.fileno(), 1)
                os.dup2(stream.fileno(), 2)
                sys.argv = command
                try:
                    runpy.run_path(command[0], run_name="__main__")
                except Exception:
                    import traceback

                    traceback.print_exc()
                    raise
            return 0
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            sys.argv = original_argv
            _ACTIVE_OUTPUT = None


@contextmanager
def entry_output(path: Path, *, resume: bool = False):
    """Direct entries lock too; entries invoked by main share its lock lifetime."""
    path = path.expanduser().resolve()
    if _ACTIVE_OUTPUT is not None:
        if _ACTIVE_OUTPUT != path:
            raise ValueError("output owner directory mismatch")
        yield path
    else:
        with output_directory(path, resume=resume) as owned:
            yield owned


@contextmanager
def entry_file(path: Path):
    path = path.expanduser().resolve()
    if _ACTIVE_OUTPUT is not None:
        if _ACTIVE_OUTPUT != path:
            raise ValueError("result owner path mismatch")
        yield path
    else:
        with output_file(path) as owned:
            yield owned


if __name__ == "__main__":
    # Keep the same module-level lock context for the imported Python entry.
    from dacsc.run_output import main as run_main

    raise SystemExit(run_main())
