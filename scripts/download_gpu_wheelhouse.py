#!/usr/bin/env python3
"""Resolve hashed lock entries to Linux wheels and download a reusable wheelhouse."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024
DEFAULT_RANGE_THRESHOLD = 64 * 1024 * 1024
DEFAULT_WORKERS = 16


class DirectoryLock:
    """Hold an exclusive advisory lock for one wheelhouse directory."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._stream = None

    def __enter__(self) -> "DirectoryLock":
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory / ".wheelhouse.lock"
        stream = lock_path.open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise RuntimeError(
                f"another wheelhouse download owns {self.directory}"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write("locked\n")
        stream.flush()
        self._stream = stream
        return self

    def __exit__(self, *_args) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


def parse_lock(path: Path) -> list[tuple[str, str, set[str]]]:
    entries: list[tuple[str, str, set[str]]] = []
    current: tuple[str, str, set[str]] | None = None
    for raw in path.read_text().splitlines():
        if raw and not raw[:1].isspace() and "==" in raw:
            if current is not None:
                entries.append(current)
            name, version = raw.split("==", 1)
            version = re.split(r"[ ;\\]", version, maxsplit=1)[0]
            current = (name.lower().replace("_", "-"), version, set())
        elif current is not None:
            for digest in re.findall(r"--hash=sha256:([0-9a-f]{64})", raw):
                current[2].add(digest.lower())
    if current is not None:
        entries.append(current)
    if not entries or any(not hashes for _, _, hashes in entries):
        raise ValueError(f"invalid or unhashed lock file: {path}")
    return entries


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compatible(filename: str) -> bool:
    name = filename.lower()
    if not name.endswith(".whl"):
        return False
    if any(
        token in name
        for token in ("win_", "macosx", "aarch64", "arm64", "universal2")
    ):
        return False
    if "x86_64" not in name and not name.endswith("-none-any.whl"):
        return False
    if "-cp3" in name and "-cp312-" not in name and "-abi3-" not in name:
        return False
    return True


def source_archive(filename: str) -> bool:
    name = filename.lower()
    return name.endswith((".tar.gz", ".zip"))


def compatible_artifact(filename: str) -> bool:
    return compatible(filename) or source_archive(filename)


def page_links(url: str) -> list[str]:
    request = Request(url, headers={"User-Agent": "dacsc-wheelhouse/1"})
    with urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")
    return [
        urljoin(url, unquote(link))
        for link in re.findall(r'href=["\']([^"\']+)', html)
    ]


def find_hashed_artifact(
    name: str,
    version: str,
    hashes: set[str],
    indexes: list[str],
) -> tuple[str, str]:
    errors = []
    source_candidates: list[tuple[str, str]] = []
    for index in indexes:
        url = f"{index.rstrip('/')}/{name}/"
        try:
            links = page_links(url)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        for link in links:
            filename = Path(urlparse(link).path).name
            fragment = urlparse(link).fragment.lower()
            digest = fragment.removeprefix("sha256=")
            if compatible(filename) and digest in hashes:
                return link.split("#", 1)[0], digest
            if source_archive(filename) and digest in hashes:
                source_candidates.append((link.split("#", 1)[0], digest))
    if source_candidates:
        return source_candidates[0]
    raise RuntimeError(
        f"no compatible locked artifact for {name}=={version}; "
        f"tried {indexes}; {'; '.join(errors)}"
    )


def find_extra_wheel(name: str, version: str, index: str) -> tuple[str, str]:
    links = page_links(f"{index.rstrip('/')}/{name}/")
    expected = f"{name.replace('-', '_')}-{version}"
    candidates = [
        link
        for link in links
        if compatible(Path(urlparse(link).path).name)
        and expected in Path(urlparse(link).path).name.lower()
    ]
    if not candidates:
        raise RuntimeError(f"no compatible wheel for {name}=={version}")
    chosen = candidates[0]
    digest = urlparse(chosen).fragment.removeprefix("sha256=").lower()
    if len(digest) != 64:
        raise ValueError(f"missing sha256 fragment: {chosen}")
    return chosen.split("#", 1)[0], digest


def _remote_size(url: str) -> int | None:
    request = Request(
        url,
        method="HEAD",
        headers={"User-Agent": "dacsc-wheelhouse/1", "Accept-Encoding": "identity"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def _supports_ranges(url: str, expected_size: int | None) -> bool:
    request = Request(
        url,
        headers={
            "User-Agent": "dacsc-wheelhouse/1",
            "Accept-Encoding": "identity",
            "Range": "bytes=0-0",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith("bytes 0-0/"):
                return False
            total = int(content_range.rsplit("/", 1)[1])
            return expected_size is None or total == expected_size
    except Exception:
        return False


def chunk_specs(size: int, chunk_size: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("remote file size must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    return [
        (start, min(start + chunk_size, size) - 1)
        for start in range(0, size, chunk_size)
    ]


def chunk_path(scratch: Path, start: int, end: int) -> Path:
    return scratch / f"{start:012d}-{end:012d}.chunk"


def download_chunk(
    url: str,
    start: int,
    end: int,
    target: Path,
    curl: str,
) -> None:
    expected = end - start + 1
    if target.is_file() and target.stat().st_size == expected:
        return
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            curl,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "10",
            "--retry-all-errors",
            "--range",
            f"{start}-{end}",
            "--output",
            str(temporary),
            url,
        ],
        check=True,
    )
    actual = temporary.stat().st_size if temporary.is_file() else -1
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"range response has {actual} bytes, expected {expected}: "
            f"{url} [{start}-{end}]"
        )
    temporary.replace(target)


def assemble_chunks(
    target: Path,
    scratch: Path,
    specs: list[tuple[int, int]],
) -> Path:
    partial = target.with_name(f".{target.name}.part")
    with partial.open("wb") as output:
        for start, end in specs:
            chunk = chunk_path(scratch, start, end)
            expected = end - start + 1
            if not chunk.is_file() or chunk.stat().st_size != expected:
                raise RuntimeError(f"missing or incomplete chunk: {chunk}")
            with chunk.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
    return partial


def parallel_download(
    url: str,
    digest: str,
    target: Path,
    size: int,
    curl: str,
    chunk_size: int,
    workers: int,
) -> None:
    scratch = target.with_name(f".{target.name}.chunks")
    scratch.mkdir(parents=True, exist_ok=True)
    specs = chunk_specs(size, chunk_size)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_chunk,
                url,
                start,
                end,
                chunk_path(scratch, start, end),
                curl,
            ): (start, end)
            for start, end in specs
        }
        for future in as_completed(futures):
            future.result()
    partial = assemble_chunks(target, scratch, specs)
    actual = sha256(partial)
    if actual != digest:
        partial.unlink(missing_ok=True)
        for start, end in specs:
            chunk_path(scratch, start, end).unlink(missing_ok=True)
        scratch.rmdir()
        raise ValueError(
            f"hash mismatch after download: {target.name} "
            f"expected {digest}, got {actual}"
        )
    partial.replace(target)
    for start, end in specs:
        chunk_path(scratch, start, end).unlink(missing_ok=True)
    scratch.rmdir()


def serial_download(url: str, digest: str, target: Path, curl: str) -> None:
    partial = target.with_name(f".{target.name}.part")
    subprocess.run(
        [
            curl,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "10",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    actual = sha256(partial)
    if actual != digest:
        raise ValueError(
            f"hash mismatch after download: {partial} "
            f"expected {digest}, got {actual}"
        )
    partial.replace(target)


def download(
    url: str,
    digest: str,
    out: Path,
    curl: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    range_threshold: int = DEFAULT_RANGE_THRESHOLD,
    workers: int = DEFAULT_WORKERS,
) -> None:
    filename = Path(urlparse(url).path).name
    target = out / filename
    if target.is_file() and sha256(target) == digest:
        print(f"cached {filename}", flush=True)
        return
    target.unlink(missing_ok=True)
    size = _remote_size(url)
    if (
        size is not None
        and size >= range_threshold
        and _supports_ranges(url, size)
    ):
        print(
            f"downloading {filename} in "
            f"{len(chunk_specs(size, chunk_size))} chunks with {workers} workers",
            flush=True,
        )
        parallel_download(url, digest, target, size, curl, chunk_size, workers)
    else:
        print(f"downloading {filename} with resumable serial transfer", flush=True)
        serial_download(url, digest, target, curl)
    print(f"downloaded {filename}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pypi", required=True)
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--curl", default="curl")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--range-threshold",
        type=int,
        default=DEFAULT_RANGE_THRESHOLD,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    indexes = [args.pypi, args.pytorch]
    with DirectoryLock(args.out):
        for lock_name in ("corp.lock.txt", "robocasa.lock.txt", "lerobot.lock.txt"):
            for name, version, hashes in parse_lock(
                args.root / "environment" / lock_name
            ):
                url, digest = find_hashed_artifact(
                    name,
                    version,
                    hashes,
                    indexes,
                )
                download(
                    url,
                    digest,
                    args.out,
                    args.curl,
                    chunk_size=args.chunk_size,
                    range_threshold=args.range_threshold,
                    workers=args.workers,
                )
        url, digest = find_extra_wheel("robosuite", "1.5.2", args.pypi)
        download(
            url,
            digest,
            args.out,
            args.curl,
            chunk_size=args.chunk_size,
            range_threshold=args.range_threshold,
            workers=args.workers,
        )
        for name in (
            "corp.lock.txt",
            "robocasa.lock.txt",
            "lerobot.lock.txt",
            "upstream.lock.json",
        ):
            source = args.root / "environment" / name
            (args.out / name).write_bytes(source.read_bytes())
        artifacts = sorted(
            path
            for path in args.out.iterdir()
            if path.is_file() and compatible_artifact(path.name)
        )
        sums = [f"{sha256(artifact)}  {artifact.name}\n" for artifact in artifacts]
        (args.out / "SHA256SUMS").write_text("".join(sums))
    print(f"Wheelhouse ready: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
