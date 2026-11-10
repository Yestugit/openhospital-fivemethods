"""Fetch immutable EMem/Memora source files, checking the committed SHA-256 list.

This only obtains code. It never downloads model weights or installs packages.
Existing differing files are preserved and reported as errors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import os
from urllib.request import Request, urlopen


def fetch_source(method_id, *, root=None, destination=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    spec = json.loads((root / "methods" / method_id / "source.json").read_text(encoding="utf-8"))
    target = Path(destination or root / ".mematlas" / "upstream" / method_id).resolve()
    repository = spec["repository_url"].removeprefix("https://github.com/").rstrip("/")
    revision = spec["revision"]
    expected_files = spec["sha256"]

    def obtain(item):
        name, checksum = item
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source path: {name}")
        path = (target / name).resolve()
        if not path.is_relative_to(target):
            raise ValueError(f"Source path escapes destination: {name}")
        if path.exists():
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Preserved differing existing source: {path}")
            return
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/{name}"
        with urlopen(Request(url, headers={"User-Agent": "MemAtlas-native-adapters"}), timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f"Upstream source checksum mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".source-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
            # Exclusive creation also preserves a concurrent writer's file.
            try:
                with path.open("xb") as stream:
                    stream.write(Path(temporary).read_bytes())
            except FileExistsError:
                if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                    raise ValueError(f"Preserved concurrently changed source: {path}")
        finally:
            Path(temporary).unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(obtain, expected_files.items()))
    return target, len(expected_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("emem", "memora"))
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    target, count = fetch_source(args.method, destination=args.destination)
    print(f"Verified {count} pinned source files: {target}")


if __name__ == "__main__":
    main()
