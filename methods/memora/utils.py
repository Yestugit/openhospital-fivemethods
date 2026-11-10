"""Shared public-contract glue (not a memory algorithm)."""
from mematlas.methods.emem.utils import (
    EpisodeNativeAdapter, canonical, checked_source, digest, get_blob,
    namespace, put_blob, run_service, storage_path,
)


def put_json_array_blob(directory, rows):
    """Stream the identical canonical array, avoiding a second full vector copy."""
    import hashlib
    import os
    from pathlib import Path
    import tempfile

    directory = storage_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checksum, size = hashlib.sha256(), 0
    fd, name = tempfile.mkstemp(dir=directory, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            def write(data):
                nonlocal size
                stream.write(data)
                checksum.update(data)
                size += len(data)
            write(b"[")
            for index, row in enumerate(rows):
                if index:
                    write(b",")
                write(canonical(row))
            write(b"]")
        key = checksum.hexdigest()
        target = directory / key
        if target.exists():
            actual = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(chunk)
            if target.stat().st_size != size or actual.hexdigest() != key:
                raise ValueError(f"Corrupt immutable checkpoint: {target}")
        else:
            os.replace(name, target)
        return {"path": str(target.resolve()), "sha256": key, "size": size}
    finally:
        Path(name).unlink(missing_ok=True)

