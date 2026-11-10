"""Verify the installed native source without importing its side-effectful package."""
from hashlib import sha256
from importlib import metadata
from importlib.util import find_spec
import json
from pathlib import Path

REVISION = "84bf468176f0c5e82493bb95aec5484eb2d92bc1"
IDENTITY = f"mem0ai/mem0@{REVISION}"


def verify_source() -> list[str]:
    errors = []
    record = json.loads(Path(__file__).with_name("source.json").read_text(encoding="utf-8"))
    try:
        distribution = metadata.distribution("mem0ai")
    except metadata.PackageNotFoundError:
        return ["Install methods/mem0/requirements-lock.txt in a dedicated interpreter"]
    if distribution.version != record["version"]:
        errors.append(f"mem0ai must be {record['version']} from {IDENTITY}")
    spec = find_spec("mem0")
    expected_init = Path(distribution.locate_file("mem0/__init__.py")).resolve()
    if spec is None or not spec.origin or Path(spec.origin).resolve() != expected_init:
        errors.append("The imported mem0 package is shadowed; use the verified dedicated interpreter")
    for name, expected in record["source_files_sha256"].items():
        path = Path(distribution.locate_file(name))
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Native source hash mismatch: {name}")
    for package, version in record["runtime_versions"].items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            errors.append(f"{package} must be {version}; installed={actual!r}")
    return errors
