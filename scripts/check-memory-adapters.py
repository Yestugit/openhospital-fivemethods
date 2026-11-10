"""Run adapter regressions from either a flat checkout or an installed package.

Example: .venv-adapters/Scripts/python scripts/check-memory-adapters.py test/native_adapters
Use --unittest FILE.py for native environments without pytest (MemOS/MemMachine).
No provider credentials are loaded and no experiment is launched by this script.
"""
import importlib.util
import os
from pathlib import Path
import sys
import shutil
from uuid import uuid4


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    package = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("mematlas", package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["mematlas"] = module
    spec.loader.exec_module(module)
    # Public hospital fixture tests require a present key at preflight; their
    # transport/execution is mocked. Never source the user's real .env here.
    os.environ.setdefault("MODEL_API_KEY", "offline-adapter-test-placeholder")
    arguments = sys.argv[1:] or ["test/native_adapters"]
    keep_temp = "--keep-temp" in arguments
    arguments = [arg for arg in arguments if arg != "--keep-temp"]
    if arguments[0] == "--unittest":
        import unittest
        if len(arguments) == 1:
            raise SystemExit("--unittest requires one or more test file paths")
        suite = unittest.TestSuite()
        for item in arguments[1:]:
            path = Path(item).resolve()
            name = ".".join(("mematlas", *path.relative_to(package).with_suffix("").parts))
            source = importlib.util.spec_from_file_location(name, path)
            tests = importlib.util.module_from_spec(source)
            sys.modules[name] = tests
            source.loader.exec_module(tests)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    import pytest
    temporary = package / '.mematlas' / ('adapter-tests-' + uuid4().hex)
    result = pytest.main([*arguments, "-o", "asyncio_mode=auto", "-o",
                        "asyncio_default_fixture_loop_scope=function",
                        f"--basetemp={temporary}"])
    # Successful disposable fixtures are not experiment evidence. Keep failed
    # runs for diagnosis and offer an explicit opt-out for fixture inspection.
    if result == 0 and not keep_temp and temporary.is_dir():
        try:
            target = Path('\\\\?\\' + str(temporary)) if os.name == 'nt' else temporary
            shutil.rmtree(target)
        except OSError as exc:
            print(f"Temporary test files retained: {temporary} ({type(exc).__name__})")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
