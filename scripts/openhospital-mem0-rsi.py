"""Run from either a flat MemAtlas checkout or a src/mematlas checkout."""
from pathlib import Path
import sys

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package.parent))
if package.name != "mematlas":
    # The repository folder name need not equal the Python package name.
    import importlib.util
    spec = importlib.util.spec_from_file_location("mematlas", package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["mematlas"] = module
    spec.loader.exec_module(module)

from mematlas.integrations.openhospital.suite import main

if __name__ == "__main__":
    raise SystemExit(main())
