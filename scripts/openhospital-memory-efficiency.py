"""Five native methods, one serial OpenHospital efficiency experiment."""
from pathlib import Path
import importlib.util
import sys

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mematlas", root / "__init__.py", submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules["mematlas"] = module
spec.loader.exec_module(module)
from mematlas.integrations.openhospital.efficiency import main

if __name__ == "__main__":
    raise SystemExit(main())
