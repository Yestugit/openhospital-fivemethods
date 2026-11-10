"""Deploy Qwen3-Embedding-0.6B from Hugging Face or serve it locally."""
from pathlib import Path
import sys
import importlib.util

package = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mematlas", package / "__init__.py",
                                            submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules["mematlas"] = module
spec.loader.exec_module(module)

from mematlas.providers.local_qwen import main

if __name__ == "__main__":
    raise SystemExit(main())
