"""CLI diagnostics: log origins and a guard on Python plan/ open/import calls.

Load by prepending this directory to PYTHONPATH. Providers and algorithm calls
are not replaced. MEMATLAS_E2E_REPO must point to the checkout under test.
"""
from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
import sys


class CredentialSafeFormatter(logging.Formatter):
    def format(self, record):
        result = super().format(record)
        for name, value in os.environ.items():
            if (name.endswith("_API_KEY") or name.endswith("_TOKEN")) and len(value) >= 8:
                result = result.replace(value, "[REDACTED]")
        return result


handler = logging.StreamHandler()
handler.setFormatter(CredentialSafeFormatter("%(levelname)s %(name)s | %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# MemOS repeatedly replaces root handlers and filters out non-MemOS records.
# Keep one named diagnostic handler after that logging-only reconfiguration.
original_dict_config = logging.config.dictConfig


def diagnostic_dict_config(config):
    original_dict_config(config)
    for name in ("mematlas.methods", "memrl", "memq", "ael"):
        logger = logging.getLogger(name)
        logger.disabled = False
        logger.setLevel(logging.INFO)
        logger.handlers = [handler]
        logger.propagate = False


logging.config.dictConfig = diagnostic_dict_config

repo = Path(os.environ["MEMATLAS_E2E_REPO"]).resolve()
forbidden = str(repo / "plan")


def guard_plan(event, args):
    candidate = None
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        candidate = os.fsdecode(args[0])
    elif event == "import" and len(args) > 1 and isinstance(args[1], str):
        candidate = args[1]
    if candidate is not None:
        absolute = os.path.abspath(candidate)
        if absolute == forbidden or absolute.startswith(forbidden + os.sep):
            raise PermissionError("CLI E2E forbids runtime access to plan/: " + absolute)


sys.addaudithook(guard_plan)
logging.getLogger("mematlas.test.cli").info("plan/ access blocked; real providers enabled")
