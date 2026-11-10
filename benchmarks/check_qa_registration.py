"""Check that all three native slices resolve their dedicated QA adapters.

Run with::

    PYTHONPATH=/home/lzy/project .venv/bin/python benchmarks/check_qa_registration.py
"""
from __future__ import annotations

from mematlas.core.registry import default_registry
from mematlas.evaluation.qa import _qa_type


EXPECTED = {
    "reasoningbank": "ReasoningBankQA",
    "ael": "AELQA",
    "memcon": "MemConQA",
}


def run() -> dict[str, str]:
    result = {method_id: _qa_type(default_registry().get(method_id)).__name__
              for method_id in EXPECTED}
    assert result == EXPECTED, result
    return result


if __name__ == "__main__":
    print(f"QA_REGISTRATION {run()}")
