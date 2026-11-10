"""
Value-driven RL configuration and utilities.

Key components:
- RLConfig: configuration for epsilon-greedy, thresholds, and Q-learning.
- _meta_to_dict / _now_iso / _get_similarity: shared utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RLConfig:
    epsilon: float = 0.1           # ε-greedy exploration probability
    tau: float = 0.35              # unknown detection threshold on similarity
    alpha: float = 0.1             # Q-learning step size
    gamma: float = 0.0             # discount factor (single-step default)
    lambd: float = 0.8             # λ for TD(λ) eligibility traces (0=TD(0), 1=MC)
    q_init_pos: float = 0        # optimistic initialization
    q_init_neg: float = 0          # negative q init
    q_floor: Optional[float] = None  # optional minimum Q value (disabled by default)
    q_ceiling: Optional[float] = None  # optional maximum Q value (disabled by default)
    success_reward: float = 1.0
    failure_reward: float = -1.0
    sim_threshold: float = 0.5     # retrieval filtering threshold (used by some runners)
    topk: int = 5                  # candidate set size for value-aware selection
    novelty_threshold: float = 0.85  # similarity to treat as non-novel (merge)
    weight_sim: float = 0.5        # weight for similarity in combined score
    weight_q: float = 0.5          # weight for Q-value in combined score
    structural_max_depth: int = 7  # max BFS depth for structural trace decay
    normalize_dq: bool = False     # divide accumulated ΔQ by contribution count before flush
    q_init_adaptive: bool = False  # use mean Q of retrieved memories instead of fixed q_init
    alpha_min: float = 0.0         # minimum alpha at end of cosine annealing
    alpha_schedule: str = "constant"  # "constant" or "cosine"


# ----------------- Utilities -----------------

def _meta_to_dict(meta: Any) -> Dict[str, Any]:
    """Convert TextualMemoryMetadata or dict-like to a plain dict safely."""
    if meta is None:
        return {}
    try:
        if hasattr(meta, "model_extra"):
            return dict(meta.model_dump())
    except Exception:
        pass
    try:
        if isinstance(meta, dict):
            return dict(meta)
    except Exception:
        pass
    # best effort fallback
    try:
        return dict(getattr(meta, "__dict__", {}))
    except Exception:
        return {}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _get_similarity(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("similarity", 0.0))
    except Exception:
        return 0.0
