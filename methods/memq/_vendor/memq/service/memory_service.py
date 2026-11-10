"""
Core MemoryService implementation for the Memp procedural memory system.

This module provides the central MemoryService class that orchestrates the
Build/Retrieve/Update memory management strategies with an in-memory vector store.
"""

import logging
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..utils.task_id import extract_task_id
from .builders import get_builder
from .keyer import AveFactKeyer, SimpleKeyer
from .memory_store import MemoryStore
from .procedural_memory import ProceduralMemory
from .retrievers import get_retriever
from .strategies import (
    BuildStrategy,
    RetrieveStrategy,
    StrategyConfiguration,
)
from .updater import get_updater

# Configure logger
logger = logging.getLogger(__name__)
from ..providers.base import BaseEmbedder, BaseLLM
from .value_driven import RLConfig, _meta_to_dict


def _now_iso() -> str:
    return datetime.now().isoformat()


def get_embedding_with_retry(embed_func, text, max_retries=5, base_delay=2.0):
    """
    Retry for embedding queries
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = embed_func(text)
            return result

        except Exception as e:
            logger.warning(
                f"[Retry {attempt}/{max_retries}] Embedding API 调用失败: {e}"
            )
            if attempt == max_retries:
                logger.error("达到最大重试次数，仍未成功。")
                raise

            sleep_time = base_delay * (2 ** (attempt - 1))
            time.sleep(sleep_time)


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


def _parse_datetime(value: Any) -> datetime | None:
    """Best-effort parse of various datetime string formats."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Try python's ISO parser first
        try:
            return datetime.fromisoformat(text)
        except Exception:
            pass
        # Fallback patterns
        patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in patterns:
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
    return None


def _coerce_success(value: Any) -> Optional[bool]:
    """Parse a best-effort boolean success flag.

    Returns:
        True / False if value can be reliably interpreted, otherwise None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return None
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
        # Treat "unknown"/"n/a" and other strings as unknown
        return None
    return None


class MemoryService:
    """
    Core memory service implementing the Memp procedural memory algorithms.

    This service orchestrates the three-phase memory management:
    1. Build: Construct procedural memories from task trajectories
    2. Retrieve: Find relevant memories for new tasks
    3. Update: Learn from new experiences and adjust existing memories

    The service uses an in-memory MemoryStore for storage and retrieval operations.
    """

    def __init__(
        self,
        llm_provider: BaseLLM,
        embedding_provider: BaseEmbedder,
        strategy_config: Optional[StrategyConfiguration] = None,
        user_id: str = "memp_user",
        num_workers: int = 32,
        **kwargs,
    ):
        """
        Initialize the memory service.

        Args:
            llm_provider: LLM provider for text generation and keyword extraction
            embedding_provider: Embedding provider for vector operations
            strategy_config: Strategy configuration (defaults to main combination)
            user_id: User ID for multi-tenant memory management
            **kwargs: Additional configuration parameters
        """
        # Set strategy configuration
        self.strategy_config = (
            strategy_config or StrategyConfiguration.main_combination()
        )
        self.user_id = user_id
        self.num_workers = num_workers
        import threading

        self.db_max_concurrency: int = int(kwargs.get("db_max_concurrency", 4))
        self._db_gate = threading.BoundedSemaphore(self.db_max_concurrency)
        # TD(λ) eligibility traces for memq algorithm
        self._eligibility_traces: dict[str, float] = {}
        self._accumulated_dq: dict[str, float] = {}  # accumulated ΔQ for epoch-end write
        self._accumulated_dq_count: dict[str, int] = {}  # contribution count per memory (for normalize_dq)
        # Structural decay: DAG reference shared by runner (new_mem_id -> [parent_ids])
        self._memid_pair: Optional[dict] = None
        # Store providers
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

        # Q values can be max-scaled before mixing with raw cosine similarity.
        self.normalize_q: bool = bool(kwargs.get("normalize_q", True))
        # Cosine annealing epoch tracking
        self._current_epoch: int = 0
        self._total_epochs: int = 1
        # Optional: LLB-only Phase-B behavior to reduce repeated same-task memories.
        self.dedup_by_task_id: bool = bool(kwargs.get("dedup_by_task_id", False))

        # Initialize in-memory vector store
        self._store = MemoryStore(vector_dim=int(kwargs.get("vector_dim", 4096)))

        # Configuration parameters (must be set before _init_key_generators)
        self.max_keywords = kwargs.get("max_keywords", 8)
        self.add_similarity_threshold = kwargs.get("add_similarity_threshold", 0.90)
        # Value-driven config and components (optional)
        self.enable_value_driven: bool = bool(kwargs.get("enable_value_driven", True))
        self.rl_config: Optional[RLConfig] = kwargs.get("rl_config", None)
        if self.enable_value_driven and self.rl_config is None:
            self.rl_config = RLConfig()

        # Cached max Q for max-scaling normalization (auto-computed from store)
        self._q_max: float = 0.0

        # Default retrieval weights (overridden below if value-driven is enabled)
        self.weight_sim = 1.0
        self.weight_q = 0.0

        # Initialize key generators for different retrieval strategies
        self._init_key_generators()

        if self.enable_value_driven and self.rl_config is not None:
            try:
                self.dict_memory = {}
                self.query_embeddings = {}
                self.weight_sim = self.rl_config.weight_sim
                self.weight_q = self.rl_config.weight_q
            except Exception:
                # Fail gracefully; callers can still use baseline flows
                self.enable_value_driven = False

    def _init_key_generators(self) -> None:
        """Initialize key generators for retrieval strategies."""
        self.avefact_keyer = AveFactKeyer(
            self.llm_provider, self.embedding_provider, max_keywords=self.max_keywords
        )
        self.simple_keyer = SimpleKeyer(self.embedding_provider)

    def _apply_q_update(self, memory_id: str, reward: float, next_max_q: float | None = None) -> float:
        """Single-step Q-learning update on a memory's metadata. Returns new Q."""
        item = self._store.get(memory_id)
        if item is None:
            raise RuntimeError(f"Memory {memory_id} not found")
        old_meta = _meta_to_dict(item.get("metadata"))
        old_q = float(old_meta.get("q_value", self.rl_config.q_init_pos))
        target = float(reward) + (self.rl_config.gamma * float(next_max_q or 0.0))
        new_q = (1.0 - self.get_alpha()) * old_q + self.get_alpha() * target
        if self.rl_config.q_floor is not None:
            new_q = max(float(self.rl_config.q_floor), new_q)
        if self.rl_config.q_ceiling is not None:
            new_q = min(float(self.rl_config.q_ceiling), new_q)
        visits = int(old_meta.get("q_visits", 0)) + 1
        old_ma = float(old_meta.get("reward_ma", 0.0))
        reward_ma = (1.0 - self.get_alpha()) * old_ma + self.get_alpha() * float(reward)
        new_meta = old_meta | {
            "q_value": float(new_q), "q_visits": visits, "last_reward": float(reward),
            "reward_ma": reward_ma, "q_updated_at": _now_iso(), "last_used_at": _now_iso(),
        }
        self._store.update(memory_id, {"id": memory_id, "memory": item.get("memory", ""), "metadata": new_meta})
        return float(new_q)

    def build_memory(
        self,
        task_description: str,
        trajectory: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build procedural memory from task trajectory using the configured build strategy.

        Args:
            task_description: Natural language description of the task
            trajectory: Detailed step-by-step trajectory of task execution
            metadata: Optional additional metadata

        Returns:
            Memory ID of the created memory

        Raises:
            RuntimeError: If memory building fails
        """
        try:
            # Extract source benchmark from metadata
            source_benchmark = (metadata or {}).get("source_benchmark", "unknown")
            raw_success = (metadata or {}).get("success", None)
            success = _coerce_success(raw_success)

            # Use Builder pattern to generate memory content
            builder = get_builder(self.strategy_config.build, self.llm_provider)
            memory_content = builder.build(task_description, trajectory)

            # Create procedural memory based on build strategy
            if self.strategy_config.build == BuildStrategy.TRAJECTORY:
                procedural_memory = ProceduralMemory.create_trajectory_memory(
                    task_description=task_description,
                    trajectory=memory_content,  # Use builder output
                    build_strategy=self.strategy_config.build,
                    retrieve_strategy=self.strategy_config.retrieve,
                    update_strategy=self.strategy_config.update,
                    source_benchmark=source_benchmark,
                )

            elif self.strategy_config.build == BuildStrategy.SCRIPT:
                procedural_memory = ProceduralMemory.create_script_memory(
                    task_description=task_description,
                    script=memory_content,  # Use builder output
                    trajectory=trajectory,  # Keep original trajectory for reference
                    build_strategy=self.strategy_config.build,
                    retrieve_strategy=self.strategy_config.retrieve,
                    update_strategy=self.strategy_config.update,
                    source_benchmark=source_benchmark,
                )

            else:  # BuildStrategy.PROCEDURALIZATION
                # For proceduralization, the builder returns combined content
                # Extract script from the combined content for metadata
                lines = memory_content.split("\n")
                script_start = None
                trajectory_start = None

                for i, line in enumerate(lines):
                    if line.strip() == "SCRIPT:":
                        script_start = i + 1
                    elif line.strip() == "TRAJECTORY:":
                        trajectory_start = i + 1
                        break

                if script_start is not None and trajectory_start is not None:
                    script = "\n".join(
                        lines[script_start : trajectory_start - 2]
                    ).strip()
                else:
                    # Fallback: generate script separately if parsing fails
                    script = self.llm_provider.generate_script(trajectory)

                procedural_memory = ProceduralMemory.create_procedural_memory(
                    task_description=task_description,
                    script=script,
                    trajectory=trajectory,
                    build_strategy=self.strategy_config.build,
                    retrieve_strategy=self.strategy_config.retrieve,
                    update_strategy=self.strategy_config.update,
                    source_benchmark=source_benchmark,
                )

            # Generate retrieval keys if needed
            if self.strategy_config.retrieve == RetrieveStrategy.AVEFACT:
                keywords = self.llm_provider.extract_keywords(
                    task_description, self.max_keywords
                )
                avefact_vector = self.avefact_keyer.generate_key(task_description)
                procedural_memory.memp_metadata.avefact_keywords = keywords
                procedural_memory.memp_metadata.avefact_vector = avefact_vector
            elif self.strategy_config.retrieve == RetrieveStrategy.QUERY:
                query_vector = self.simple_keyer.generate_key(task_description)
                procedural_memory.memp_metadata.query_vector = query_vector

            # Write using _store with retrieval key = task_description (embedding),
            # and full content stored in metadata.full_content
            full_content = (
                f"Task: {task_description}\n\n{procedural_memory.memory_content}"
            )

            # Build metadata and item
            base_meta = {
                "type": "procedure",
                "source": "conversation",
                "source_benchmark": source_benchmark,
                "success": success,
                "strategy_build": self.strategy_config.build.value,
                "strategy_retrieve": self.strategy_config.retrieve.value,
                "strategy_update": self.strategy_config.update.value,
                "full_content": full_content,
            }
            # Initialize value fields when enabled
            if (
                getattr(self, "enable_value_driven", False)
                and getattr(self, "rl_config", None) is not None
            ):
                # User-selected behavior: unknown success defaults to q_init_pos.
                is_success = True if success is None else bool(success)
                base_meta |= {
                    "q_value": (
                        float(self.rl_config.q_init_pos)
                        if is_success
                        else float(self.rl_config.q_init_neg)
                    ),
                    "q_visits": 0,
                    "retrieval_count": 0,
                    "q_updated_at": datetime.now().isoformat(),
                    "last_used_at": datetime.now().isoformat(),
                    "reward_ma": 0.0,
                }

            import uuid
            mem_id = str(uuid.uuid4())
            vector = self.embedding_provider.embed([task_description])[0]
            payload = {"id": mem_id, "memory": task_description, "metadata": base_meta}
            self._store.add(mem_id, vector, payload)
            return mem_id

        except Exception as e:
            raise RuntimeError(f"Failed to build memory: {e}")

    def retrieve(
        self,
        task_description: str,
        k: int = 1,
        threshold: float = 0.0,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a task using the configured retrieval strategy.

        Args:
            task_description: Natural language description of the task
            k: Number of memories to retrieve
            threshold: Minimum similarity threshold (for applicable strategies)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay (in seconds) between retries (exponential backoff)

        Returns:
            List of retrieved memory dictionaries

        Raises:
            RuntimeError: If memory retrieval fails after retries
        """
        retriever = get_retriever(
            self.strategy_config.retrieve,
            store=self._store,
            user_id=self.user_id,
            llm=self.llm_provider,
            keyer=self.avefact_keyer,
            max_keywords=self.max_keywords,
            embedder=self.embedding_provider,
        )

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return retriever.retrieve(task_description, k, threshold)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    logger.warning(
                        f"Retrieve attempt {attempt} failed: {e}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Retrieve failed after {max_retries} attempts. Last error: {e}"
                    )
                    raise RuntimeError(
                        f"Failed to retrieve memories after {max_retries} attempts: {last_error}"
                    )

    def update_value(
        self,
        memory_id: Optional[str],
        reward: float,
        *,
        next_max_q: Optional[float] = None
    ) -> Optional[float]:
        """
        Update Q-value for the selected memory. If memory_id is None (null action),
        this is a no-op here (curation can be applied by caller if needed).

        Returns the new Q if updated, else None.
        """
        if not getattr(self, 'enable_value_driven', False):
            return None
        if memory_id is None:
            return None
        try:
            return self._apply_q_update(memory_id, reward, next_max_q=next_max_q)
        except Exception as e:
            raise RuntimeError(f"Failed to update Q-value: {e}")


    def update_values(self, successes: list[float], retrieved_ids_list: list[list[str]]) -> dict[str, Optional[float]]:
        """
        Concurrently update Q-values for all retrieved memory_ids.

        Args:
            successes: list of rewards (one per trajectory)
            retrieved_ids_list: list of lists of memory_ids (aligned with successes)

        Returns:
            dict mapping memory_id -> new Q value (or None if failed).
        """
        if not getattr(self, "enable_value_driven", False):
            return {}

        # Build update tasks: (memory_id, reward, next_max_q)
        updates = []
        for success, mem_ids in zip(successes, retrieved_ids_list):
            # Map success to 1 (True) or -1 (False/0)
            reward = (
                self.rl_config.success_reward
                if success
                else self.rl_config.failure_reward
            )
            for mem_id in mem_ids:
                updates.append((mem_id, reward, None))

        results = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_mem = {
                executor.submit(
                    self._apply_q_update, mem_id, reward, next_max_q
                ): mem_id
                for mem_id, reward, next_max_q in updates
            }
            for future in as_completed(future_to_mem):
                mem_id = future_to_mem[future]
                try:
                    new_q = future.result()
                    results[mem_id] = new_q
                except Exception as e:
                    results[mem_id] = None
                    logger.info(f"Failed to update Q-value for {mem_id}: {e}")
        return results

    def set_memid_pair(self, memid_pair: dict) -> None:
        """Share the runner's memory reference DAG by reference for structural decay.

        Args:
            memid_pair: OrderedDict mapping new_mem_id -> [parent_ids].
        """
        self._memid_pair = memid_pair

    def reset_traces(self, keep_eligibility: bool = False) -> None:
        """Clear accumulated ΔQ (called at epoch boundaries)."""
        if not keep_eligibility:
            self._eligibility_traces.clear()
            logger.info("[MemQ] Reset eligibility traces and accumulated ΔQ")
        else:
            logger.info(
                f"[MemQ] Reset accumulated ΔQ, keeping {len(self._eligibility_traces)} "
                f"eligibility traces for cross-epoch credit"
            )
        self._accumulated_dq.clear()
        self._accumulated_dq_count.clear()

    def record_retrieval_traces(self, retrieved_ids_list: list[list[str]]) -> None:
        """No-op in structural mode (traces are computed from DAG, not accumulated)."""
        return

    def decay_traces(self) -> None:
        """No-op in structural mode (no temporal trace accumulation)."""
        return

    def accumulate_memq_updates(
        self,
        td_errors: list[float],
        retrieved_ids_list: Optional[List[List[str]]] = None,
    ) -> None:
        """Step C: Accumulate ΔQ from TD errors (per mini-batch, in-memory only).

        Uses structural mode: aggregate every-path credit with frontier DP.

        Args:
            td_errors: List of TD error values (one per task in mini-batch).
            retrieved_ids_list: Per-task list of retrieved memory IDs.
        """
        if self.rl_config is None:
            return
        self._accumulate_structural(td_errors, retrieved_ids_list)

    @staticmethod
    def _provenance_path_credits(
        root_mem_id: str,
        dag: Optional[dict],
        decay: float,
        max_depth: int,
    ) -> Dict[str, float]:
        """Aggregate credit from every bounded provenance path from one root.

        ``frontier[m]`` is the sum of the already-discounted weights of all
        paths that reach ``m`` at the current depth.  Aggregating by
        ``(memory, depth)`` retains path multiplicity without explicitly
        enumerating an exponential number of paths.  Credits for the same
        memory at different depths are added to implement::

            C(m) = sum((gamma * lambda) ** len(path) for path in root -> m)

        Parent IDs are de-duplicated within each adjacency list so duplicated
        storage entries are not interpreted as parallel provenance edges.

        Provenance is expected to be a DAG.  A malformed cycle cannot make
        this traversal infinite because expansion is strictly bounded by
        ``max_depth``; within that bound, cyclic walks retain the same
        every-path weighting semantics.
        """
        if not root_mem_id or max_depth < 0:
            return {}

        credits: Dict[str, float] = {}
        frontier: Dict[str, float] = {root_mem_id: 1.0}
        for depth in range(max_depth + 1):
            if not frontier:
                break

            for mem_id, path_weight in frontier.items():
                credits[mem_id] = credits.get(mem_id, 0.0) + path_weight

            if depth == max_depth or dag is None:
                break

            next_frontier: Dict[str, float] = {}
            for mem_id, path_weight in frontier.items():
                parents = dag.get(mem_id) or []
                for parent_id in dict.fromkeys(parents):
                    if not parent_id:
                        continue
                    contribution = path_weight * decay
                    if abs(contribution) < 1e-12:
                        continue
                    next_frontier[parent_id] = (
                        next_frontier.get(parent_id, 0.0) + contribution
                    )
            frontier = next_frontier

        return credits

    def _accumulate_provenance_root(
        self,
        root_mem_id: str,
        delta: float,
        *,
        alpha: float,
        decay: float,
        max_depth: int,
        dag: Optional[dict],
    ) -> None:
        """Accumulate one root's every-path credit as one event per target.

        ``_accumulated_dq_count`` counts a root-to-memory aggregate event, not
        the number of paths or depths contributing to that event.  Therefore
        ``normalize_dq`` can average independent root/task evidence without
        cancelling the intended all-path multiplicity inside ``C(m)``.
        """
        credits = self._provenance_path_credits(
            root_mem_id,
            dag,
            decay,
            max_depth,
        )
        for mem_id, path_credit in credits.items():
            increment = alpha * delta * path_credit
            self._accumulated_dq[mem_id] = (
                self._accumulated_dq.get(mem_id, 0.0) + increment
            )
            self._accumulated_dq_count[mem_id] = (
                self._accumulated_dq_count.get(mem_id, 0) + 1
            )

    def _accumulate_structural(
        self,
        td_errors: list[float],
        retrieved_ids_list: Optional[List[List[str]]] = None,
    ) -> None:
        """Propagate credit along every path in the memory provenance DAG.

        For each task j with TD error δ_j and retrieved set A_j:
            For each m in A_j, aggregate all backward provenance paths:
                depth 0 (m itself):     ΔQ(m)  += α · δ_j · (γλ)^0
                depth 1 (parents):      ΔQ(p)  += α · δ_j · (γλ)^1
                depth 2 (grandparents): ΔQ(gp) += α · δ_j · (γλ)^2
                ...up to structural_max_depth

        If _memid_pair is None or memory not in DAG: depth-0 only (graceful fallback).
        Multiple paths to the same memory are summed, including paths of
        different lengths.  Frontier aggregation avoids explicit path enumeration.
        """
        if self.rl_config is None or retrieved_ids_list is None:
            return
        alpha = self.get_alpha()
        gamma = self.rl_config.gamma
        lambd = self.rl_config.lambd
        max_depth = getattr(self.rl_config, "structural_max_depth", 7)
        dag = self._memid_pair  # may be None

        for j, delta in enumerate(td_errors):
            if abs(delta) < 1e-12:
                continue
            if j >= len(retrieved_ids_list):
                continue
            retrieved_ids = retrieved_ids_list[j]
            if not retrieved_ids:
                continue

            # Each retrieved memory is an independent direct-credit root.
            for root_mem_id in retrieved_ids:
                self._accumulate_provenance_root(
                    root_mem_id,
                    delta,
                    alpha=alpha,
                    decay=gamma * lambd,
                    max_depth=max_depth,
                    dag=dag,
                )

    def _update_memory_item_memq(self, mem_id: str, q_increment: float) -> Optional[tuple]:
        """Apply accumulated ΔQ to a single memory item.

        Reads old Q, computes new_q = clamp(old_q + increment, q_floor, q_ceiling),
        writes back to store.

        Returns:
            (mem_id, new_q_value) on success, None on failure.
        """
        try:
            item = self._store.get(mem_id)
            if item is None:
                return None

            old_meta = _meta_to_dict(item.get("metadata"))
            old_q = float(old_meta.get("q_value", self.rl_config.q_init_pos if self.rl_config else 0.0))

            new_q = old_q + q_increment

            # Clamp to [q_floor, q_ceiling]
            if self.rl_config is not None:
                if self.rl_config.q_floor is not None:
                    new_q = max(float(self.rl_config.q_floor), new_q)
                if self.rl_config.q_ceiling is not None:
                    new_q = min(float(self.rl_config.q_ceiling), new_q)

            new_meta = old_meta | {
                "q_value": float(new_q),
                "q_updated_at": _now_iso(),
                "last_used_at": _now_iso(),
            }

            memory_text = item.get("memory", "")
            self._store.update(mem_id, {"id": mem_id, "memory": memory_text, "metadata": new_meta})

            return mem_id, float(new_q)
        except Exception as e:
            logger.error(f"[MemQ] Unable to update memory {mem_id}: {e}")
            return None

    def compute_bellman_td_and_accumulate(
        self,
        edges: List[Dict[str, Any]],
        memid_pair: dict,
    ) -> Tuple[float, float, int]:
        """Backward-pass Bellman TD: compute per-edge δ = R + γ·Q(child) - Q(parent) and accumulate ΔQ.

        Called at epoch end when all new memories exist in the store.

        Each edge represents a task that retrieved parent memories and created a child memory:
            edge = {"reward": float, "retrieved_ids": [parent_ids], "new_mem_id": child_id | None}

        For each (parent ← child) edge:
            δ = R + γ · Q(child) - Q(parent)
            ΔQ(parent) += α · δ  (+ every-path provenance propagation)

        Args:
            edges: Per-task edge records collected during mini-batches.
            memid_pair: DAG mapping child_mem → [parent_mems].

        Returns:
            (sum_td, sum_td_sq, count) for TensorBoard logging of bias/variance.
        """
        if self.rl_config is None:
            return 0.0, 0.0, 0

        alpha = self.get_alpha()
        gamma = self.rl_config.gamma
        lambd = self.rl_config.lambd
        max_depth = getattr(self.rl_config, "structural_max_depth", 7)
        dag = memid_pair if memid_pair is not None else self._memid_pair

        sum_td_sq = 0.0
        sum_td = 0.0
        td_count = 0

        for edge in edges:
            reward = edge["reward"]
            retrieved_ids = edge.get("retrieved_ids") or []
            child_id = edge.get("new_mem_id")

            # Get child Q (next-state value)
            child_q = 0.0
            if child_id and gamma > 0:
                child_item = self._store.get(child_id)
                if child_item is not None:
                    child_meta = _meta_to_dict(child_item.get("metadata"))
                    try:
                        child_q = float(child_meta.get("q_value", 0.0))
                    except (TypeError, ValueError):
                        child_q = 0.0

            # Compute per-edge δ for each parent
            for parent_id in retrieved_ids:
                if not parent_id:
                    continue
                parent_item = self._store.get(parent_id)
                if parent_item is None:
                    continue
                parent_meta = _meta_to_dict(parent_item.get("metadata"))
                try:
                    parent_q = float(parent_meta.get("q_value", 0.0))
                except (TypeError, ValueError):
                    parent_q = 0.0

                delta = reward + gamma * child_q - parent_q

                sum_td += delta
                sum_td_sq += delta * delta
                td_count += 1

                self._accumulate_provenance_root(
                    parent_id,
                    delta,
                    alpha=alpha,
                    decay=gamma * lambd,
                    max_depth=max_depth,
                    dag=dag,
                )

        logger.info(
            f"[MemQ] Computed {td_count} per-edge TD errors, "
            f"mean_δ²={sum_td_sq / td_count:.4f}" if td_count > 0 else "[MemQ] No edges",
        )
        return sum_td, sum_td_sq, td_count

    def flush_memq_updates(self) -> Dict[str, Optional[float]]:
        """Write phase: apply all accumulated ΔQ to store (called at epoch end).

        For each memory with accumulated ΔQ:
        - Read old Q from store payload
        - new_q = clamp(old_q + accumulated_dq[m], q_floor, q_ceiling)
        - Write to store via ThreadPoolExecutor

        Returns:
            Dict mapping memory_id -> new Q-value (or None if update failed).
        """
        if not self._accumulated_dq:
            logger.info("[MemQ] No accumulated ΔQ to flush")
            return {}

        if not getattr(self, "enable_value_driven", False):
            return {}

        logger.info(f"[MemQ] Flushing ΔQ updates for {len(self._accumulated_dq)} memories")

        results: Dict[str, Optional[float]] = {}
        do_normalize = getattr(self.rl_config, "normalize_dq", False)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_mem = {}
            for mem_id, dq in self._accumulated_dq.items():
                if do_normalize:
                    count = self._accumulated_dq_count.get(mem_id, 1)
                    dq = dq / count if count > 0 else dq
                future_to_mem[executor.submit(self._update_memory_item_memq, mem_id, dq)] = mem_id
            for future in as_completed(future_to_mem):
                mem_id = future_to_mem[future]
                try:
                    result = future.result()
                    if result:
                        updated_mem_id, new_q_value = result
                        results[updated_mem_id] = new_q_value
                except Exception as exc:
                    logger.error(f"[MemQ] {mem_id} update caused an exception: {exc}")

        logger.info(
            f"[MemQ] Flushed {len(results)} Q-value updates. Store size: {len(self._store)}"
        )

        self._accumulated_dq.clear()
        self._accumulated_dq_count.clear()
        return results


    def update_memory(
        self,
        task_description: str,
        trajectory: str,
        success: bool,
        retrieved_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Update memory based on new task experience using the configured update strategy.

        Args:
            task_description: Natural language description of the task
            trajectory: Detailed step-by-step trajectory of task execution
            success: Whether the task was completed successfully
            retrieved_memory_ids: IDs of memories that were retrieved for this task
            metadata: Optional additional metadata

        Returns:
            Memory ID if new memory was created or updated, None otherwise

        Raises:
            RuntimeError: If memory update fails
        """

        updater = get_updater(
            self.strategy_config.update,
            store=self._store,
            num_workers=self.num_workers,
            user_id=self.user_id,
            strategies=self.strategy_config,
            llm=self.llm_provider,
            embedder=self.embedding_provider,
            adjustment_mode=getattr(self, "adjustment_mode", "append"),
        )
        return updater.update(
            task_description=task_description,
            trajectory=trajectory,
            success=success,
            retrieved_memory_ids=retrieved_memory_ids,
            metadata=metadata,
        )

    def _prepare_memory_item(
        self,
        task_description: str,
        trajectory: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        [NEW HELPER METHOD]
        This is the "thinking" part of build_memory. It performs all CPU/IO-bound
        operations (LLM calls, embedding) to create a memory item dict, but does NOT
        write it to the store. This method is safe to run in parallel.

        Args:
            task_description: Natural language description of the task.
            trajectory: Detailed step-by-step trajectory of task execution.
            metadata: Optional additional metadata.

        Returns:
            A fully prepared memory item dict ready for insertion.
        """
        # This code is a direct copy of the logic inside `build_memory`, but stops before store.add

        # Extract source benchmark from metadata
        source_benchmark = (metadata or {}).get("source_benchmark", "unknown")
        raw_success = (metadata or {}).get("success", None)
        success = _coerce_success(raw_success)
        # Use Builder pattern to generate memory content
        builder = get_builder(self.strategy_config.build, self.llm_provider)
        memory_content = builder.build(task_description, trajectory)

        # Create procedural memory based on build strategy
        if self.strategy_config.build == BuildStrategy.TRAJECTORY:
            procedural_memory = ProceduralMemory.create_trajectory_memory(
                task_description=task_description,
                trajectory=memory_content,  # Use builder output
                build_strategy=self.strategy_config.build,
                retrieve_strategy=self.strategy_config.retrieve,
                update_strategy=self.strategy_config.update,
                source_benchmark=source_benchmark,
            )

        elif self.strategy_config.build == BuildStrategy.SCRIPT:
            procedural_memory = ProceduralMemory.create_script_memory(
                task_description=task_description,
                script=memory_content,  # Use builder output
                trajectory=trajectory,  # Keep original trajectory for reference
                build_strategy=self.strategy_config.build,
                retrieve_strategy=self.strategy_config.retrieve,
                update_strategy=self.strategy_config.update,
                source_benchmark=source_benchmark,
            )

        else:  # BuildStrategy.PROCEDURALIZATION
            # For proceduralization, the builder returns combined content
            # Extract script from the combined content for metadata
            lines = memory_content.split("\n")
            script_start = None
            trajectory_start = None

            for i, line in enumerate(lines):
                if line.strip() == "SCRIPT:":
                    script_start = i + 1
                elif line.strip() == "TRAJECTORY:":
                    trajectory_start = i + 1
                    break

            if script_start is not None and trajectory_start is not None:
                script = "\n".join(lines[script_start : trajectory_start - 2]).strip()
            else:
                # Fallback: generate script separately if parsing fails
                script = self.llm_provider.generate_script(trajectory)

            procedural_memory = ProceduralMemory.create_procedural_memory(
                task_description=task_description,
                script=script,
                trajectory=trajectory,
                build_strategy=self.strategy_config.build,
                retrieve_strategy=self.strategy_config.retrieve,
                update_strategy=self.strategy_config.update,
                source_benchmark=source_benchmark,
            )

        # Generate retrieval keys if needed
        if self.strategy_config.retrieve == RetrieveStrategy.AVEFACT:
            keywords = self.llm_provider.extract_keywords(
                task_description, self.max_keywords
            )
            avefact_vector = self.avefact_keyer.generate_key(task_description)
            procedural_memory.memp_metadata.avefact_keywords = keywords
            procedural_memory.memp_metadata.avefact_vector = avefact_vector
        elif self.strategy_config.retrieve == RetrieveStrategy.QUERY:
            query_vector = self.simple_keyer.generate_key(task_description)
            procedural_memory.memp_metadata.query_vector = query_vector

        # Write using _store with retrieval key = task_description (embedding),
        # and full content stored in metadata.full_content
        full_content = f"Task: {task_description}\n\n{procedural_memory.memory_content}"

        base_meta = {
            "type": "procedure",
            "source": "conversation",
            "source_benchmark": source_benchmark,
            "success": success,
            "strategy_build": self.strategy_config.build.value,
            "strategy_retrieve": self.strategy_config.retrieve.value,
            "strategy_update": self.strategy_config.update.value,
            "full_content": full_content,
        }
        if (
            getattr(self, "enable_value_driven", False)
            and getattr(self, "rl_config", None) is not None
        ):
            # User-selected behavior: unknown success defaults to q_init_pos.
            is_success = True if success is None else bool(success)
            base_meta |= {
                "q_value": (
                    float(self.rl_config.q_init_pos)
                    if is_success
                    else float(self.rl_config.q_init_neg)
                ),
                "q_visits": 0,
                "retrieval_count": 0,
                "q_updated_at": datetime.now().isoformat(),
                "last_used_at": datetime.now().isoformat(),
                "reward_ma": 0.0,
            }

        import uuid
        mem_id = str(uuid.uuid4())
        vector = self.embedding_provider.embed([task_description])[0]
        item = {"id": mem_id, "memory": task_description, "metadata": base_meta, "_vector": vector}
        return item

    def build_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> Dict[str, str]:
        """
        Builds memories in a batch from lists of data.
        Returns a dict mapping task_description -> memory_id.
        """
        logger.info(f"Starting parallel build for {len(task_descriptions)} memories...")

        items_to_add = []
        metadatas = metadatas or [{} for _ in task_descriptions]

        # --- Phase 1: Parallel "Thinking" ---
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Use zip to iterate through all lists in parallel
            tasks = [
                executor.submit(self._prepare_memory_item, td, traj, meta)
                for td, traj, meta in zip(task_descriptions, trajectories, metadatas)
            ]

            for future in tqdm(tasks, desc="Building memories (Parallel Processing)"):
                try:
                    items_to_add.append(future.result())
                except Exception as e:
                    logger.error(f"Failed to prepare a memory item in parallel: {e}")

        if not items_to_add:
            logger.warning(
                "No memory items were successfully prepared. Aborting build."
            )
            return {}

        # --- Phase 2: Serial "Writing" ---
        # Now, add all the prepared items to the store in a single, safe, serial operation.
        logger.info(
            f"Writing {len(items_to_add)} prepared memory items to the store..."
        )

        # The actual store write operation
        for item in tqdm(items_to_add):
            mem_id = item["id"]
            vector = item.pop("_vector")
            self._store.add(mem_id, vector, item)

        # Create the results map
        results: Dict[str, str] = {
            str(item["memory"]): str(item["id"]) for item in items_to_add
        }
        logger.info("Batch memory build complete.")
        return results

    def update_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_ids_list: List[List[str]],
        metadatas: Optional[List[Dict]] = None,
        images_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    ) -> List[Optional[str]]:
        """
        Updates memories in a batch by delegating to the parallel-capable
        updater's `update_batch` method.
        """
        logger.info("Delegating batch update to the configured updater...")

        # 1. Get the appropriate updater instance
        updater = get_updater(
            self.strategy_config.update,
            store=self._store,
            num_workers=self.num_workers,
            user_id=self.user_id,
            strategies=self.strategy_config,
            llm=self.llm_provider,
            embedder=self.embedding_provider,
            adjustment_mode=getattr(self, "adjustment_mode", "append"),
        )

        # 2. Call its parallel-capable batch method
        return updater.update_batch(
            task_descriptions=task_descriptions,
            trajectories=trajectories,
            successes=successes,
            retrieved_ids_list=retrieved_ids_list,
            metadatas=metadatas,
            images_list=images_list,
        )

    def add_memory(
        self,
        task_description: str,
        trajectory: str,
        success: bool,
        retrieved_memory_query: Optional[List[Tuple[str, float]]] = None,
        retrieved_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Add a single memory entry into the system.

        This is a thin wrapper around `add_memories()` to avoid maintaining two
        divergent implementations. It preserves the historical single-item API
        while reusing the batch-safe update/write path.
        """
        try:
            results = self.add_memories(
                task_descriptions=[task_description],
                trajectories=[trajectory],
                successes=[bool(success)],
                retrieved_memory_queries=[retrieved_memory_query],
                retrieved_memory_ids_list=[retrieved_memory_ids],
                metadatas=[metadata],
            )
            if not results:
                return None
            first = results[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                return first[1]
            return None
        except Exception as e:
            import traceback

            print(f"[add_memory] Error: {e}\n{traceback.format_exc()}")
            return None


    def set_epoch_info(self, current_epoch: int, total_epochs: int) -> None:
        """Called by runners at the start of each epoch."""
        self._current_epoch = current_epoch
        self._total_epochs = max(total_epochs, 1)

    def get_alpha(self) -> float:
        """Return scheduled alpha based on current epoch progress."""
        cfg = self.rl_config
        if cfg is None:
            return 0.1
        if cfg.alpha_schedule != "cosine":
            return cfg.alpha
        import math
        t = self._current_epoch
        T = self._total_epochs
        return cfg.alpha_min + 0.5 * (cfg.alpha - cfg.alpha_min) * (1.0 + math.cos(math.pi * t / T))

    def _normalize_q_score(self, q: float) -> float:
        """Max-scale Q value to [0, 1]."""
        if not self.normalize_q:
            return float(q)
        if self._q_max < 1e-9:
            self._update_q_max()
        q_max = self._q_max if self._q_max > 1e-9 else 1.0
        return float(q) / q_max

    def _update_q_max(self) -> None:
        """Recompute max Q-value from all memories in the store."""
        all_items = self._store.get_all()
        q_max = 0.0
        for item in all_items:
            meta = item.get("metadata")
            if isinstance(meta, dict) and "q_value" in meta:
                try:
                    q_max = max(q_max, float(meta["q_value"]))
                except (TypeError, ValueError):
                    pass
        self._q_max = q_max
        if q_max > 0:
            logger.info(f"[Q-Norm] Updated q_max={self._q_max:.4f}, n={len(all_items)}")

    def retrieve_query(
        self,
        task_description: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Unified retrieval using similarity-Q hybrid weighting.
        No two-stage retrieval; directly scores candidates by sim and Q mixture.

        Returns:
            {
                "actions": [...],
                "selected": [...],
                "candidates": [...],
                "simmax": float
            }
        """

        try:
            # -------- Basic checks --------
            if not hasattr(self, "dict_memory") or not self.dict_memory:
                return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

            if not getattr(self, "embedding_provider", None):
                raise RuntimeError("embedding_provider is required for local retrieval")

            embed = getattr(self.embedding_provider, "embed", None)
            if not callable(embed):
                raise RuntimeError("embedding_provider.embed() not callable")

            # -------- Compute query embedding --------
            queries = list(self.dict_memory.keys())
            if not queries:
                return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

            query_vec = get_embedding_with_retry(embed, [task_description])[0]
            query_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-8

            query_embeddings = getattr(self, "query_embeddings", {})
            missing_queries = [q for q in queries if q not in query_embeddings]

            if missing_queries:
                logger.info(
                    f"Missing embeddings for {len(missing_queries)} queries, fetching..."
                )
                new_vecs = get_embedding_with_retry(embed, missing_queries)
                for q, v in zip(missing_queries, new_vecs):
                    query_embeddings[q] = v
                self.query_embeddings.update(query_embeddings)

            # -------- Compute similarity for all queries --------
            sim_list = []
            for q in queries:
                qv = query_embeddings[q]
                q_norm = math.sqrt(sum(x * x for x in qv)) or 1e-8
                sim = sum(a * b for a, b in zip(query_vec, qv)) / (query_norm * q_norm)
                if sim >= threshold:
                    sim_list.append((q, sim))
            sim_list.sort(key=lambda x: x[1], reverse=True)

            if k is not None:
                sim_list = sim_list[:k]
            if not sim_list:
                return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

            # -------- Fetch memory objects, build candidate list --------
            candidates = []
            seen_candidate_ids: set[str] = set()
            for q, sim in sim_list:
                mem_ids = self.dict_memory.get(q, [])
                for mid in mem_ids:
                    if mid in seen_candidate_ids:
                        continue
                    seen_candidate_ids.add(mid)
                    try:
                        mem_obj = self._store.get(mid)

                        if mem_obj is not None:
                            md = mem_obj.get("metadata", {})
                            content = None
                            try:
                                if isinstance(md, dict):
                                    content = md.get("full_content")
                            except Exception:
                                content = None

                            candidates.append(
                                {
                                    "memory_id": mid,
                                    "content": content,
                                    "similarity": float(sim),
                                    "metadata": md,
                                    "memory_item": mem_obj,
                                }
                            )
                    except Exception:
                        logger.info(f"Failed to load memory {mid}", exc_info=True)

            if not candidates:
                return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

            # -------- Compute Q value for each candidate --------
            enriched = []
            simmax = 0.0
            q_values = []
            for c in candidates:
                sim = c["similarity"]
                simmax = max(simmax, sim)

                # Read Q-value from store payload metadata
                mem_id = c.get("memory_id")
                md = _meta_to_dict(c.get("metadata"))
                # Stable "task" identifier for de-dup (LLB-only usage).
                # NOTE: task_id can legally be 0, so avoid truthiness-based fallbacks.
                task_id = extract_task_id(md)

                has_meta_q = isinstance(md, dict) and ("q_value" in md)
                q = md.get("q_value", self.rl_config.q_init_pos)
                try:
                    q = float(q)
                except Exception:
                    q = self.rl_config.q_init_pos

                # Optional Q floor (LLB may set this via experiment.llb_q_floor).
                q_floor = getattr(self.rl_config, "q_floor", None)
                if q_floor is not None:
                    try:
                        q = max(float(q_floor), float(q))
                    except Exception:
                        pass

                c_local = dict(c)
                c_local["q_estimate"] = q
                c_local["task_id"] = (str(task_id) if task_id is not None else None)
                q_values.append(q)
                enriched.append(c_local)

            # -------- Optional Q threshold --------
            q_min = getattr(self.rl_config, "q_min_threshold", None)
            if q_min is not None:
                enriched = [c for c in enriched if c["q_estimate"] >= q_min]

            if not enriched:
                return {
                    "actions": [],
                    "selected": [],
                    "candidates": [],
                    "simmax": simmax,
                }

            # -------- Hybrid scoring (similarity + Q) --------
            w_sim = self.weight_sim
            w_q = self.weight_q

            similarities, q_zs, scores = [], [], []
            for c in enriched:
                sim = c["similarity"]
                q = c["q_estimate"]

                q_z = self._normalize_q_score(q)

                c["q_z"] = q_z
                c["score"] = sim * w_sim + q_z * w_q

                similarities.append(sim)
                q_zs.append(q_z)
                scores.append(c["score"])

                logger.debug(
                    f"[Scoring] cosine={sim:.4f} | q={q:.4f} q_norm={q_z:.4f} | "
                    f"score={c['score']:.4f} (w_sim={w_sim}, w_q={w_q}) | "
                    f"q_max={self._q_max:.4f}"
                )

            if similarities:
                logger.info(
                    f"[Scoring Summary] candidates={len(enriched)}, "
                    f"cosine: [{min(similarities):.2f}, {max(similarities):.2f}], "
                    f"q_norm: [{min(q_zs):.2f}, {max(q_zs):.2f}], "
                    f"score: [{min(scores):.2f}, {max(scores):.2f}]"
                )

            # -------- Sort by hybrid score --------
            enriched_sorted = sorted(enriched, key=lambda x: x["score"], reverse=True)

            # -------- epsilon-greedy sampling --------
            topk = min(self.rl_config.topk, len(enriched_sorted))
            if not getattr(self, "dedup_by_task_id", False):
                if random.random() < self.rl_config.epsilon:
                    selected = random.sample(enriched_sorted, topk)
                else:
                    selected = enriched_sorted[:topk]
            else:
                # LLB-only: de-dup by task_id while keeping epsilon-greedy behavior.
                # - greedy: iterate score-desc
                # - epsilon: shuffle before taking unique tasks
                pool = list(enriched_sorted)
                if random.random() < self.rl_config.epsilon:
                    random.shuffle(pool)

                selected = []
                seen_tasks: set[str] = set()
                for cand in pool:
                    tid = cand.get("task_id")
                    # If task_id missing, treat as unique by memory_id to avoid collapsing unrelated entries.
                    key = str(tid) if tid else f"__missing_task_id__:{cand.get('memory_id')}"
                    if key in seen_tasks:
                        continue
                    seen_tasks.add(key)
                    selected.append(cand)
                    if len(selected) >= topk:
                        break

            # Increment retrieval count for selected memories
            for s in selected:
                mid = s.get("memory_id")
                if mid:
                    item = self._store.get(mid)
                    if item is not None:
                        md = item.get("metadata", {})
                        if isinstance(md, dict):
                            md["retrieval_count"] = md.get("retrieval_count", 0) + 1
                            md["last_used_at"] = _now_iso()

            return {
                "actions": [s["memory_id"] for s in selected],
                "selected": selected,
                "candidates": enriched_sorted,
                "simmax": simmax,
            }, sim_list

        except Exception as e:
            logger.info(
                f"Local retrieve failed, task_desc: {task_description}", exc_info=True
            )
            raise RuntimeError(f"Local retrieve failed: {e}")

    def add_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_memory_queries: Optional[List[List[Tuple[str, float]]]] = None,
        retrieved_memory_ids_list: Optional[List[Optional[List[str]]]] = None,
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
        images_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Batch version of `add_memory`.

        For each (task_description, trajectory, success), determine if a similar query already exists.
        If yes, attach the trajectory to that query; otherwise, create a new query entry.
        Stores query embeddings for new queries to avoid recomputation during retrieval.

        Args:
            task_descriptions: List of task descriptions.
            trajectories: List of trajectories corresponding to each task.
            successes: List of success flags.
            retrieved_memory_queries: Optional list of lists, each containing (query, score) pairs.
            retrieved_memory_ids_list: Optional list of retrieved memory ID lists for each task.
            metadatas: Optional list of metadata dicts.

        Returns:
            Dict mapping task_description → memory_id (or None if failed).
        """
        try:
            if not hasattr(self, "dict_memory"):
                self.dict_memory = {}
            if not hasattr(self, "query_embeddings"):
                self.query_embeddings = {}

            similarity_threshold = getattr(self, "add_similarity_threshold", 0.8)
            n = len(task_descriptions)

            if retrieved_memory_queries is None:
                retrieved_memory_queries = [None] * n
            if retrieved_memory_ids_list is None:
                retrieved_memory_ids_list = [None] * n
            if metadatas is None:
                metadatas = [None] * n
            if images_list is None:
                images_list = [None] * n
            td_list = []
            traj_list = []
            succ_list = []
            retrieved_ids_payload = []
            metadata_list = []
            imgs_list = []

            for i in range(n):
                td = task_descriptions[i]
                traj = trajectories[i]
                succ = successes[i]
                retrieved_ids = retrieved_memory_ids_list[i]
                meta = dict(metadatas[i] or {})

                # 记录本次检索到的记忆，以便成功样本也能追溯引用链路
                if retrieved_ids:
                    meta["related_memory_ids"] = [
                        str(rid) for rid in retrieved_ids if rid
                    ]

                # Attach RL-style meta defaults.
                #
                # IMPORTANT: do NOT blindly override upstream metadata (runner may
                # provide q_value based on success/failure). Only fill missing
                # fields, and when q_value is missing/invalid, initialize from
                # q_init_pos/q_init_neg according to the success flag.
                meta.setdefault("success", bool(succ))

                rl_cfg = getattr(self, "rl_config", None)
                try:
                    q_init_pos = float(getattr(rl_cfg, "q_init_pos", 0.0))
                except Exception:
                    q_init_pos = 0.0
                try:
                    q_init_neg = float(getattr(rl_cfg, "q_init_neg", 0.0))
                except Exception:
                    q_init_neg = 0.0

                default_q = q_init_pos if bool(succ) else q_init_neg

                # Adaptive Q init: use mean Q of retrieved neighbors (overrides upstream default)
                use_adaptive = getattr(rl_cfg, "q_init_adaptive", False)
                adaptive_q = None
                if use_adaptive and retrieved_ids:
                    neighbor_qs = []
                    for rid in retrieved_ids:
                        item = self._store.get(rid)
                        if item is not None:
                            meta_d = _meta_to_dict(item.get("metadata"))
                            try:
                                neighbor_qs.append(float(meta_d["q_value"]))
                            except (KeyError, TypeError, ValueError):
                                pass
                    if neighbor_qs:
                        adaptive_q = sum(neighbor_qs) / len(neighbor_qs)

                if adaptive_q is not None:
                    # Adaptive overrides any upstream q_value
                    meta["q_value"] = adaptive_q
                elif "q_value" not in meta or meta.get("q_value") is None:
                    meta["q_value"] = default_q
                else:
                    # If provided but not castable, fall back to the default.
                    try:
                        meta["q_value"] = float(meta["q_value"])
                    except Exception:
                        meta["q_value"] = default_q

                meta.setdefault("q_visits", 0)
                meta.setdefault("retrieval_count", 0)
                meta.setdefault("q_updated_at", datetime.now().isoformat())
                meta.setdefault("last_used_at", datetime.now().isoformat())
                meta.setdefault("reward_ma", 0.0)

                # append to parallel lists
                td_list.append(td[:4096])  # Truncate to avoid excessive length)
                traj_list.append(traj)
                succ_list.append(succ)
                retrieved_ids_payload.append(retrieved_ids)
                metadata_list.append(meta)
                imgs_list.append(images_list[i])

            results = self.update_memories(
                task_descriptions=td_list,
                trajectories=traj_list,
                successes=succ_list,
                retrieved_ids_list=retrieved_ids_payload,
                metadatas=metadata_list,
                images_list=imgs_list,
            )

            for i, task_description in enumerate(task_descriptions):
                if i < len(results):
                    recorded_task, mem_id = results[i]
                    if recorded_task != task_description:
                        logger.warning(
                            f"Task description mismatch at index {i}: expected '{task_description}', got '{recorded_task}'"
                        )
                    mem_id = mem_id
                else:
                    logger.warning(f"No result found for task {task_description}")
                    mem_id = None

                retrieved_qs = retrieved_memory_queries[i]
                matched_query = None
                best_score = -1.0
                if retrieved_qs:
                    for query, score in retrieved_qs:
                        if score >= similarity_threshold and score > best_score:
                            matched_query, best_score = query, score

                if matched_query:
                    if matched_query not in self.dict_memory:
                        self.dict_memory[matched_query] = []
                    self.dict_memory[matched_query].append(mem_id)
                else:
                    self.dict_memory[task_description] = [mem_id]

                    if getattr(self, "embedding_provider", None):
                        embed = getattr(self.embedding_provider, "embed", None)
                        if callable(embed):
                            try:
                                vec = embed([task_description])[0]
                                self.query_embeddings[task_description] = vec
                                logger.info(
                                    f"Cached embedding for new query: {task_description[:40]}..."
                                )
                            except Exception:
                                logger.info(
                                    f"Failed to embed query '{task_description}'",
                                    exc_info=True,
                                )

            return results

        except Exception as e:
            import traceback

            print(f"[add_memories] Error: {e}\n{traceback.format_exc()}")
            return {}

    def save_checkpoint_snapshot(self, target_ck_dir: str, ckpt_id: str) -> dict:
        """Save a snapshot of the current MemoryStore to the checkpoint directory.

        Directory structure:
            <ck_dir>/snapshot/<ckpt_id>/
                - vectors.npy + payloads.json   (MemoryStore native format)
                - snapshot_meta.json
                - local_cache/                   (dict_memory, query_embeddings)

        Returns:
            Metadata dict with checkpoint info.
        """
        import json
        import os

        os.makedirs(target_ck_dir, exist_ok=True)
        snapshot_root = os.path.join(target_ck_dir, f"snapshot/{ckpt_id}")
        os.makedirs(snapshot_root, exist_ok=True)

        # Save memory store
        self._store.save(snapshot_root)

        # Count and metadata
        total_count = len(self._store)
        meta = {
            "user_id": self.user_id,
            "checkpoint_id": ckpt_id,
            "visible_memories": total_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(snapshot_root, "snapshot_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        try:
            self._persist_local_caches(snapshot_root)
        except Exception:
            logger.warning("Failed to persist local caches during snapshot save", exc_info=True)
        return meta

    def load_checkpoint_snapshot(
        self, snapshot_root: str, *, mem_cube_id: str | None = None
    ) -> int:
        """Load a snapshot from the checkpoint directory into the current MemoryStore.

        - If snapshot_root doesn't contain a specific epoch number, auto-finds the latest.
        - Tries MemoryStore.load() (handles both native and legacy formats).
        - Rebuilds local memory index from loaded data.

        Returns:
            int: The checkpoint_id (epoch number) from the loaded snapshot.
        """
        import json
        import os

        # If snapshot_root is not a specific epoch directory, auto-find the latest
        if os.path.isdir(snapshot_root) and not os.path.isfile(
            os.path.join(snapshot_root, "snapshot_meta.json")
        ):
            try:
                epoch_dirs = []
                for item in os.listdir(snapshot_root):
                    item_path = os.path.join(snapshot_root, item)
                    if os.path.isdir(item_path) and item.isdigit():
                        epoch_dirs.append(int(item))

                if epoch_dirs:
                    max_epoch = max(epoch_dirs)
                    snapshot_root = os.path.join(snapshot_root, str(max_epoch))
                    logger.info(
                        f"Auto-selected latest checkpoint: epoch {max_epoch} from {snapshot_root}"
                    )
            except Exception as e:
                logger.warning(f"Failed to auto-detect epoch directory: {e}")

        # Read snapshot metadata for checkpoint_id
        meta_path = os.path.join(snapshot_root, "snapshot_meta.json")
        checkpoint_id = 0
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                checkpoint_id = int(meta.get("checkpoint_id", 0))
            except Exception:
                pass

        # Load store from snapshot
        try:
            self._store = MemoryStore.load(snapshot_root)
        except FileNotFoundError:
            raise ValueError(f"No loadable snapshot data found in {snapshot_root}")

        # Try restoring local caches from snapshot
        cache_dir = os.path.join(snapshot_root, "local_cache")
        restored_cache = False
        if os.path.isdir(cache_dir):
            try:
                restored_cache = self._restore_local_caches(cache_dir)
            except Exception:
                logger.warning(
                    "Failed to restore local caches from %s", cache_dir, exc_info=True
                )

        if not restored_cache or not getattr(self, "dict_memory", None):
            # Rebuild from store data
            all_items = self._store.get_all()
            text_epochs = [{"memories": all_items}]
            logger.info("Rebuilding local memory")
            rebuilt = self._rebuild_local_memory_index(text_epochs)
            logger.info("Local memory index rebuilt with %s entries", rebuilt)
        else:
            logger.info(
                "Local caches restored from snapshot cache directory %s", cache_dir
            )

        logger.info(f"Checkpoint loaded from epoch {checkpoint_id}")
        return checkpoint_id

    def _rebuild_local_memory_index(
        self,
        text_epochs: List[Dict[str, Any]],
        *,
        cutoff_before: datetime | None = None,
    ) -> int:
        """Reconstruct dict_memory from store data."""
        if not hasattr(self, "dict_memory") or not isinstance(self.dict_memory, dict):
            self.dict_memory = {}
        else:
            self.dict_memory.clear()

        if not hasattr(self, "query_embeddings") or not isinstance(
            self.query_embeddings, dict
        ):
            self.query_embeddings = {}
        else:
            self.query_embeddings.clear()

        added = 0
        for sec in text_epochs or []:
            mems = sec.get("memories", [])
            if not isinstance(mems, list):
                continue
            for mem in mems:
                # Support both plain dicts and objects with attributes
                if isinstance(mem, dict):
                    mem_id = mem.get("id")
                    query = mem.get("memory")
                    md = mem.get("metadata")
                else:
                    mem_id = getattr(mem, "id", None)
                    query = getattr(mem, "memory", None)
                    md = getattr(mem, "metadata", None)
                if not mem_id or not query:
                    continue

                md_dict = _meta_to_dict(md)

                mem_time_value = md_dict.get("memory_time")
                mem_dt = _parse_datetime(mem_time_value)
                if cutoff_before:
                    if mem_dt is None:
                        # If we cannot determine the timestamp, skip to keep filtering explicit.
                        continue
                    if mem_dt >= cutoff_before:
                        continue

                bucket = self.dict_memory.setdefault(query, [])
                bucket.append(mem_id)
                added += 1

        query_embeddings = getattr(self, "query_embeddings", {})
        queries = list(self.dict_memory.keys())
        missing_queries = [q for q in queries if q not in query_embeddings]
        embed = getattr(self.embedding_provider, "embed", None)
        if missing_queries and callable(embed):
            batch_size = getattr(self, "embedding_batch_size", 256)
            for start in tqdm(
                range(0, len(missing_queries), batch_size),
                desc="Rebuilding query embeddings",
            ):
                batch_queries = missing_queries[start : start + batch_size]
                batch_vecs = get_embedding_with_retry(embed, batch_queries)
                for q, v in zip(batch_queries, batch_vecs):
                    query_embeddings[q] = v
            self.query_embeddings.update(query_embeddings)

        return added

    def _persist_local_caches(self, snapshot_root: str) -> None:
        """Persist local cache dictionaries alongside the store snapshot."""
        import json
        import os

        cache_dir = os.path.join(snapshot_root, "local_cache")
        os.makedirs(cache_dir, exist_ok=True)

        def _write_json(path: str, payload: dict) -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

        dict_mem = getattr(self, "dict_memory", {}) or {}
        dict_payload: Dict[str, List[str]] = {}
        if isinstance(dict_mem, dict):
            for query, mem_ids in dict_mem.items():
                if not isinstance(mem_ids, list):
                    continue
                cleaned_ids = [str(mid) for mid in mem_ids if mid]
                if cleaned_ids:
                    dict_payload[str(query)] = cleaned_ids
        _write_json(os.path.join(cache_dir, "dict_memory.json"), dict_payload)

        query_embeddings = getattr(self, "query_embeddings", {}) or {}
        emb_payload: Dict[str, List[float]] = {}
        if isinstance(query_embeddings, dict):
            for query, vec in query_embeddings.items():
                if vec is None:
                    continue
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                try:
                    emb_payload[str(query)] = [float(x) for x in vec]
                except Exception:
                    continue
        _write_json(os.path.join(cache_dir, "query_embeddings.json"), emb_payload)

        logger.info(
            "Persisted local caches to %s (dict=%d, embeddings=%d)",
            cache_dir,
            len(dict_payload),
            len(emb_payload),
        )

    def _restore_local_caches(self, cache_dir: str) -> bool:
        """Restore dict_memory/query_embeddings if cache files exist."""
        import json
        import os

        restored_any = False
        dict_path = os.path.join(cache_dir, "dict_memory.json")
        if os.path.isfile(dict_path):
            try:
                with open(dict_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.dict_memory = {
                        str(k): [str(x) for x in (v or []) if x] for k, v in raw.items()
                    }
                    restored_any = True
            except Exception:
                logger.warning(
                    "Failed to restore dict_memory cache from %s",
                    dict_path,
                    exc_info=True,
                )

        emb_path = os.path.join(cache_dir, "query_embeddings.json")
        if os.path.isfile(emb_path):
            try:
                with open(emb_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.query_embeddings = {}
                    for k, vec in raw.items():
                        if isinstance(vec, list):
                            try:
                                self.query_embeddings[str(k)] = [float(x) for x in vec]
                            except Exception:
                                continue
                    restored_any = True
            except Exception:
                logger.warning(
                    "Failed to restore query embeddings cache from %s",
                    emb_path,
                    exc_info=True,
                )

        return restored_any
