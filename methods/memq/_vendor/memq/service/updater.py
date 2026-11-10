"""
Updaters for different update strategies in the Memp system.

This module mirrors builders.py and retrievers.py patterns and provides:
- BaseUpdater (abstract)
- VanillaUpdater / ValidationUpdater / AdjustmentUpdater (concrete)
- get_updater factory

Key goals:
- Use MemoryStore for add/update to ensure real memory_id is available
- Support Adjustment in append and inplace modes
- Unify metadata fields (task_description, strategies, updated_at, etc.)
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .builders import get_builder
from .memory_store import MemoryStore
from .strategies import StrategyConfiguration, UpdateStrategy

logger = logging.getLogger(__name__)


# ------------ Helper structures ------------

@dataclass
class AdjustmentConfig:
    mode: str = "append"  # "append" | "inplace"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _build_standard_metadata(
    *,
    base: Optional[Dict[str, Any]],
    task_description: str,
    strategies: StrategyConfiguration,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose metadata dict with unified fields."""
    meta: Dict[str, Any] = dict(base or {})
    meta.setdefault("updated_at", _now_iso())
    meta.setdefault("memory_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meta["task_description"] = task_description
    meta["strategy_build"] = strategies.build.value
    meta["strategy_retrieve"] = strategies.retrieve.value
    meta["strategy_update"] = strategies.update.value
    if extra:
        meta.update(extra)
    return meta


def _store_add_with_retry(
    store: MemoryStore,
    embedder: Any,
    mem_id: str,
    task_description: str,
    payload: Dict[str, Any],
    max_retries: int = 5,
    retry_delays: tuple = (10, 60, 180, 300, 600),
) -> None:
    """Add to store with retry for embedding API failures."""
    for attempt in range(1, max_retries + 1):
        try:
            vector = embedder.embed([task_description])[0]
            store.add(mem_id, vector, payload)
            return
        except Exception as e:
            logger.warning(f"[Retry {attempt}/{max_retries}] Store add failed: {e}")
            if attempt == max_retries:
                logger.error("Max retries reached, add failed.")
                raise
            idx = min(attempt - 1, len(retry_delays) - 1)
            time.sleep(retry_delays[idx])


# ------------ Base class ------------

class BaseUpdater(ABC):
    def __init__(
        self,
        store: MemoryStore,
        num_workers: int,
        user_id: str,
        strategies: StrategyConfiguration,
        llm: Any,
        embedder: Any,
        *,
        adjustment_config: Optional[AdjustmentConfig] = None,
    ) -> None:
        self.store = store
        self.num_workers = num_workers
        self.user_id = user_id
        self.strategies = strategies
        self.llm = llm
        self.embedder = embedder
        self.adjustment_config = adjustment_config or AdjustmentConfig()

    @abstractmethod
    def prepare_update_op(
        self,
        task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        ...

    def execute_update_op(self, op: Dict) -> Optional[str]:
        """Execute a prepared operation dictionary."""
        if not op or op.get("op") == "noop":
            return None

        op_type = op["op"]
        if op_type == "add":
            mem_id = op["mem_id"]
            task_desc = op["task_description"]
            payload = op["payload"]
            _store_add_with_retry(self.store, self.embedder, mem_id, task_desc, payload)
            return mem_id
        elif op_type == "update":
            mem_id = op["id"]
            data = op["data"]
            self.store.update(mem_id, data)
            return str(mem_id)

        logger.warning(f"Unknown operation type '{op_type}' in execute_update_op.")
        return None

    def update(
        self,
        task_description: str,
        trajectory: str,
        success: bool,
        retrieved_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """A single, synchronous update operation for convenience and testing."""
        op = self.prepare_update_op(task_description, trajectory, success, retrieved_memory_ids, metadata, images=images)
        return self.execute_update_op(op)

    def update_batch(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_ids_list: List[List[str]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        images_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    ) -> List[Optional[str]]:
        """Hybrid parallel/serial batch update."""
        num_tasks = len(task_descriptions)
        logger.info(f"Starting hybrid parallel update for {num_tasks} memories...")

        metadatas = metadatas or [None] * num_tasks
        images_list = images_list or [None] * num_tasks
        ops_to_execute = [None] * num_tasks

        # Phase 1: Parallel "Thinking" (prepare_update_op)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_index = {
                executor.submit(self.prepare_update_op, td, traj, success, r_ids, meta, imgs): i
                for i, (td, traj, success, r_ids, meta, imgs) in enumerate(zip(
                    task_descriptions, trajectories, successes, retrieved_ids_list, metadatas, images_list
                ))
            }
            for future in tqdm(as_completed(future_to_index), total=num_tasks, desc="Updating memories (Parallel Processing)"):
                index = future_to_index[future]
                try:
                    op = future.result()
                    ops_to_execute[index] = op
                except Exception as e:
                    logger.error(f"Failed to prepare update for task '{task_descriptions[index]}': {e}", exc_info=True)

        # Phase 2: Serial "Writing" (execute_update_op)
        logger.info(f"Executing {len(ops_to_execute)} prepared memory operations serially...")
        results = []
        for op in tqdm(ops_to_execute, desc="Updating memories (Serial Writing)"):
            if op:
                task_desc = op.get("task_description", "unknown_task")
                try:
                    mem_id = self.execute_update_op(op)
                    results.append((task_desc, mem_id))
                except Exception as e:
                    logger.error(f"Failed to execute update for task '{task_desc}': {e}", exc_info=True)
                    results.append((task_desc, None))

        return results

    def _generate_reflection(self, task_description: str, failed_trajectory: str, images: Optional[List[Dict[str, Any]]] = None) -> str:
        prompt = f"""
Task: {task_description}

Failed trajectory:
{failed_trajectory}

This task failed. Analyze what went wrong and suggest improvements for future similar tasks.
Focus on:
1. Incorrect assumptions
2. Steps to improve
3. What to avoid next time

Provide a brief reflection:
"""
        if images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img["url"]}})
            messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]
        try:
            return self.llm.generate(messages, temperature=0.3)
        except Exception as e:
            logger.warning(f"LLM reflection generation failed: {e}")
            return "Reflection not available due to LLM error."


# ------------ Concrete updaters ------------

class VanillaUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        builder = get_builder(self.strategies.build, self.llm)
        memory_body = builder.build(task_description, trajectory, images=images)
        memory_content = f"Task: {task_description}\n\n{memory_body}"
        mem_id = str(uuid.uuid4())
        meta = _build_standard_metadata(
            base=metadata, task_description=task_description, strategies=self.strategies,
            extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
        )
        payload = {"id": mem_id, "memory": task_description, "metadata": meta}
        return {"op": "add", "mem_id": mem_id, "payload": payload, "task_description": task_description}


class ValidationUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        if not success:
            return {"op": "noop", "task_description": task_description}
        builder = get_builder(self.strategies.build, self.llm)
        memory_body = builder.build(task_description, trajectory, images=images)
        memory_content = f"Task: {task_description}\n\n{memory_body}"
        mem_id = str(uuid.uuid4())
        meta = _build_standard_metadata(
            base=metadata, task_description=task_description, strategies=self.strategies,
            extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
        )
        payload = {"id": mem_id, "memory": task_description, "metadata": meta}
        return {"op": "add", "mem_id": mem_id, "payload": payload, "task_description": task_description}


class AdjustmentUpdater(BaseUpdater):
    def prepare_update_op(
        self, task_description: str, trajectory: str, success: bool,
        retrieved_memory_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        if success:
            builder = get_builder(self.strategies.build, self.llm)
            memory_body = builder.build(task_description, trajectory, images=images)
            memory_content = f"Task: {task_description}\n\n{memory_body}"
            mem_id = str(uuid.uuid4())
            meta = _build_standard_metadata(
                base=metadata, task_description=task_description, strategies=self.strategies,
                extra={"type": "procedure", "source": "conversation", "full_content": memory_content},
            )
            payload = {"id": mem_id, "memory": task_description, "metadata": meta}
            return {"op": "add", "mem_id": mem_id, "payload": payload, "task_description": task_description}

        # Failure path
        reflection = self._generate_reflection(task_description, trajectory, images=images)
        mode = (self.adjustment_config.mode or "append").lower()

        if mode == "inplace":
            return self._prepare_inplace_adjust(task_description, trajectory, reflection, retrieved_memory_ids or [], metadata)
        return self._prepare_append_adjust(task_description, trajectory, reflection, retrieved_memory_ids or [], metadata)

    def _prepare_append_adjust(self, task_description: str, failed_trajectory: str, reflection: str,
                               related_ids: List[str], metadata: Optional[Dict[str, Any]]) -> Dict:
        adjustment_content = (
            f"TASK REFLECTION:\nTask: {task_description}\n\n"
            f"What went wrong:\n{reflection}\n\nFailed approach:\n{failed_trajectory}\n"
        )
        mem_id = str(uuid.uuid4())
        meta = _build_standard_metadata(
            base=metadata, task_description=task_description, strategies=self.strategies,
            extra={
                "type": "adjustment", "source": "conversation", "source_detail": "reflection",
                "related_memory_ids": related_ids, "full_content": adjustment_content,
            },
        )
        payload = {"id": mem_id, "memory": task_description, "metadata": meta}
        return {"op": "add", "mem_id": mem_id, "payload": payload, "task_description": task_description}

    def _prepare_inplace_adjust(self, task_description: str, failed_trajectory: str, reflection: str,
                                related_ids: List[str], metadata: Optional[Dict[str, Any]]) -> Dict:
        if not related_ids:
            return {"op": "noop", "task_description": task_description}

        mem_id_to_update = related_ids[0]
        old_item = self.store.get(mem_id_to_update)
        if old_item is None:
            logger.warning(f"Could not find memory {mem_id_to_update} to update. Skipping.")
            return {"op": "noop", "task_description": task_description}

        old_meta = old_item.get("metadata", {})
        if not isinstance(old_meta, dict):
            old_meta = {}
        prev_full = old_meta.get("full_content", f"Task: {old_item.get('memory', '')}\n\n(Original content unavailable)")

        new_full_content = (
            f"{prev_full}\n\n--- ADJUSTMENT NOTE ({_now_iso()}) ---\n"
            f"A similar task failed: {task_description}\n\n"
            f"Reflection:\n{reflection}\n"
        )

        new_meta = _build_standard_metadata(
            base={**old_meta, **(metadata or {})},
            task_description=old_item.get("memory", ""),
            strategies=self.strategies,
            extra={"full_content": new_full_content},
        )

        return {
            "op": "update",
            "id": mem_id_to_update,
            "data": {"id": mem_id_to_update, "memory": old_item.get("memory", ""), "metadata": new_meta},
            "task_description": task_description,
        }


# ------------ Factory ------------

def get_updater(
    strategy: UpdateStrategy,
    *,
    store: MemoryStore,
    user_id: str,
    strategies: StrategyConfiguration,
    llm: Any,
    embedder: Any,
    num_workers: int = 32,
    adjustment_mode: str = "append",
) -> BaseUpdater:
    cfg = AdjustmentConfig(mode=adjustment_mode)
    kwargs = dict(adjustment_config=cfg)
    if strategy == UpdateStrategy.VANILLA:
        return VanillaUpdater(store, num_workers, user_id, strategies, llm, embedder, **kwargs)
    if strategy == UpdateStrategy.VALIDATION:
        return ValidationUpdater(store, num_workers, user_id, strategies, llm, embedder, **kwargs)
    return AdjustmentUpdater(store, num_workers, user_id, strategies, llm, embedder, **kwargs)
