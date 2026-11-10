from __future__ import annotations

from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.models import (
    ContentBlock,
    DecisionContext,
    MemoryContribution,
    MemoryEvent,
    MemoryItem,
    MemoryScope,
)
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin, MethodContext, MethodServices, call_maybe_async
from mematlas.methods.common import event_images
from mematlas.methods.m2a.core import M2ACore
from mematlas.methods.manifest import load_method_descriptor
from mematlas.methods.services import prepare_services


DESCRIPTOR = load_method_descriptor(__file__)


class M2AMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params = dict(config.params)
        self.services = services
        self.wrapper = services.object("m2a.instance")
        self._conversation_index = 0
        self._started = False
        self._active_scope: str | None = None

    def _ensure_scope(self, scope: MemoryScope) -> None:
        key = scope.key()
        if self._active_scope is None:
            self._active_scope = key
        elif key != self._active_scope:
            raise ValueError("M2A storage is conversation-scoped; call reset before changing scope")

    async def setup(self, context: MethodContext) -> None:
        if self.wrapper is None:
            prepare_services(self.params, self.services, method_id="m2a", llm=True, embedding=True)
        if self.wrapper is None:
            self.wrapper = M2ACore(
                self.services,
                db_dir=self.params.get("db_dir"),
                storage_path=self.params.get("storage_path")
                or self.params.get("data_storage_path"),
                **{key: value for key, value in self.params.items()
                   if key not in {"db_dir", "storage_path", "data_storage_path"}},
            )
        await self._start_conversation()

    async def _start_conversation(self) -> None:
        await call_maybe_async(
            self.wrapper.start_conversation,
            {
                "conv_idx": self._conversation_index,
                "speaker_0": self.params.get("speaker_0", "user"),
                "speaker_1": self.params.get("speaker_1", "assistant"),
            },
            run_sync_in_thread=True,
        )
        self._started = True

    def _speaker_index(self, role: str | None) -> int:
        role = (role or "user").lower()
        speaker_1 = str(self.params.get("speaker_1", "assistant")).lower()
        return 1 if role in {"assistant", "agent", speaker_1} else 0

    async def observe(self, event: MemoryEvent) -> None:
        self._ensure_scope(event.scope)
        images = event_images(event.content)
        if not event.text and not images:
            return
        await call_maybe_async(
            self.wrapper.chat,
            [{
                "speaker": self._speaker_index(event.role),
                "dia_id": event.event_id,
                "images": images,
                "text": event.text,
                "timestamp": event.timestamp.isoformat(),
            }],
            run_sync_in_thread=True,
        )

    async def retrieve(self, context: DecisionContext) -> MemoryContribution:
        if not context.metadata.get("read_only") or self._active_scope is not None:
            self._ensure_scope(context.scope)
        images = event_images(context.query)
        if hasattr(self.wrapper, "query"):
            raw = await call_maybe_async(
                self.wrapper.query,
                text=context.text,
                images=images,
                top_k=context.top_k,
            )
            if isinstance(raw, Mapping):
                items = []
                for memory in raw.get("memories", []):
                    if not isinstance(memory, Mapping):
                        continue
                    blocks = [ContentBlock.text(str(memory.get("text", "")))]
                    blocks.extend(
                        ContentBlock.image(str(path))
                        for path in memory.get("image_paths", [])
                        if path
                    )
                    items.append(MemoryItem(
                        content=tuple(blocks),
                        memory_id=str(memory.get("id", "")) or None,
                        score=memory.get("score"),
                        source_event_ids=tuple(
                            f"raw:{start}-{end}"
                            for start, end in memory.get("evidence_ids", [])
                        ),
                        memory_type="multimodal_semantic",
                        metadata={"method_id": "m2a", **dict(memory)},
                    ))
                answer = str(raw.get("answer", ""))
                return MemoryContribution(
                    items=tuple(items),
                    context=(ContentBlock.text(answer, kind="multimodal_memory",
                                               source_item_indices=list(range(len(items)))),) if answer else (),
                    direct_answer=answer or None,
                    metadata={"method_id": "m2a", "fidelity": DESCRIPTOR.fidelity},
                )
        answer = await call_maybe_async(
            self.wrapper.question,
            text=context.text,
            image=[images[0]] if images else None,
            run_sync_in_thread=True,
        )
        item = MemoryItem(
            content=(ContentBlock.text(str(answer)),),
            memory_type="multimodal_retrieval",
            metadata={"method_id": "m2a"},
        )
        return MemoryContribution.from_items(
            [item],
            direct_answer=str(answer),
            metadata={"method_id": "m2a"},
        )

    async def commit(self) -> None:
        if hasattr(self.wrapper, "commit"):
            await call_maybe_async(self.wrapper.commit)

    async def reset(self, scope: MemoryScope | None = None) -> None:
        if hasattr(self.wrapper, "clear"):
            await call_maybe_async(self.wrapper.clear)
        self._conversation_index += 1
        self._active_scope = scope.key() if scope is not None else None
        await self._start_conversation()

    async def save_state(self) -> Mapping[str, Any]:
        core_state = None
        if hasattr(self.wrapper, "export_state"):
            core_state = await call_maybe_async(self.wrapper.export_state)
        return {
            "version": 1,
            "conversation_index": self._conversation_index,
            "started": self._started,
            "active_scope": self._active_scope,
            "core": core_state,
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        self._conversation_index = int(state.get("conversation_index", 0))
        self._started = bool(state.get("started", True))
        self._active_scope = state.get("active_scope")
        core_state = state.get("core")
        if hasattr(self.wrapper, "import_state") and isinstance(core_state, Mapping):
            await call_maybe_async(self.wrapper.import_state, core_state)
        elif self._started:
            await self._start_conversation()

    async def close(self) -> None:
        if self.wrapper is not None and hasattr(self.wrapper, "over"):
            await call_maybe_async(self.wrapper.over, run_sync_in_thread=True)


class M2APlugin(MethodPlugin):
    descriptor = DESCRIPTOR

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return M2AMemory(config, services)


PLUGIN = M2APlugin()
