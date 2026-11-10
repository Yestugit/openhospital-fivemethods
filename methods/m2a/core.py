from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from mematlas.core.plugin import MethodServices
from mematlas.methods.core_utils import parse_json_object, vision_content
from mematlas.methods.storage import SQLiteState, method_data_dir, load_mapping


class M2ACore:
    """M2A raw-message layer plus managed semantic text-image layer."""

    def __init__(self, services: MethodServices, *, top_k: int = 8, context_window: int = 20,
                 db_dir: str | None = None, storage_path: str | None = None,
                 implementation_mode: str = "standard", multimodal_model: str | None = None,
                 multimodal_base_url: str | None = None,
                 multimodal_api_key_env: str = "OPENAI_API_KEY", **_: Any):
        self.services = services
        self.top_k = top_k
        self.context_window = context_window
        self.offline = implementation_mode == "offline"
        self.multimodal = services.object("m2a.multimodal_embedder")
        if self.multimodal is None and multimodal_model:
            if not multimodal_base_url:
                raise ValueError("M2A multimodal_model requires multimodal_base_url")
            from .embeddings import MultimodalEmbedder
            self.multimodal = MultimodalEmbedder(multimodal_model, multimodal_base_url, multimodal_api_key_env)
        if self.multimodal is not None:
            from mematlas.providers.instrumentation import MeteredService
            self.multimodal = MeteredService(self.multimodal, services.usage, "multimodal_embedding")
        self.conversations: dict[int, list[dict[str, Any]]] = {}
        self.semantic_memories: dict[str, dict[str, Any]] = {}
        self.current = 0
        self._next_message_id = 1
        self._last_semantic_message_id = 0
        root = method_data_dir(services, "m2a", db_dir or storage_path)
        self._raw_store = SQLiteState(root / "raw.db")
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "M2A requires pymilvus[milvus-lite]; install mematlas[native-methods]"
            ) from exc
        self._semantic_store = MilvusClient(str(root / "semantic.db"))
        self._semantic_collection = "semantic_memory"
        self._image_collection = "image_embeddings"
        self._semantic_dimension = 128
        self._indexed_ids: set[str] = set()
        self._load_persistent_state()

    def _ensure_semantic_collection(self) -> None:
        if not self._semantic_store.has_collection(self._semantic_collection):
            from pymilvus import DataType, Function, FunctionType
            schema = self._semantic_store.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._semantic_dimension)
            schema.add_field("search_text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                input_field_names=["search_text"], output_field_names=["sparse"]))
            indexes = self._semantic_store.prepare_index_params()
            indexes.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="IP")
            indexes.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
            self._semantic_store.create_collection(
                collection_name=self._semantic_collection,
                schema=schema, index_params=indexes,
            )

    def _load_persistent_state(self) -> None:
        raw = load_mapping(self._raw_store, "raw")
        semantic = {}
        self.conversations = {
            int(key): [dict(turn) for turn in turns]
            for key, turns in raw.get("conversations", {}).items()
            if isinstance(turns, list)
        }
        self.current = int(raw.get("current", 0))
        self._next_message_id = int(raw.get("next_message_id", 1))
        self._last_semantic_message_id = int(raw.get("last_semantic_message_id", 0))
        if not self._semantic_store.has_collection(self._semantic_collection):
            return
        records = self._semantic_store.query(
            collection_name=self._semantic_collection,
            output_fields=["*"],
        )
        self.semantic_memories = {}
        for value in records:
            item = dict(value)
            item_id = str(item.get("id", ""))
            if item_id:
                item.pop("id", None)
                item["embedding"] = list(item.pop("vector", []))
                item["id"] = item_id
                item.pop("search_text", None)
                item.pop("sparse", None)
                for key in ("image_paths", "evidence_ids"):
                    if isinstance(item.get(key), str):
                        try:
                            item[key] = json.loads(item[key])
                        except json.JSONDecodeError:
                            item[key] = []
                self.semantic_memories[item_id] = item
                self._indexed_ids.add(item_id)
        if self._semantic_store.has_collection(self._image_collection):
            for value in self._semantic_store.query(self._image_collection, output_fields=["*"]):
                parent = str(value["memory_id"])
                if parent in self.semantic_memories:
                    self.semantic_memories[parent].setdefault("image_embeddings", {})[value["image_path"]] = list(value["vector"])
        # Upgrade the earlier dense-only local schema from authoritative rows.
        fields = self._semantic_store.describe_collection(self._semantic_collection)["fields"]
        if not any(field["name"] == "sparse" for field in fields):
            self._semantic_store.drop_collection(self._semantic_collection)
            self._indexed_ids.clear()
            self._persist()
        if not self.semantic_memories and isinstance(semantic, Mapping):
            self.semantic_memories = {
                str(key): dict(value)
                for key, value in semantic.get("semantic_memories", {}).items()
                if isinstance(value, Mapping)
            }

    def _persist(self) -> None:
        self._raw_store.save("raw", {
            "conversations": self.conversations,
            "current": self.current,
            "next_message_id": self._next_message_id,
            "last_semantic_message_id": self._last_semantic_message_id,
        })
        rows = []
        for item in self.semantic_memories.values():
            vector = item.get("embedding")
            if vector is None:
                raise ValueError("M2A must embed semantic memories before persistence")
            item["embedding"] = list(vector)
            rows.append({
                "id": str(item["id"]),
                "vector": list(vector),
                "search_text": f"{item.get('text', '')}\nWith image: {item.get('image_caption', '')}".strip(),
                "text": str(item.get("text", "")),
                "image_caption": str(item.get("image_caption", "")),
                "image_paths": json.dumps(item.get("image_paths", []), ensure_ascii=False),
                "evidence_ids": json.dumps(item.get("evidence_ids", []), ensure_ascii=False),
                "created_at": str(item.get("created_at", "")),
            })
        if rows:
            self._semantic_dimension = len(rows[0]["vector"])
            if any(len(row["vector"]) != self._semantic_dimension for row in rows):
                raise ValueError("M2A semantic vectors have inconsistent dimensions")
            self._ensure_semantic_collection()
            self._semantic_store.upsert(self._semantic_collection, rows)
        removed = self._indexed_ids - self.semantic_memories.keys()
        if removed and self._semantic_store.has_collection(self._semantic_collection):
            self._semantic_store.delete(self._semantic_collection, ids=list(removed))
        self._indexed_ids = set(self.semantic_memories)
        self._persist_images()

    def _persist_images(self) -> None:
        rows = [{"id": f"{item['id']}:{index}", "memory_id": item["id"],
                 "image_path": uri, "vector": vector}
                for item in self.semantic_memories.values()
                for index, (uri, vector) in enumerate(item.get("image_embeddings", {}).items())]
        if rows and not self._semantic_store.has_collection(self._image_collection):
            self._semantic_store.create_collection(self._image_collection,
                dimension=len(rows[0]["vector"]), id_type="string", primary_field_name="id",
                vector_field_name="vector", metric_type="IP", auto_id=False)
        if self._semantic_store.has_collection(self._image_collection):
            existing = self._semantic_store.query(self._image_collection, output_fields=["id"])
            stale = {row["id"] for row in existing} - {row["id"] for row in rows}
            if stale:
                self._semantic_store.delete(self._image_collection, ids=list(stale))
            if rows:
                self._semantic_store.upsert(self._image_collection, rows)

    @staticmethod
    def _fallback_embedding(text: str, dimensions: int = 128) -> list[float]:
        import hashlib
        values = [0.0] * dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            values[index] += 1.0 if digest[4] & 1 else -1.0
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values] if norm else [0.0] * dimensions

    def start_conversation(self, info: Mapping[str, Any]):
        self.current = int(info.get("conv_idx", len(self.conversations)))
        self.conversations.setdefault(self.current, [])

    @staticmethod
    def _image_caption(images: list[str], provided: Any = None) -> str:
        if provided:
            return str(provided)
        return ", ".join(Path(image).stem.replace("_", " ") for image in images)

    async def chat(self, turns: list[dict[str, Any]]):
        if any(turn.get("images") for turn in turns) and self.multimodal is None and not self.offline:
            raise ValueError("M2A image input requires a multimodal embedding service; configure multimodal_model/base_url")
        conversation = self.conversations.setdefault(self.current, [])
        for raw_turn in turns:
            turn = dict(raw_turn)
            turn["msg_id"] = self._next_message_id
            self._next_message_id += 1
            turn.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            turn["images"] = list(turn.get("images", []))
            conversation.append(turn)
            if int(turn.get("speaker", 0)) == 1:
                await self._update_semantic(conversation[-self.context_window :])
                self._last_semantic_message_id = int(turn["msg_id"])
                await self._ensure_embeddings()
                self._persist()
        await self._ensure_embeddings()
        self._persist()

    async def _update_semantic(self, context: list[dict[str, Any]]) -> None:
        if not context:
            return
        operations = await self._semantic_operations(context)
        if operations is None:
            self._fallback_add(context)
            return
        for operation in operations:
            action = str(operation.get("action", "")).upper()
            memory_id = str(operation.get("id") or operation.get("memory_id") or "")
            if action in {"DELETE", "REMOVE"}:
                self.semantic_memories.pop(memory_id, None)
            elif action in {"ADD", "CREATE", "INSERT"}:
                self._add_semantic(
                    text=str(operation.get("text") or operation.get("content") or ""),
                    image_paths=self._operation_images(operation, context),
                    image_caption=str(operation.get("image_caption", "")),
                    evidence_ids=self._normalize_evidence(operation.get("evidence_ids"), context),
                )

    async def _semantic_operations(self, context: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if self.services.llm is None:
            return None
        existing = await self.search(" ".join(str(turn.get("text", "")) for turn in context))
        messages = [{"role": "user", "content": vision_content(
            "Maintain high-level semantic text-image memories from raw dialogue. Return JSON "
            "with operations (ADD/DELETE) and optional reads. ADD fields: text, image_path, "
            "image_caption, evidence_ids as [[start,end]]. DELETE uses id. Resolve contradictions "
            "by deleting old memories and adding refined memories. Skip greetings and duplicates. "
            "If more evidence is needed, request reads: {action:SEARCH,text,images,top_k}, "
            "{action:RAW,evidence_ids:[[start,end]]}, or {action:TIME,start_date,end_date}. "
            "Read results will be returned for a follow-up decision. Final response has no reads.\n"
            f"Relevant semantic memories: {existing}\nRaw messages: {context}",
            [uri for turn in context for uri in turn.get("images", [])])}]
        operations = []
        for _ in range(8):
            response = await self.services.llm.complete(messages, temperature=0.0)
            parsed = parse_json_object(str(response))
            if not parsed or not isinstance(parsed.get("operations"), list):
                raise ValueError("M2A model response must contain an operations array")
            operations.extend(dict(item) for item in parsed["operations"] if isinstance(item, Mapping))
            reads = parsed.get("reads", [])
            if not reads:
                return operations
            results = []
            for request in reads:
                action = str(request.get("action", "")).upper()
                if action == "SEARCH":
                    result = await self.search(str(request.get("text", "")), request.get("images"),
                                               min(int(request.get("top_k", 8)), 20))
                elif action == "RAW":
                    result = self._raw_evidence(request.get("evidence_ids", []))[:self.context_window]
                elif action == "TIME":
                    start = str(request["start_date"])
                    end = str(request["end_date"])
                    result = [turn for turns in self.conversations.values() for turn in turns
                              if start <= str(turn.get("timestamp", ""))[:10] <= end][:self.context_window]
                else:
                    raise ValueError(f"Unknown M2A read operation: {action}")
                results.append({"request": request, "result": result})
            messages.extend([{"role": "assistant", "content": str(response)},
                             {"role": "user", "content": "Memory tool results: " + json.dumps(results, ensure_ascii=False)}])
        raise ValueError("M2A memory manager exceeded 8 evidence-reading rounds")

    @staticmethod
    def _operation_images(operation: Mapping[str, Any], context: list[dict[str, Any]]) -> list[str]:
        value = operation.get("image_paths", operation.get("image_path"))
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(item) for item in value if item]
        evidence = M2ACore._normalize_evidence(operation.get("evidence_ids"), context)
        return [
            str(image)
            for turn in context
            if any(
                start <= int(turn.get("msg_id", -1)) <= end
                for start, end in evidence
            )
            for image in turn.get("images", [])
        ]

    @staticmethod
    def _normalize_evidence(value: Any, context: list[dict[str, Any]]) -> list[list[int]]:
        if isinstance(value, list):
            normalized = []
            for pair in value:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    normalized.append([int(pair[0]), int(pair[1])])
            if normalized:
                return normalized
        ids = [int(turn["msg_id"]) for turn in context if turn.get("msg_id") is not None]
        return [[min(ids), max(ids)]] if ids else []

    def _fallback_add(self, context: list[dict[str, Any]]) -> None:
        useful = [turn for turn in context if str(turn.get("text", "")).strip() or turn.get("images")]
        if not useful:
            return
        recent = useful[-2:]
        text = "\n".join(
            f"speaker_{turn.get('speaker', 0)}: {str(turn.get('text', '')).strip()}"
            for turn in recent
            if str(turn.get("text", "")).strip()
        )
        images = [str(image) for turn in recent for image in turn.get("images", [])]
        self._add_semantic(
            text=text,
            image_paths=images,
            image_caption=self._image_caption(images),
            evidence_ids=self._normalize_evidence(None, recent),
        )

    def _add_semantic(
        self,
        *,
        text: str,
        image_paths: list[str],
        image_caption: str,
        evidence_ids: list[list[int]],
    ) -> None:
        text = text.strip()
        images = list(dict.fromkeys(image_paths))
        if not text and not images:
            return
        normalized = " ".join(text.casefold().split())
        duplicate = next(
            (
                item
                for item in self.semantic_memories.values()
                if " ".join(str(item["text"]).casefold().split()) == normalized
                and item["image_paths"] == images
            ),
            None,
        )
        if duplicate:
            for pair in evidence_ids:
                if pair not in duplicate["evidence_ids"]:
                    duplicate["evidence_ids"].append(pair)
            return
        memory_id = uuid4().hex
        self.semantic_memories[memory_id] = {
            "id": memory_id,
            "text": text,
            "image_paths": images,
            "image_caption": image_caption or self._image_caption(images),
            "evidence_ids": evidence_ids,
            "embedding": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _ensure_embeddings(self) -> None:
        for item in self.semantic_memories.values():
            vectors = item.setdefault("image_embeddings", {})
            for uri in item["image_paths"]:
                if uri not in vectors:
                    if self.multimodal is None and not self.offline:
                        raise ValueError("M2A image memories require a multimodal embedding service")
                    vectors[uri] = (await self.multimodal.encode_image(uri) if self.multimodal
                                    else self._fallback_embedding(uri))
        pending = [item for item in self.semantic_memories.values() if item.get("embedding") is None]
        if not pending:
            return
        texts = [f"{item['text']}\n{item['image_caption']}".strip() for item in pending]
        embeddings = (
            await self.services.embedder.embed(texts)
            if self.services.embedder is not None
            else [self._fallback_embedding(text) for text in texts]
        )
        for item, embedding in zip(pending, embeddings):
            item["embedding"] = list(embedding)

    @staticmethod
    def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, memory_id in enumerate(ranking, start=1):
                scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)
        return scores

    async def search(self, text: str = "", images: list[str] | None = None, top_k: int | None = None):
        images = list(images or [])
        await self._ensure_embeddings()
        memories = list(self.semantic_memories.values())
        if not memories:
            return []
        from pymilvus import AnnSearchRequest, RRFRanker
        rankings = []
        if text:
            vector = ((await self.services.embedder.embed([text]))[0] if self.services.embedder
                      else self._fallback_embedding(text))
            hits = self._semantic_store.hybrid_search(
                collection_name=self._semantic_collection,
                reqs=[AnnSearchRequest(data=[vector], anns_field="vector", param={"metric_type": "IP"}, limit=5),
                      AnnSearchRequest(data=[text], anns_field="sparse", param={"metric_type": "BM25"}, limit=5)],
                ranker=RRFRanker(), limit=5,
            )
            rankings.append([str(hit["id"]) for hit in hits[0]])
        for uri in images:
            if self.multimodal is None and not self.offline:
                raise ValueError("M2A image queries require a multimodal embedding service")
            if not self._semantic_store.has_collection(self._image_collection):
                continue
            vector = (await self.multimodal.encode_image(uri) if self.multimodal
                      else self._fallback_embedding(uri))
            hits = self._semantic_store.search(self._image_collection, data=[vector],
                anns_field="vector", limit=3, output_fields=["memory_id"],
                search_params={"metric_type": "IP"})
            rankings.append(list(dict.fromkeys(str(hit["entity"]["memory_id"]) for hit in hits[0])))
        scores = self._rrf(rankings)
        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k or self.top_k]
        return [{**self.semantic_memories[key], "score": scores[key]} for key in ordered]

    def _raw_evidence(self, ranges: list[list[int]]) -> list[dict[str, Any]]:
        output = []
        for conversation in self.conversations.values():
            for turn in conversation:
                message_id = int(turn.get("msg_id", -1))
                if any(start <= message_id <= end for start, end in ranges):
                    output.append(dict(turn))
        output.sort(key=lambda item: int(item.get("msg_id", 0)))
        return output

    async def query(self, text: str = "", images: list[str] | None = None, top_k: int | None = None):
        memories = await self.search(text, images, top_k)
        evidence_ranges = [pair for item in memories for pair in item["evidence_ids"]]
        evidence = self._raw_evidence(evidence_ranges)
        context = "\n".join(
            f"Memory: {item['text']}"
            + (f" [image: {', '.join(item['image_paths'])}]" if item["image_paths"] else "")
            + f" [evidence: {item['evidence_ids']}]"
            for item in memories
        )
        if not memories:
            evidence = list(self.conversations.get(self.current, []))[-self.context_window :]
        if evidence:
            context += "\nRaw evidence:\n" + "\n".join(
                f"[{turn['msg_id']}] speaker_{turn.get('speaker', 0)}: {turn.get('text', '')}"
                for turn in evidence
            )
        answer = context
        if self.services.llm is not None and context:
            answer = await self.services.llm.complete([
                {"role": "system", "content": "Answer using retrieved text-image memories and their raw evidence."},
                {"role": "user", "content": vision_content(f"Memory:\n{context}\n\nQuestion: {text}",
                    [*(images or []), *(uri for item in memories for uri in item["image_paths"])])},
            ])
        return {"answer": answer, "memories": memories, "evidence": evidence}

    async def question(self, text: str = "", image=None):
        result = await self.query(text, list(image or []))
        return result["answer"]

    def clear(self) -> None:
        self.conversations.clear()
        self.semantic_memories.clear()
        self.current = 0
        self._next_message_id = 1
        self._persist()

    def export_state(self) -> dict[str, Any]:
        return {
            "conversations": self.conversations,
            "semantic_memories": self.semantic_memories,
            "current": self.current,
            "next_message_id": self._next_message_id,
            "last_semantic_message_id": self._last_semantic_message_id,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        self.conversations = {
            int(key): [dict(turn) for turn in turns]
            for key, turns in state.get("conversations", {}).items()
            if isinstance(turns, list)
        }
        self.semantic_memories = {
            str(key): dict(value)
            for key, value in state.get("semantic_memories", {}).items()
            if isinstance(value, Mapping)
        }
        self.current = int(state.get("current", 0))
        self._next_message_id = int(state.get("next_message_id", 1))
        self._last_semantic_message_id = int(state.get("last_semantic_message_id", 0))
        self._persist()

    async def commit(self) -> None:
        conversation = self.conversations.get(self.current, [])
        if conversation and int(conversation[-1]["msg_id"]) > self._last_semantic_message_id:
            await self._update_semantic(conversation[-self.context_window:])
            await self._ensure_embeddings()
            self._last_semantic_message_id = int(conversation[-1]["msg_id"])
        self._persist()

    def over(self, **_: Any):
        self._persist()
        self._raw_store.close()
        self._semantic_store.close()
