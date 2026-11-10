from __future__ import annotations

import asyncio
import math
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from mematlas.core.plugin import MethodServices
from mematlas.methods.core_utils import cosine_similarity, lexical_score, parse_json_object, tokens
from mematlas.methods.storage import ascii_database_path, atomic_write_json, method_data_dir, read_json

from . import prompts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ShortTermMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: list[dict[str, Any]] = []

    def get_all(self):
        return list(self.items)


class _LongTermMemory:
    def __init__(self, owner: "MemoryOSCore"):
        self.owner = owner

    def get_raw_user_profile(self, user_id: str):
        del user_id
        return self.owner.user_profile


class _Retriever:
    def __init__(self, owner: "MemoryOSCore"):
        self.owner = owner

    async def retrieve_context(self, **kwargs: Any) -> dict[str, Any]:
        return await self.owner.retrieve_context(**kwargs)


class MemoryOSCore:
    """MemoryOS hierarchy with capacity migration, heat, and long-term consolidation."""

    def __init__(
        self,
        *,
        user_id: str,
        services: MethodServices | None = None,
        assistant_id: str = "default_assistant_profile",
        short_term_capacity: int = 10,
        mid_term_capacity: int = 2000,
        long_term_knowledge_capacity: int = 100,
        retrieval_queue_capacity: int = 7,
        mid_term_heat_threshold: float = 5.0,
        mid_term_similarity_threshold: float = 0.6,
        storage_path: str | None = None,
        **_: Any,
    ):
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.services = services or MethodServices()
        self.short_term_memory = _ShortTermMemory(short_term_capacity)
        self.user_long_term_memory = _LongTermMemory(self)
        self.mid_term_capacity = mid_term_capacity
        self.long_term_knowledge_capacity = long_term_knowledge_capacity
        self.retrieval_queue_capacity = retrieval_queue_capacity
        self.mid_term_heat_threshold = mid_term_heat_threshold
        self.mid_term_similarity_threshold = mid_term_similarity_threshold
        self.mid_term_sessions: dict[str, dict[str, Any]] = {}
        self.user_knowledge: list[dict[str, Any]] = []
        self.assistant_knowledge: list[dict[str, Any]] = []
        self.user_profile = ""
        self._last_evicted_page_id: str | None = None
        self.retriever = _Retriever(self)
        storage_root = method_data_dir(self.services, "memoryos", storage_path)
        self._storage_dir = storage_root / self.user_id
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_files()
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "MemoryOS requires faiss-cpu; install mematlas[native-methods]"
            ) from exc
        self._faiss = faiss
        self._mid_term_index = None
        self._long_term_index = None
        self._mid_term_vector_ids: list[str] = []
        self._long_term_vector_ids: list[str] = []
        self._load_faiss_indexes()

    def _file(self, name: str):
        return self._storage_dir / name

    def _faiss_file(self, name: str) -> Path:
        return ascii_database_path(self._storage_dir, name, services=self.services)

    @staticmethod
    def _fallback_embedding(text: str, dimensions: int = 128) -> list[float]:
        values = [0.0] * dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            values[index] += 1.0 if digest[4] & 1 else -1.0
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values] if norm else [0.0] * dimensions

    async def _embed_text(self, text: str) -> list[float]:
        if self.services.embedder is not None:
            return list((await self.services.embedder.embed([text]))[0])
        return self._fallback_embedding(text)

    def _load_faiss_indexes(self) -> None:
        ids_payload = read_json(self._file("faiss_ids.json"), {})
        self._mid_term_vector_ids = [str(value) for value in ids_payload.get("mid_term", [])]
        self._long_term_vector_ids = [str(value) for value in ids_payload.get("long_term", [])]
        for name, attribute in (("mid_term.faiss", "_mid_term_index"), ("long_term.faiss", "_long_term_index")):
            path = self._faiss_file(name)
            if path.exists():
                try:
                    setattr(self, attribute, self._faiss.read_index(str(path)))
                except Exception:
                    setattr(self, attribute, None)

    def _rebuild_faiss_indexes(self) -> None:
        import numpy as np
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        mid_vectors = []
        mid_ids = []
        for page in self.mid_term_pages:
            vector = page.get("embedding")
            if vector:
                mid_vectors.append(vector)
                mid_ids.append(str(page.get("page_id")))
        if mid_vectors:
            values = np.asarray(mid_vectors, dtype="float32")
            self._faiss.normalize_L2(values)
            index = self._faiss.IndexFlatIP(values.shape[1])
            index.add(values)
            self._mid_term_index = index
            self._mid_term_vector_ids = mid_ids
            target = self._faiss_file("mid_term.faiss")
            target.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(index, str(target))
        else:
            self._mid_term_index = None
            self._mid_term_vector_ids = []
            self._faiss_file("mid_term.faiss").unlink(missing_ok=True)
        long_vectors = []
        long_ids = []
        for item in [*self.user_knowledge, *self.assistant_knowledge]:
            vector = item.get("embedding")
            if vector:
                long_vectors.append(vector)
                long_ids.append(str(item.get("id")))
        if long_vectors:
            values = np.asarray(long_vectors, dtype="float32")
            self._faiss.normalize_L2(values)
            index = self._faiss.IndexFlatIP(values.shape[1])
            index.add(values)
            self._long_term_index = index
            self._long_term_vector_ids = long_ids
            target = self._faiss_file("long_term.faiss")
            target.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(index, str(target))
        else:
            self._long_term_index = None
            self._long_term_vector_ids = []
            self._faiss_file("long_term.faiss").unlink(missing_ok=True)
        atomic_write_json(self._file("faiss_ids.json"), {
            "mid_term": self._mid_term_vector_ids,
            "long_term": self._long_term_vector_ids,
        })

    def _persist(self) -> None:
        atomic_write_json(self._file("short_term.json"), self.short_term_memory.items)
        atomic_write_json(self._file("mid_term.json"), self.mid_term_sessions)
        atomic_write_json(
            self._file("long_term_user.json"),
            {"profile": self.user_profile, "knowledge": self.user_knowledge},
        )
        atomic_write_json(
            self._file("long_term_assistant.json"),
            {"knowledge": self.assistant_knowledge},
        )
        atomic_write_json(
            self._file("meta.json"),
            {"last_evicted_page_id": self._last_evicted_page_id},
        )
        self._rebuild_faiss_indexes()

    def _load_files(self) -> None:
        short_term = read_json(self._file("short_term.json"), [])
        mid_term = read_json(self._file("mid_term.json"), {})
        user_long = read_json(self._file("long_term_user.json"), {})
        assistant_long = read_json(self._file("long_term_assistant.json"), {})
        meta = read_json(self._file("meta.json"), {})
        if isinstance(short_term, list):
            self.short_term_memory.items = [dict(value) for value in short_term if isinstance(value, Mapping)]
        if isinstance(mid_term, Mapping):
            self.mid_term_sessions = {
                str(key): dict(value) for key, value in mid_term.items()
                if isinstance(value, Mapping)
            }
        if isinstance(user_long, Mapping):
            self.user_profile = str(user_long.get("profile", ""))
            self.user_knowledge = [dict(value) for value in user_long.get("knowledge", []) if isinstance(value, Mapping)]
        if isinstance(assistant_long, Mapping):
            self.assistant_knowledge = [dict(value) for value in assistant_long.get("knowledge", []) if isinstance(value, Mapping)]
        if isinstance(meta, Mapping):
            self._last_evicted_page_id = meta.get("last_evicted_page_id")

    @property
    def mid_term_pages(self) -> list[dict[str, Any]]:
        return [
            page
            for session in self.mid_term_sessions.values()
            for page in session.get("details", [])
        ]

    @staticmethod
    def _heat(session: Mapping[str, Any]) -> float:
        timestamp = str(session.get("last_visit_time", ""))
        try:
            last = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            age_hours = max(0.0, (datetime.now(timezone.utc) - last).total_seconds() / 3600)
            recency = math.exp(-age_hours / 24)
        except (TypeError, ValueError):
            recency = 1.0
        return float(session.get("N_visit", 0)) + float(session.get("L_interaction", 0)) + recency

    async def _similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        vectors = (await self._embed_text(left), await self._embed_text(right))
        return cosine_similarity(vectors[0], vectors[1])

    @staticmethod
    def _dialogue_text(pair: Mapping[str, Any]) -> str:
        """Only conversational evidence belongs in LLM inputs, never vectors or IDs."""
        return (f"User: {pair.get('user_input', '')}\nAssistant: {pair.get('agent_response', '')}\n"
                f"Time: {pair.get('timestamp', '')}")

    async def _summarize_topics(self, pair: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
        text = self._dialogue_text(pair)
        if self.services.llm is not None:
            response = await self.services.llm.complete([
                {"role": "system", "content": prompts.TOPIC_SYSTEM},
                {"role": "user", "content": prompts.TOPIC_USER.format(conversation=text)},
            ], temperature=0.0)
            parsed = parse_json_object(str(response))
            if parsed and isinstance(parsed.get("summaries"), list):
                themes = []
                for item in parsed["summaries"]:
                    if not isinstance(item, Mapping) or not isinstance(item.get("content"), str) or not item["content"].strip():
                        continue
                    keywords = item.get("keywords", [])
                    themes.append((item["content"].strip(),
                                   [str(value) for value in keywords] if isinstance(keywords, list) else []))
                    if len(themes) == 2:
                        break
                if themes:
                    return themes
        return [(text, list(dict.fromkeys(tokens(text)))[:12])]

    async def _is_continuous(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        if self.services.llm is None:
            return False
        response = await self.services.llm.complete([
            {"role": "system", "content": prompts.CONTINUITY_SYSTEM},
            {"role": "user", "content": prompts.CONTINUITY_USER.format(
                previous=self._dialogue_text(previous), current=self._dialogue_text(current))},
        ], temperature=0.0)
        parsed = parse_json_object(str(response))
        if parsed is None or type(parsed.get("continuous")) is not bool:
            raise ValueError("MemoryOS continuity response requires a boolean continuous field")
        return parsed["continuous"]

    async def _generate_meta_info(self, pair: Mapping[str, Any], previous_summary: str | None) -> str:
        text = self._dialogue_text(pair)
        if self.services.llm is None:
            return text
        response = await self.services.llm.complete([
            {"role": "system", "content": prompts.META_INFO_SYSTEM},
            {"role": "user", "content": prompts.META_INFO_USER.format(
                previous_summary=previous_summary or "None", conversation=text)},
        ], temperature=0.3)
        parsed = parse_json_object(str(response))
        if parsed is None or not isinstance(parsed.get("meta_info"), str) or not parsed["meta_info"].strip():
            raise ValueError("MemoryOS meta-summary response requires a nonempty meta_info field")
        return parsed["meta_info"].strip()

    async def _migrate_to_mid_term(self, pair: Mapping[str, Any]) -> None:
        previous = next((item for item in self.mid_term_pages if item["page_id"] == self._last_evicted_page_id), None)
        continuous = previous is not None and await self._is_continuous(previous, pair)
        meta_info = await self._generate_meta_info(pair, previous.get("meta_info") if continuous else None)
        themes = await self._summarize_topics(pair)
        page = {**dict(pair), "page_id": uuid4().hex, "memory_id": uuid4().hex,
                "preloaded": False, "analyzed": False, "pre_page": None, "next_page": None,
                "meta_info": meta_info, "page_keywords": themes[0][1],
                "embedding": await self._embed_text(f"User: {pair.get('user_input', '')} Assistant: {pair.get('agent_response', '')}")}
        if continuous:
            page["pre_page"] = previous["page_id"]
            connected = {previous["page_id"]}
            changed = True
            while changed:
                changed = False
                for item in self.mid_term_pages:
                    if item["page_id"] in connected or item.get("pre_page") in connected or item.get("next_page") in connected:
                        if item["page_id"] not in connected:
                            connected.add(item["page_id"])
                            changed = True
                        item["meta_info"] = meta_info
                        if item["page_id"] == previous["page_id"]:
                            item["next_page"] = page["page_id"]
        for theme, theme_keywords in themes:
            await self._insert_page({**page, "page_keywords": theme_keywords}, theme, theme_keywords)
        self._last_evicted_page_id = page["page_id"]
        self._evict_mid_term_if_needed()

    async def _insert_page(self, page, summary, keywords):
        summary_vector = await self._embed_text(summary)
        best_id, best_score = None, -1.0
        for session_id, session in self.mid_term_sessions.items():
            vector = session.get("summary_embedding") or await self._embed_text(session.get("summary", ""))
            score = cosine_similarity(summary_vector, vector)
            left, right = set(keywords), set(session.get("summary_keywords", []))
            if left and right:
                score += len(left & right) / len(left | right)
            if score > best_score:
                best_id, best_score = session_id, score
        now = _now()
        if best_id is None or best_score < self.mid_term_similarity_threshold:
            best_id = uuid4().hex
            self.mid_term_sessions[best_id] = {
                "id": best_id, "summary": summary, "summary_embedding": summary_vector,
                "summary_keywords": keywords, "details": [], "L_interaction": 0,
                "R_recency": 1.0, "N_visit": 0, "H_segment": 0.0,
                "timestamp": now, "last_visit_time": now, "access_count_lfu": 0}
        session = self.mid_term_sessions[best_id]
        session["details"].append(dict(page))
        session["L_interaction"] += 1
        session["last_visit_time"] = now
        session["H_segment"] = self._heat(session)

    def _evict_mid_term_if_needed(self) -> None:
        while len(self.mid_term_sessions) > self.mid_term_capacity:
            victim = min(
                self.mid_term_sessions,
                key=lambda key: self.mid_term_sessions[key].get("access_count_lfu", 0),
            )
            self.mid_term_sessions.pop(victim, None)

    async def _consolidate_hot_sessions(self) -> None:
        if self.services.llm is None:
            return
        for session in sorted(self.mid_term_sessions.values(), key=self._heat, reverse=True)[:1]:
            if self._heat(session) < self.mid_term_heat_threshold:
                continue
            pages = [page for page in session.get("details", []) if not page.get("analyzed")]
            if not pages:
                continue
            conversation = "\n".join(
                f"User: {page.get('user_input', '')} (Timestamp: {page.get('timestamp', '')})\n"
                f"Assistant: {page.get('agent_response', '')} (Timestamp: {page.get('timestamp', '')})"
                for page in pages
            )
            existing_profile = self.user_profile
            if not existing_profile or existing_profile.lower() == "none":
                existing_profile = "No existing profile data."
            # Wait for both native parallel tasks before mutating or rolling back state.
            results = await asyncio.gather(
                self.services.llm.complete([
                    {"role": "system", "content": prompts.PROFILE_SYSTEM},
                    {"role": "user", "content": prompts.PROFILE_USER.format(
                        existing_user_profile=existing_profile, conversation=conversation)},
                ], temperature=0.7, max_tokens=2000),
                self.services.llm.complete([
                    {"role": "system", "content": prompts.KNOWLEDGE_SYSTEM},
                    {"role": "user", "content": prompts.KNOWLEDGE_USER.format(conversation=conversation)},
                ], temperature=0),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            profile_response, knowledge_response = results
            knowledge_data = parse_json_object(str(knowledge_response))
            if knowledge_data is None:
                raise ValueError("MemoryOS knowledge response must be a JSON object")
            profile = profile_response.strip() if profile_response else "None"
            if profile and profile.lower() != "none" and len(profile) >= 30:
                self.user_profile = profile
            await self._append_knowledge(self.user_knowledge, knowledge_data.get("user_knowledge", []))
            await self._append_knowledge(self.assistant_knowledge, knowledge_data.get("assistant_knowledge", []))
            for page in pages:
                page["analyzed"] = True
            session["N_visit"] = 0
            session["L_interaction"] = 0
            session["H_segment"] = self._heat(session)

    async def _append_knowledge(self, target: list[dict[str, Any]], values: Any) -> None:
        if isinstance(values, str):
            values = [line.strip(" -") for line in values.splitlines() if line.strip(" -")]
        if not isinstance(values, list):
            return
        existing = {str(item.get("knowledge", "")).casefold() for item in target}
        for value in values:
            text = str(value.get("knowledge", "") if isinstance(value, Mapping) else value).strip()
            if text and text.casefold() not in {"none", "- none", "- none."} and text.casefold() not in existing:
                target.append({"id": uuid4().hex, "knowledge": text, "timestamp": _now(), "embedding": await self._embed_text(text)})
                existing.add(text.casefold())
        del target[:-self.long_term_knowledge_capacity]

    async def add_memory(
        self,
        user_input: str,
        agent_response: str,
        *,
        timestamp: str = "",
        meta_data: Mapping[str, Any] | None = None,
    ):
        pair = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp or _now(),
            "meta_data": dict(meta_data or {}),
        }
        if len(self.short_term_memory.items) >= self.short_term_memory.capacity:
            oldest = self.short_term_memory.items.pop(0)
            await self._migrate_to_mid_term(oldest)
        self.short_term_memory.items.append(pair)
        await self._consolidate_hot_sessions()
        self._persist()

    async def retrieve_context(
        self,
        *,
        user_query: str,
        user_id: str,
        segment_similarity_threshold: float = 0.1,
        page_similarity_threshold: float = 0.1,
        knowledge_threshold: float = 0.01,
        top_k_sessions: int = 5,
        top_k_knowledge: int = 20,
        read_only: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        del user_id
        vector_scores: dict[str, float] = {}
        query_embedding = await self._embed_text(user_query)
        if self._mid_term_index is not None:
            import numpy as np
            query_vector = np.asarray([query_embedding], dtype="float32")
            self._faiss.normalize_L2(query_vector)
            scores, indices = self._mid_term_index.search(query_vector, self._mid_term_index.ntotal)
            vector_scores = {
                self._mid_term_vector_ids[index]: float(score)
                for score, index in zip(scores[0], indices[0])
                if 0 <= index < len(self._mid_term_vector_ids)
            }
        matching_sessions = []
        sessions = list(self.mid_term_sessions.values())
        if sessions:
            import numpy as np
            vectors = [session.get("summary_embedding") or await self._embed_text(session.get("summary", "")) for session in sessions]
            matrix = np.asarray(vectors, dtype="float32")
            self._faiss.normalize_L2(matrix)
            query = np.asarray([query_embedding], dtype="float32")
            self._faiss.normalize_L2(query)
            index = self._faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            scores, indices = index.search(query, min(top_k_sessions, len(sessions)))
            matching_sessions = [(float(score), sessions[int(i)]) for score, i in zip(scores[0], indices[0])
                                 if i >= 0 and score >= segment_similarity_threshold]
        pages = []
        for _, session in matching_sessions[:top_k_sessions]:
            matched = []
            for page in session.get("details", []):
                score = vector_scores.get(str(page.get("page_id")), -1.0)
                if score >= page_similarity_threshold:
                    matched.append({**page, "score": score})
            if matched and not read_only:
                session["N_visit"] += 1
                session["access_count_lfu"] += 1
                session["last_visit_time"] = _now()
                session["H_segment"] = self._heat(session)
            pages.extend(matched)
        pages.sort(key=lambda item: item["score"], reverse=True)

        if not read_only:
            self._persist()
        knowledge_scores = {}
        if self._long_term_index is not None:
            import numpy as np
            query_vector = np.asarray([query_embedding], dtype="float32")
            self._faiss.normalize_L2(query_vector)
            scores, indices = self._long_term_index.search(query_vector, self._long_term_index.ntotal)
            knowledge_scores = {self._long_term_vector_ids[int(index)]: float(score)
                                for score, index in zip(scores[0], indices[0]) if index >= 0}

        async def search_knowledge(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
            ranked = [{**item, "score": knowledge_scores[item["id"]]} for item in values
                      if knowledge_scores.get(item["id"], -1.0) >= knowledge_threshold]
            return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k_knowledge]

        return {
            "retrieved_pages": pages[: self.retrieval_queue_capacity],
            "retrieved_user_knowledge": await search_knowledge(self.user_knowledge),
            "retrieved_assistant_knowledge": await search_knowledge(self.assistant_knowledge),
            "retrieved_at": _now(),
        }

    def get_memory_stats(self):
        return {
            "short_term": len(self.short_term_memory.items),
            "mid_term_sessions": len(self.mid_term_sessions),
            "mid_term_pages": len(self.mid_term_pages),
            "user_knowledge": len(self.user_knowledge),
            "assistant_knowledge": len(self.assistant_knowledge),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "short_term": list(self.short_term_memory.items),
            "mid_term_sessions": self.mid_term_sessions,
            "user_knowledge": self.user_knowledge,
            "assistant_knowledge": self.assistant_knowledge,
            "user_profile": self.user_profile,
            "last_evicted_page_id": self._last_evicted_page_id,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        self.short_term_memory.items = [dict(item) for item in state.get("short_term", [])]
        self.mid_term_sessions = {
            str(key): dict(value)
            for key, value in state.get("mid_term_sessions", {}).items()
            if isinstance(value, Mapping)
        }
        self.user_knowledge = [dict(item) for item in state.get("user_knowledge", [])]
        self.assistant_knowledge = [dict(item) for item in state.get("assistant_knowledge", [])]
        self.user_profile = str(state.get("user_profile", ""))
        self._last_evicted_page_id = state.get("last_evicted_page_id")
        self._persist()

    async def close(self) -> None:
        self._persist()
