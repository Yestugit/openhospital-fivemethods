"""Fixed official EMem source with provider and checkpoint adapters only."""
from __future__ import annotations

from contextlib import closing
import asyncio
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from threading import RLock
from types import SimpleNamespace

from mematlas.core.models import content_text
from mematlas.core.provider_config import resolve_providers
from .utils import canonical, checked_source, digest, get_blob, namespace, put_blob, run_service, storage_path

REVISION = "4390269ecda5ab7e3ecce0e10bf8ebfd692ec406"
_IMPORT_LOCK = RLock()


def trace_value(value):
    """Lossless JSON view of native diagnostics, including tuple-key maps."""
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: trace_value(item) for key, item in value.items()}
        return {"__native_mapping_items__": [[trace_value(key), trace_value(item)]
                                               for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return [trace_value(item) for item in value]
    if hasattr(value, "tolist"):
        return trace_value(value.tolist())
    return value


class NativeEMem:
    def __init__(self, directory, scope, params, services, state=None):
        self.directory = storage_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.params, self.services, self.scope = params, services, scope
        self.io_loop = getattr(services, "_native_io_loop", None)
        self.next_session = 1
        self.sources = {}
        self.failures = []
        root = checked_source(services, "emem", REVISION)
        if state is not None:
            self.validate_state(state)
            self._restore_files(state)
            self.next_session, self.sources = state["next_session"], state["sources"]
        providers = resolve_providers(params, services.provider_configs)
        llm_config, embedding_config = providers["llm"], providers["embedding"]
        self.providers = providers
        with _IMPORT_LOCK:
            namespace("emem", root / "src" / "emem")
            namespace("emem.evaluation", root / "src" / "emem" / "evaluation")
            logging_module = importlib.import_module("emem.utils.logging_utils")
            original_logger = getattr(logging_module, "_mematlas_original_logger", logging_module.get_logger)
            logging_module._mematlas_original_logger = original_logger
            logging_module.LOG_DIR = str(self.directory / "logs")
            Path(logging_module.LOG_DIR).mkdir(parents=True, exist_ok=True)
            def portable_logger(name, log_file=None, level=logging_module.LOG_LEVEL):
                # Upstream defaults to `emem->...log`, invalid on Windows.
                # Keep its handlers/formatters; only select a portable filename.
                return original_logger(name, log_file=log_file or name.replace(".", "_") + ".log", level=level)
            logging_module.get_logger = portable_logger
            # The upstream embedding factory eagerly imports every local model.
            # Only replace that provider-selection module; retain its base/cache,
            # all stores, OpenIE, graph construction and retrieval verbatim.
            embedding_package = namespace("emem.embedding_model", root / "src" / "emem" / "embedding_model")
            base = importlib.import_module("emem.embedding_model.base")
            embedding_package.BaseEmbeddingModel = base.BaseEmbeddingModel
            embedding_package._get_embedding_model_class = lambda **kw: self._embedding_class(base)
            config_module = importlib.import_module("emem.utils.config_utils")
            schema_mode = params.get("support_json_schema", False)
            from mematlas.core.errors import ConfigurationError
            prior_mode = getattr(config_module, "_mematlas_schema_mode", schema_mode)
            if prior_mode != schema_mode:
                raise ConfigurationError("EMem prompt JSON-schema modes require separate executor processes")
            config_module.get_support_json_schema = lambda: schema_mode
            # Upstream's PromptTemplateManager tries src.emem before emem.
            # A source-root PYTHONPATH in the isolated worker would import a
            # second EMem package (and its optional eager imports). Alias each
            # exact original template under that legacy spelling, preserving
            # one class/schema/logger identity and the unmodified loader.
            for template in (root / "src" / "emem" / "prompts" / "templates").glob("*.py"):
                if template.stem != "__init__":
                    loaded = importlib.import_module("emem.prompts.templates." + template.stem)
                    sys.modules["src.emem.prompts.templates." + template.stem] = loaded
            module = importlib.import_module("emem.EMem")
            config_module._mematlas_schema_mode = schema_mode
            BaseConfig = config_module.BaseConfig
            cfg = BaseConfig(save_dir=str(self.directory / "native"),
                llm_name=llm_config.model or getattr(services.llm, "model", "configured-llm"),
                llm_base_url=llm_config.base_url,
                embedding_model_name=embedding_config.model or getattr(services.embedder, "model", "configured-embedding"),
                embedding_base_url=embedding_config.base_url,
                openie_mode="edu_based_contextual_ee_online", skip_edu_context_gen=True,
                skip_retrieval_ppr=params.get("variant", "emem") == "emem",
                linking_top_k=30, retrieval_top_k=params.get("top_k", 10), qa_top_k=params.get("top_k", 10),
                synonymy_edge_sim_threshold=0.9, seed=42, max_new_tokens=8192)
            for key in ("temperature", "seed"):
                if key in llm_config.options:
                    setattr(cfg, key, llm_config.options[key])
            if "max_tokens" in llm_config.options:
                cfg.max_new_tokens = llm_config.options["max_tokens"]
            original_llm = module._get_llm_class
            original_embedding = module._get_embedding_model_class
            structuring = importlib.import_module("emem.information_extraction.structuring_openai")
            original_schema = structuring.get_support_json_schema
            # Scope this override to native construction; no shared environment
            # changes and no model-name spoofing for Qwen embeddings.
            module._get_llm_class = self._llm
            module._get_embedding_model_class = lambda **kw: self._embedding_class(base)
            structuring.get_support_json_schema = lambda: params.get("support_json_schema", False)
            config_module.get_support_json_schema = lambda: schema_mode
            try:
                self.engine = module.EMem(global_config=cfg)
            finally:
                module._get_llm_class, module._get_embedding_model_class = original_llm, original_embedding
                structuring.get_support_json_schema = original_schema
                # Native rerank functions import this getter dynamically during
                # retrieval. Keep the process-local explicit capability binding;
                # conflicting modes are rejected above before any native work.
                config_module.get_support_json_schema = lambda: schema_mode
        if state is not None:
            self.engine.speaker_a, self.engine.speaker_b = state["speakers"]

    def _embedding_class(self, base):
        services = self.services
        io_loop = self.io_loop

        class PublicEmbedding(base.BaseEmbeddingModel):
            def __init__(self, global_config=None, embedding_model_name=None):
                super().__init__(global_config=global_config)
                self.embedding_config = base.EmbeddingConfig.from_dict({"norm": True, "encode_params": {}})

            def encode(self, texts):
                import numpy as np
                return np.asarray(run_service(services.embedder.embed(list(texts)), io_loop), dtype=float)

            def batch_encode(self, texts, **kwargs):
                import numpy as np
                if isinstance(texts, str):
                    texts = [texts]
                # Same API-embedding behavior as official OpenAI.py: instruction
                # argument is accepted but API texts are sent without a prefix.
                batch_size = kwargs.get("batch_size", self.global_config.embedding_batch_size)
                result = self.encode(texts) if len(texts) <= batch_size else np.concatenate([
                    self.encode(texts[offset:offset + batch_size])
                    for offset in range(0, len(texts), batch_size)])
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                if (norms == 0).any():
                    raise ValueError("EMem embedding service returned a zero vector")
                return result / norms

        return PublicEmbedding

    def _llm(self, cfg):
        from dataclasses import asdict
        from mematlas.core.errors import ConfigurationError
        from mematlas.core.telemetry import UsageRecord
        module = importlib.import_module("emem.llm.openai_gpt_batch")
        # Upstream's with-connection cache write commits but leaves Windows
        # handles until GC. Preserve the real SQLite statements/transactions,
        # closing only the connection at the native context manager boundary.
        class ClosingConnection(sqlite3.Connection):
            def __exit__(connection, *args):
                try:
                    return super().__exit__(*args)
                finally:
                    connection.close()
        module.sqlite3 = SimpleNamespace(connect=lambda *args, **kwargs:
            sqlite3.connect(*args, factory=ClosingConnection, **kwargs))
        cls = module.CacheOpenAI
        provider = self.providers["llm"]
        key = os.environ.get(provider.api_key_env) or getattr(self.services.llm, "api_key", None)
        if not key:
            raise ConfigurationError(f"EMem native CacheOpenAI requires {provider.api_key_env} for the configured LLM endpoint")
        llm = cls(cache_dir=str(self.directory / "llm_cache"), api_key=key, **{**asdict(cfg), "sdk_max_retries": 0, "max_retries": 1, "max_attempts": 1})
        for name, value in provider.options.items():
            if name not in {"max_tokens", "temperature", "seed", "response_journal", "prompt_cache"}:
                llm.llm_config.generate_params[name] = value
        ledger = self.services.usage
        transport = None
        if self.params.get("json_output_mode", "native") == "json_unicode_v1":
            from .structured import JSONTransport
            transport = JSONTransport(provider.options.get("response_journal") or self.directory / "structured-responses.jsonl")

        def record(response):
            usage = getattr(response, "usage", None)
            ledger.add(UsageRecord(provider="emem_native_openai", model=provider.model or cfg.llm_name,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                usage_known=usage is not None,
                metadata={"native_cache": False,
                          "prompt_tokens_details": getattr(getattr(usage, "prompt_tokens_details", None), "model_dump", lambda: {})()}))
            return response

        def failure(exc):
            ledger.add(UsageRecord(provider="emem_native_openai", model=provider.model or cfg.llm_name,
                usage_known=False, status="failed", metadata={"native_cache": False, "error_type": type(exc).__name__}))

        # Preserve native SDKs, retry/cache/batching; meter actual cache misses.
        for client, asynchronous in ((llm.openai_client, False), (llm.async_openai_client, True)):
            for target, operation in ((client.chat.completions, "create"), (client.beta.chat.completions, "parse")):
                function = getattr(target, operation)
                if asynchronous:
                    async def wrapped(_fn=function, **kwargs):
                        from mematlas.providers.requests import metered_request
                        async def send(params):
                            async def invoke():
                                if self.io_loop is None:
                                    return await _fn(**params)
                                return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_fn(**params), self.io_loop))
                            return await metered_request(invoke, ledger=ledger, provider="emem_native_openai",
                                model=provider.model or cfg.llm_name, kind="llm")
                        try:
                            return await transport.invoke(kwargs, send) if transport else await send(kwargs)
                        except BaseException as exc:
                            self.failures.append(exc)
                            raise
                else:
                    def wrapped(_fn=function, **kwargs):
                        from mematlas.providers.requests import metered_request
                        async def send(params):
                            return await metered_request(lambda: asyncio.to_thread(_fn, **params), ledger=ledger,
                                provider="emem_native_openai", model=provider.model or cfg.llm_name, kind="llm")
                        async def invoke():
                            return await transport.invoke(kwargs, send) if transport else await send(kwargs)
                        try:
                            return run_service(invoke(), self.io_loop)
                        except BaseException as exc:
                            self.failures.append(exc)
                            raise
                setattr(target, operation, wrapped)
        # Upstream cache hit returns original token metadata but costs no new
        # provider tokens. Keep it visible without double-counting that usage.
        native_infer, native_ainfer, native_abatch = llm.infer, llm.ainfer, llm.abatch_infer

        def cache_record(result):
            if result[2]:
                ledger.add(UsageRecord(provider="emem_native_cache", model=provider.model or cfg.llm_name,
                    cost=0, resource_kind="cache", usage_source="cache", metadata={"native_cache": True, "cached_prompt_tokens": result[1].get("prompt_tokens"),
                                      "cached_completion_tokens": result[1].get("completion_tokens")}))
            return result

        def infer(*args, **kwargs):
            try:
                return cache_record(native_infer(*args, **kwargs))
            except BaseException as exc:
                self.failures.append(exc)
                raise

        async def ainfer(*args, **kwargs):
            return await native_ainfer(*args, **kwargs)

        async def abatch_infer(*args, **kwargs):
            result = await native_abatch(*args, **kwargs)
            for item in result:
                if not isinstance(item, tuple) or len(item) != 3 or not isinstance(item[0], str):
                    self.failures.append(RuntimeError("Official EMem batch exhausted provider retries"))
                else:
                    cache_record(item)
            return result

        llm.infer, llm.ainfer = infer, ainfer
        llm.abatch_infer = abatch_infer
        return llm

    def ingest(self, events, episode_ref, part):
        self.failures.clear()
        data = importlib.import_module("emem.utils.conversation_data_utils")
        session_id = self.next_session
        # LoCoMo has two participants; roles are normalized to the hospital
        # clinician and external counterpart. Original roles remain in text.
        turns = [data.Turn(speaker="Doctor" if event.role == "assistant" else "Environment",
                  dia_id=f"D{session_id}:{i + 1}",
                  text=f"[{event.role or event.kind.value}] {content_text(event.content)}")
                 for i, event in enumerate(events)]
        stamp = events[0].timestamp
        date = f"{stamp.strftime('%I:%M %p').lower()} on {stamp.day} {stamp.strftime('%B, %Y')}"
        session = data.Session(session_id=session_id, date_time=date, turns=turns)
        sample = data.LoCoMoSample(sample_id=f"{digest(self.scope)}:{episode_ref}:{part}", qa=[],
            conversation=data.Conversation(speaker_a="Doctor", speaker_b="Environment", sessions={session_id: session}),
            event_summary=data.EventSummary(events={}), observation=data.Observation(observations={}), session_summary={})
        self.engine.index_conversation(sample)
        if self.failures:
            raise RuntimeError("EMem provider failed; native fallback is not a valid benchmark write") from self.failures[-1]
        self.engine.ready_to_retrieve = False
        for key, row in self.engine.session_content_store.get_all_id_to_rows().items():
            if row["content"].session_id == session_id:
                self.sources[key] = {"episode_ref": episode_ref, "event_ids": [e.event_id for e in events]}
        self.next_session += 1

    def _row(self, edu):
        data = dict(vars(edu))
        text = data.get("edu_text", "")
        native_id = importlib.import_module("emem.utils.misc_utils").compute_mdhash_id(text, prefix="edu-")
        event_ids = []
        # Original enriched objects retain source session and turn information.
        session_id = data.get("session_id")
        if str(session_id) in self.sources:
            all_events = self.sources[str(session_id)]["event_ids"]
            turns = data.get("source_turn_ids")
            event_ids = [all_events[index - 1] for index in turns if 0 < index <= len(all_events)] if turns else all_events
        return {"memory_id": native_id, "content": text, "metadata": data,
                "source_event_ids": event_ids}

    def search(self, query, top_k):
        self.failures.clear()
        if not self.engine.edu_embedding_store.get_all_ids():
            return [], []
        results, trace = self.engine.retrieve_structuring_conversation_enhanced_v2(
            [query], num_to_retrieve=top_k, no_query_trace_saving=True, use_batched_reranking=True)
        if self.failures:
            raise RuntimeError("EMem provider failed; native fallback is not a valid benchmark read") from self.failures[-1]
        solution = results[0]
        rows = [self._row(edu) for edu in solution.edus]
        # Native no-PPR branch exposes placeholder scores [1,2]; do not report
        # those as real relevance probabilities. Real PPR scores are preserved.
        if self.params.get("variant", "emem") == "emem_g":
            for row, score in zip(rows, solution.edu_scores):
                row["score"] = float(score)
        # Native diagnostics contain numpy values and tuple-key weight maps.
        # Expose maps as tagged key/value pairs, preserving every key and value;
        # this changes only the JSON envelope after native retrieval completes.
        encoder = importlib.import_module("emem.llm.openai_gpt_batch").cache_json_encoder
        return rows, json.loads(json.dumps(trace_value(trace), default=encoder))

    def detail(self, memory_id):
        store = self.engine.edu_embedding_store
        if memory_id not in store.get_all_ids():
            return []
        text = store.get_row(memory_id)["content"]
        return [{"memory_id": memory_id, "content": text}]

    def export_state(self, snapshots):
        files = {}
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file() or path.suffix in {".lock", ".jsonl"} or path.name.endswith(("-wal", "-shm")):
                continue
            if path.suffix not in {".pickle", ".pkl", ".parquet", ".sqlite"}:
                continue
            if path.suffix == ".sqlite":
                # Logical SQLite export is stable, with native cache strings and
                # schemas retained; transient WAL/checkpoint headers are excluded.
                with closing(sqlite3.connect(path)) as connection:
                    payload = "\n".join(connection.iterdump()).encode("utf-8")
                kind = "sqlite_dump"
            else:
                payload, kind = path.read_bytes(), "bytes"
            files[path.relative_to(self.directory).as_posix()] = {"kind": kind, "blob": put_blob(snapshots, payload)}
        return {"files": files, "next_session": self.next_session, "sources": self.sources,
                "speakers": [getattr(self.engine, "speaker_a", "Doctor"), getattr(self.engine, "speaker_b", "Environment")]}

    @staticmethod
    def validate_state(state):
        for name, row in state["files"].items():
            path = Path(name)
            if path.is_absolute() or path.drive or path.root or ".." in path.parts or row["kind"] not in {"bytes", "sqlite_dump"}:
                raise ValueError("Invalid EMem checkpoint path")
            get_blob(row["blob"])

    def _restore_files(self, state):
        # Only remove files owned by this scope; no recursive directory deletion.
        for path in self.directory.rglob("*"):
            self._owned(path)
            if path.is_file() and path.suffix in {".pickle", ".pkl", ".parquet", ".sqlite"}:
                path.unlink()
        for name, row in state["files"].items():
            path = self.directory / name
            self._owned(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = get_blob(row["blob"])
            if row["kind"] == "sqlite_dump":
                with closing(sqlite3.connect(path)) as connection:
                    connection.executescript(data.decode("utf-8"))
            else:
                path.write_bytes(data)

    def reset(self):
        self.close()
        for path in self.directory.rglob("*"):
            self._owned(path)
            if path.is_file() and path.suffix in {".pickle", ".pkl", ".parquet", ".sqlite", ".jsonl", ".lock"}:
                path.unlink()

    def _owned(self, path):
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("EMem storage path escapes the owned scope directory")

    def close(self):
        llm = self.engine.llm_model
        llm.openai_client.close()
        run_service(llm.async_openai_client.close(), self.io_loop)

