"""Execute unchanged upstream methods in a small AST-only harness.

The harness avoids upstream imports that download embedding models. It checks
decisions with identical model responses and neighbor input, not model quality.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import typing
import uuid

import pytest

from mematlas.core.plugin import MethodServices
from mematlas.methods.a_mem.core import AgenticMemoryCore, MemoryNote
from mematlas.methods.a_mem.prompts import EVOLUTION_PROMPT
from mematlas.test.a_mem.fixtures import FixedVectors


def upstream_class():
    path = Path(__file__).parent / "reference" / "memory_system.py"
    manifest = json.loads(path.with_name("source.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == manifest["sha256"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    note = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MemoryNote")
    system = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgenticMemorySystem")
    system.body = [node for node in system.body if isinstance(node, ast.FunctionDef) and
                   node.name in {"process_memory", "analyze_content", "search_agentic"}]
    namespace = {**vars(typing), "uuid": uuid, "datetime": datetime, "json": json,
                 "logger": logging.getLogger("upstream-reference")}
    exec(compile(ast.Module(body=[note, system], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["AgenticMemorySystem"], namespace["MemoryNote"]


class Reply:
    def __init__(self, response):
        self.response = response

    def get_completion(self, *args, **kwargs):
        return self.response

    async def complete(self, *args, **kwargs):
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize("actions", [[], ["strengthen"], ["update_neighbor"], ["strengthen", "update_neighbor"]])
async def test_evolution_actions_match_upstream_with_identical_neighbor_input(tmp_path, actions):
    native_type, native_note = upstream_class()
    decision = {"should_evolve": bool(actions), "actions": actions, "suggested_connections": ["old"],
                "tags_to_update": ["service", "new"], "new_context_neighborhood": ["Updated context"],
                "new_tags_neighborhood": [["updated"]]}
    reply = Reply(json.dumps(decision))
    old = MemoryNote("Helios port is 8443", id="old", keywords=["Helios"], tags=["old"],
                     context="Original context", embedding=[1.0, 0.0, 0.0])
    new = MemoryNote("Helios port is 9443", id="new", keywords=["port"], tags=["new"], context="Port change")
    core = AgenticMemoryCore(MethodServices(llm=reply, embedder=FixedVectors(), artifact_dir=tmp_path))
    core.memories["old"] = old
    native = native_type.__new__(native_type)
    native.memories = {"old": native_note(content=old.content, id=old.id, context=old.context,
                                           keywords=list(old.keywords), tags=list(old.tags))}
    native.llm_controller = SimpleNamespace(llm=reply)
    native._evolution_system_prompt = EVOLUTION_PROMPT
    # Original process_memory uses this exact input boundary; only external retrieval is controlled.
    native.find_related_memories = lambda query, k: ("Original context: Helios port is 8443", [0])
    native_new = native_note(content=new.content, id=new.id, keywords=list(new.keywords),
                             context=new.context, tags=list(new.tags))
    expected_evolved, expected_new = native.process_memory(native_new)
    actual_evolved = await core._evolve(new, [{**deepcopy(vars(old)), "score": 1.0}])
    assert actual_evolved == expected_evolved
    for actual, expected in ((new, expected_new), (old, native.memories["old"])):
        assert (actual.content, actual.keywords, actual.context, actual.tags, actual.links) == (
            expected.content, expected.keywords, expected.context, expected.tags, expected.links)
    assert old.links == []  # strengthen is directed from the new note, never reciprocal.


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{"keywords":["port"],"context":"Service","tags":["config"]}', "not json"])
async def test_analysis_and_failure_defaults_match_upstream(tmp_path, response):
    native_type, _ = upstream_class()
    reply = Reply(response)
    native = native_type.__new__(native_type)
    native.llm_controller = SimpleNamespace(llm=reply)
    core = AgenticMemoryCore(MethodServices(llm=reply, embedder=FixedVectors(), artifact_dir=tmp_path))
    assert await core._analyze("Helios 8443") == native.analyze_content("Helios 8443")


@pytest.mark.asyncio
async def test_search_order_and_budget_match_upstream_for_the_same_chroma_results(tmp_path):
    native_type, native_note = upstream_class()
    core = AgenticMemoryCore(MethodServices(llm=Reply("not json"), embedder=FixedVectors(), artifact_dir=tmp_path))
    for identity, content, vector in (("old", "unrelated note", [0., 1., 0.]), ("new", "Helios port", [1., 0., 0.])):
        core.memories[identity] = MemoryNote(content, id=identity, embedding=vector, links=["old"] if identity == "new" else [])
        core._upsert_note(core.memories[identity])
    native = native_type.__new__(native_type)
    native.memories = {key: native_note(content=value.content, id=key, links=list(value.links))
                       for key, value in core.memories.items()}
    class Retriever:
        def search(self, query, k):
            raw = core._collection.query(query_embeddings=[[1., 0., 0.]], n_results=min(k, 2),
                                         include=["distances"])
            raw["metadatas"] = [[{**vars(native.memories[key])} for key in raw["ids"][0]]]
            return raw
    native.retriever = Retriever()
    for k in (1, 2, 3):
        actual = await core.search_agentic("Helios", k=k)
        expected = native.search_agentic("Helios", k=k)
        assert [item["id"] for item in actual] == [item["id"] for item in expected]
        assert [item["content"] for item in actual] == [item["content"] for item in expected]
    assert core.memories["new"].retrieval_count == 0
