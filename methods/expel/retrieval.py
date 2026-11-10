"""FAISS documents for native ExpeL retrieval; queries never write state."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .utils import environment_name, strip_task, trajectory_fields


def vectors_array(vectors, count: int, dimensions: int | None = None) -> np.ndarray:
    if len(vectors) != count or any(not row or any(isinstance(value, bool) or not isinstance(value, Real)
        for value in row) for row in vectors):
        raise ValueError('ExpeL Embedding requires one nonempty numeric vector per text')
    lengths = {len(row) for row in vectors}
    if len(lengths) != 1 or dimensions is not None and lengths != {dimensions}:
        raise ValueError('ExpeL Embedding dimensions do not match the saved document index')
    result = np.asarray(vectors, dtype='float32')
    if not np.isfinite(result).all():
        raise ValueError('ExpeL Embedding vectors must be finite')
    return result


class ExperienceIndex:
    def __init__(self, services, benchmark: str, path: Path):
        import faiss
        self.faiss, self.services, self.benchmark, self.path = faiss, services, benchmark, path
        self.documents: list[dict[str, Any]] = []

    def trial_documents(self, trial: dict, *, include_task: bool) -> list[dict]:
        """One native component per key; reused to validate complete snapshot lineage."""
        fields = trajectory_fields(trial['content'], self.benchmark, trial['reflections'])
        if include_task:
            fields = {'task': [strip_task(trial['task'], self.benchmark)], **fields}
        documents = [dict(type=kind, content=value, task=trial['task'], trial_id=trial['id'],
                          env_name=trial['env_name']) for kind, values in fields.items() for value in values]
        for offset, doc in enumerate(documents):
            doc['id'] = hashlib.sha256(json.dumps([trial['id'], offset, doc['type'], doc['content']]).encode()).hexdigest()
        return documents

    async def add_success(self, trial: dict) -> None:
        documents = self.trial_documents(trial, include_task=not any(
            doc['type'] == 'task' and doc['task'] == trial['task'] for doc in self.documents))
        if not documents:
            return
        vectors = vectors_array(await self.services.embedder.embed([d['content'] for d in documents]),
                                len(documents), len(self.documents[0]['embedding']) if self.documents else None)
        for doc, vector in zip(documents, vectors):
            doc['embedding'] = vector.tolist()
        self.documents.extend(documents)

    def persist(self) -> None:
        if not self.documents:
            self.path.unlink(missing_ok=True)
            return
        values = vectors_array([doc['embedding'] for doc in self.documents], len(self.documents))
        index = self.faiss.IndexFlatL2(values.shape[1])
        index.add(values)
        self.faiss.write_index(index, str(self.path))

    def restore(self, documents) -> None:
        documents = deepcopy(documents)
        if documents:
            vectors_array([doc['embedding'] for doc in documents], len(documents))
        self.documents = documents
        self.persist()

    async def search(self, query: str, field: str, env_name: str | None, tasks: dict, limit: int):
        docs = [doc for doc in self.documents if doc['type'] == field and doc['env_name'] == env_name]
        if self.benchmark == 'webshop':
            docs = [d for d in docs if 'Observation: Invalid action!' not in tasks[d['task']][0]['content']
                and 'think[]' not in tasks[d['task']][0]['content']
                and len(tasks[d['task']][0]['content'].split('Observation: You have clicked')) >= 3]
        if not docs or limit < 1:
            return []
        values = vectors_array([doc['embedding'] for doc in docs], len(docs))
        query_vector = vectors_array(await self.services.embedder.embed([query]), 1, values.shape[1])
        index = self.faiss.IndexFlatL2(values.shape[1])
        index.add(values)
        distances, indices = index.search(query_vector, min(limit, len(docs)))
        return [(docs[int(i)], float(distance)) for i, distance in zip(indices[0], distances[0])]

    async def rerank(self, candidates, query: str, field: str = 'thought'):
        candidate_pairs = {(doc['task'], doc['env_name']) for doc, _ in candidates}
        docs = [doc for doc in self.documents if (doc['task'],doc['env_name']) in candidate_pairs and doc['type'] == field]
        if not docs:
            return []
        values = vectors_array([doc['embedding'] for doc in docs], len(docs))
        vector = vectors_array(await self.services.embedder.embed([query]), 1, values.shape[1])[0]
        # Source reranking is cosine; primary FAISS retrieval remains unnormalized L2.
        norm = np.linalg.norm(values, axis=1) * np.linalg.norm(vector)
        if np.any(norm == 0):
            raise ValueError('ExpeL cosine reranking requires nonzero Embedding vectors')
        scores = 1.0 - values @ vector / norm
        return sorted(zip(docs, map(float, scores)), key=lambda item: item[1])
