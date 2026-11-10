"""Serializable provider responses; real Memora algorithms and Chroma run."""
from copy import deepcopy
import json
import re


class ScriptedLLM:
    def __init__(self, deny=False):
        self.calls = []
        self.policy = 0
        self.fact = 0
        self.deny = deny

    def construction_spec(self):
        return {"factory": "mematlas.test.memora.native_support:ScriptedLLM", "kwargs": {"deny": self.deny}}

    async def complete(self, messages, **kwargs):
        if self.deny:
            raise AssertionError("LLM called during state restoration")
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        schema = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
        if schema == "SegmentationOutput":
            conversation = messages[0]["content"].split("Segment the following conversation:")[-1]
            indices = [int(index) for index in re.findall(r"^(\d+)\.\s", conversation, re.MULTILINE)]
            result = {"episodes": [{"topic": "Clinical observations", "indices": indices}]}
        elif schema == "MemoryOutputs":
            self.fact += 1
            result = {"entries": [{"memory_type": "Factual", "index": f"Patient allergy {self.fact}",
                                    "value": "Patient reports a penicillin allergy."}]}
        elif schema == "BatchCueIndices":
            indices = re.findall(r"Primary Index: ([^\n]+)", messages[0]["content"])
            result = {"results": [{"memory_index": index, "cue_indices": ["Patient medication safety"]}
                                    for index in indices]}
        elif schema == "MemoryUpdateDecision":
            result = {"should_update": False}
        elif schema == "CueIndices":
            result = {"cue_indices": ["Patient medication safety"]}
        else:
            self.policy += 1
            result = {"action": "RE_QUERY" if self.policy % 2 else "STOP", "reason": "fixture transport response",
                      "new_query": "penicillin allergy", "confidence": 1.0}
        return json.dumps(result)


class FixedEmbedding:
    def __init__(self, deny=False):
        self.calls = []
        self.deny = deny

    def construction_spec(self):
        return {"factory": "mematlas.test.memora.native_support:FixedEmbedding", "kwargs": {"deny": self.deny}}

    async def embed(self, texts):
        if self.deny:
            raise AssertionError("Embedding called during restoration")
        self.calls.append(list(texts))
        return [[1.0, 0.1, 0.2, 0.3] for _ in texts]
