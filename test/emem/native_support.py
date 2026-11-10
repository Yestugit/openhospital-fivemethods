"""Serializable controlled transport for pinned official EMem tests."""


class FixedEmbedding:
    def __init__(self, deny=False):
        self.calls = []
        self.deny = deny

    def construction_spec(self):
        return {"factory": "mematlas.test.emem.native_support:FixedEmbedding", "kwargs": {"deny": self.deny}}

    async def embed(self, texts):
        if self.deny:
            raise AssertionError("Embedding called during restore")
        self.calls.append(list(texts))
        return [[1.0, 0.1, 0.2, 0.3] for _ in texts]
