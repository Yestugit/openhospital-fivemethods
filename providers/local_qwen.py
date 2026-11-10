"""Download and serve the official Qwen3 embedding model on loopback."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Lock
from collections import OrderedDict
from time import monotonic
from urllib.request import urlopen

from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import file_sha256, json_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_DIR = ROOT / "models/Qwen3-Embedding-0.6B"
BASE_URL = "http://127.0.0.1:8766/v1"
FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "special_tokens_map.json", "modules.json",
         "sentence_bert_config.json", "config_sentence_transformers.json", "1_Pooling/config.json",
         "README.md", "LICENSE")


def download_model(directory=MODEL_DIR):
    """Pin one HF revision and download only the Transformers inference files."""
    from huggingface_hub import HfApi, snapshot_download
    directory = Path(directory).expanduser().resolve()
    info = HfApi().model_info(MODEL_ID, token=False)
    revision = info.sha
    snapshot_download(MODEL_ID, revision=revision, local_dir=directory,
        allow_patterns=list(FILES), token=False, max_workers=2)
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    if any(not (directory / name).is_file() for name in required):
        raise ValueError("Downloaded model is missing required inference files")
    hashes = {str(path.relative_to(directory)).replace("\\", "/"): file_sha256(path)
              for path in directory.rglob("*") if path.is_file()
              and ".cache" not in path.relative_to(directory).parts and path.name != "deployment.json"}
    manifest = {"model_id": MODEL_ID, "revision": revision, "files": hashes,
                "dimensions": 1024, "pooling": "last_non_padding_token", "normalization": "l2",
                "instruction": None, "max_tokens": 32768}
    manifest["fingerprint"] = json_sha256(manifest)
    atomic_write_json(directory / "deployment.json", manifest)
    print(json.dumps({"model_dir": str(directory), "revision": revision,
          "size_bytes": sum((directory / name).stat().st_size for name in hashes),
          "fingerprint": manifest["fingerprint"]}, indent=2), flush=True)
    return manifest


def model_spec(directory=MODEL_DIR, *, verify=False):
    directory = Path(directory).expanduser().resolve()
    path = directory / "deployment.json"
    if not path.is_file():
        raise ValueError(f"Local model is not deployed: {directory}. Run scripts/local-qwen-embedding.py download")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    fingerprint = payload.pop("fingerprint", None)
    if json_sha256(payload) != fingerprint or manifest.get("model_id") != MODEL_ID or manifest.get("dimensions") != 1024:
        raise ValueError("Invalid local embedding deployment manifest")
    for name, digest in manifest["files"].items():
        source = (directory / name).resolve()
        if not source.is_relative_to(directory) or not source.is_file():
            raise ValueError(f"Missing or invalid local embedding file: {name}")
        if verify and file_sha256(source) != digest:
            raise ValueError(f"Local embedding file checksum changed: {name}")
    return {"model_dir": str(directory), "model_id": MODEL_ID, "revision": manifest["revision"],
            "fingerprint": fingerprint, "dimensions": 1024, "pooling": manifest["pooling"],
            "normalization": manifest["normalization"], "instruction": manifest["instruction"],
            "max_tokens": manifest["max_tokens"]}


def efficient_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Use the exact SDPA operator without its memory-heavy GQA math fallback.

    Expand grouped KV heads to avoid the quadratic GQA math fallback. CPU
    PyTorch provides a fused flash kernel for float32; Windows CUDA uses the
    memory-efficient kernel. Neither path truncates or chunks the input text.
    """
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.integrations.sdpa_attention import repeat_kv
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    backend = SDPBackend.EFFICIENT_ATTENTION if query.device.type == "cuda" else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(backend):
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value,
            attn_mask=attention_mask, dropout_p=dropout, scale=scaling,
            is_causal=attention_mask is None and query.shape[2] > 1)
    return result.transpose(1, 2).contiguous(), None


class QwenEmbeddingEngine:
    def __init__(self, directory=MODEL_DIR, *, device="auto"):
        self.spec = model_spec(directory, verify=True)
        self._exact_cache = OrderedDict()
        self._exact_lock = Lock()
        self.cache_hits = self.inferred_texts = self.inferred_tokens = self.saved_tokens = 0
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; install CUDA PyTorch or choose cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS
        attention = "mematlas_efficient_sdpa"
        AttentionInterface.register(attention, efficient_attention)
        AttentionMaskInterface.register(attention, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                       trust_remote_code=False, padding_side="left")
        self.model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
            dtype=self.dtype, attn_implementation=attention, low_cpu_mem_usage=True)
        self.model.to(self.device).eval()
        self.lock = Lock()
        self.spec.update(device=self.device, dtype=str(self.dtype), attention=attention, use_cache=False,
                         torch_version=torch.__version__, transformers_version=transformers.__version__,
                         implementation_sha256=file_sha256(__file__))

    def embed(self, texts):
        with self._exact_lock:
            return self._cached_embed(texts)

    def _cached_embed(self, texts):
        # Keep inference sequential on the small GPU. The existing implementation
        # holds its own model lock; this cache is accessed in that same service.
        vectors, tokens = [], 0
        for text in texts:
            key = (self.spec["fingerprint"], text)
            if key in self._exact_cache:
                vector, count = self._exact_cache.pop(key)
                self.cache_hits += 1
                self.saved_tokens += count
            else:
                values, count = self._infer([text])
                vector = values[0]
                self.inferred_texts += 1
                self.inferred_tokens += count
            self._exact_cache[key] = (vector, count)
            while len(self._exact_cache) > 2048:
                self._exact_cache.popitem(last=False)
            vectors.append(list(vector))
            tokens += count
        return vectors, tokens

    def _infer(self, texts):
        torch = self.torch
        vectors, total = [], 0
        with self.lock, torch.inference_mode():
            encoded = []
            for text in texts:
                tokens = self.tokenizer(text, return_tensors="pt", truncation=False)
                count = tokens["input_ids"].shape[1]
                if count > self.spec["max_tokens"]:
                    raise ValueError(f"Input has {count} tokens; local model limit is {self.spec['max_tokens']}. Input was not truncated.")
                if not count:
                    raise ValueError("Embedding text cannot be empty")
                encoded.append(tokens)
                total += count
            # Batch size one keeps three RSI workers within the laptop GPU budget.
            for tokens in encoded:
                inputs = {key: value.to(self.device) for key, value in tokens.items()}
                try:
                    outputs = self.model(**inputs, use_cache=False)
                    # The model card uses last-token pooling with left padding.
                    vector = outputs.last_hidden_state[:, -1].float()
                    vector = torch.nn.functional.normalize(vector, p=2, dim=1)
                    if not torch.isfinite(vector).all():
                        raise RuntimeError("Local embedding produced nonfinite values")
                    vectors.append(vector[0].cpu().tolist())
                    del outputs, vector, inputs
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    raise RuntimeError("Local embedding GPU memory exhausted. Free GPU memory or restart with --device cpu; input was not truncated.") from None
        return vectors, total


def create_app(engine):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    app = FastAPI(title="Local Qwen3 Embedding")

    @app.get("/health")
    def health():
        return {"status": "ready", **engine.spec, "exact_cache_hits": engine.cache_hits, "inferred_texts": engine.inferred_texts,
            "inferred_tokens": engine.inferred_tokens, "saved_tokens": engine.saved_tokens}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        if request.client is None or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse({"error": {"message": "Loopback clients only"}}, status_code=403)
        try:
            body = await request.json()
            if not isinstance(body, dict) or body.get("model") != MODEL_ID:
                raise ValueError(f"model must be {MODEL_ID}")
            if body.get("dimensions", 1024) != 1024 or body.get("encoding_format", "float") not in {"float", "base64"}:
                raise ValueError("Only 1024 dimensions and float/base64 output are supported")
            texts = body.get("input")
            texts = [texts] if isinstance(texts, str) else texts
            if not isinstance(texts, list) or not 1 <= len(texts) <= 32 or any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError("input must contain 1 to 32 nonempty strings")
            vectors, total = await asyncio.to_thread(engine.embed, texts)
            if body.get("encoding_format") == "base64":
                import base64
                import struct
                values = [base64.b64encode(struct.pack("<1024f", *vector)).decode("ascii") for vector in vectors]
            else:
                values = vectors
            return {"object": "list", "model": MODEL_ID,
                "data": [{"object": "embedding", "index": index, "embedding": vector}
                         for index, vector in enumerate(values)],
                "usage": {"prompt_tokens": total, "total_tokens": total},
                "cache": {"exact_cache_hits": engine.cache_hits, "inferred_texts": engine.inferred_texts,
            "inferred_tokens": engine.inferred_tokens, "saved_tokens": engine.saved_tokens}}
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error"}}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"error": {"message": str(exc), "type": "local_inference_error"}}, status_code=503)
    return app


class LocalEmbeddingService:
    """One owned GPU model shared by all suite workers; never adopts occupied ports."""
    def __init__(self, spec, *, log_path, device="auto", port=8766, python_executable=None):
        self.spec, self.log_path, self.device = spec, Path(log_path), device
        self.port, self.python_executable = port, python_executable or sys.executable
        self.process = None
        self.log = None
        self.health = None

    async def __aenter__(self):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", self.port))
            except OSError as exc:
                raise RuntimeError(f"Local embedding port {self.port} is occupied; choose another port") from exc
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("ab")
        try:
            self.process = await asyncio.create_subprocess_exec(self.python_executable, "-X", "utf8",
                str(ROOT / "scripts/local-qwen-embedding.py"), "serve", "--model-dir", self.spec["model_dir"],
                "--device", self.device, "--port", str(self.port), stdout=self.log, stderr=self.log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            deadline = monotonic() + 180
            while monotonic() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError(f"Local embedding service exited; see {self.log_path}")
                try:
                    def health():
                        with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                            return json.loads(response.read().decode("utf-8"))
                    info = await asyncio.to_thread(health)
                    if info.get("fingerprint") != self.spec["fingerprint"]:
                        raise ValueError("Local embedding model fingerprint differs from the experiment")
                    self.health = info
                    return self
                except (OSError, json.JSONDecodeError):
                    await asyncio.sleep(0.25)
            raise RuntimeError(f"Local embedding service startup timed out; see {self.log_path}")
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *exc):
        from mematlas.integrations.openhospital.services import terminate_owned_process
        if self.process is not None:
            await terminate_owned_process(self.process)
        if self.log is not None:
            self.log.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "serve", "verify"))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if args.command == "download":
        download_model(args.model_dir)
    elif args.command == "verify":
        print(json.dumps(model_spec(args.model_dir, verify=True), indent=2))
    else:
        import uvicorn
        engine = QwenEmbeddingEngine(args.model_dir, device=args.device)
        print(json.dumps(engine.spec, indent=2), flush=True)
        uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, access_log=False)
    return 0
