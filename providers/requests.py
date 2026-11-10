"""Capture every provider attempt before application-level parsing."""
import asyncio
from collections.abc import Mapping

from mematlas.core.telemetry import UsageRecord


class LLMRequestPaused(RuntimeError):
    """The provider is temporarily unavailable; preserve resumable experiment state."""


async def metered_request(request, *, ledger, provider, model, kind, retries=7):
    from time import perf_counter
    from uuid import uuid4
    from mematlas.core.recovery import error_info
    request_id = uuid4().hex
    for attempt in range(retries + 1):
        started = perf_counter()
        try:
            response = await request()
        except BaseException as exc:
            if ledger is not None:
                ledger.add(UsageRecord(provider, model, usage_known=False, resource_kind=kind,
                                       status="failed", metadata={"attempt": attempt + 1, "error_type": type(exc).__name__, "request_id": request_id, "elapsed_s": perf_counter() - started, "status_code": getattr(exc, "status_code", None)}))
            if not error_info(exc)["recoverable"]:
                raise
            if attempt == retries:
                raise LLMRequestPaused(f"{kind} API unavailable after {attempt + 1} attempts ({type(exc).__name__})") from exc
            headers = getattr(getattr(exc, "response", None), "headers", {})
            try:
                delay = max(min(2 ** (attempt + 1), 30), min(120, float(headers.get("Retry-After", 0))))
            except (TypeError, ValueError):
                delay = min(2 ** (attempt + 1), 30)
            await asyncio.sleep(delay)
            continue
        usage = getattr(response, "usage", None)
        known = usage is not None
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None and kind == "embedding":
            prompt = getattr(usage, "total_tokens", None)
        completion = getattr(usage, "completion_tokens", 0) if kind == "embedding" else getattr(usage, "completion_tokens", None)
        known = known and isinstance(prompt, int) and isinstance(completion, int)
        if ledger is not None:
            metadata = {"attempt": attempt + 1, "request_id": request_id, "elapsed_s": perf_counter() - started}
            if kind == "embedding" and isinstance(getattr(response, "cache", None), Mapping):
                metadata["local_embedding_cache"] = dict(response.cache)
            details = getattr(usage, "prompt_tokens_details", None)
            if hasattr(details, "model_dump"):
                metadata["prompt_tokens_details"] = details.model_dump(exclude_none=True)
            elif isinstance(details, Mapping):
                metadata["prompt_tokens_details"] = dict(details)
            ledger.add(UsageRecord(provider, model, prompt_tokens=prompt or 0, completion_tokens=completion or 0,
                                   usage_known=known, resource_kind=kind, metadata=metadata))
        return response
