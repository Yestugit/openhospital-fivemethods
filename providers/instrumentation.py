"""Account for provider calls even when a custom provider omits usage data."""

from uuid import uuid4

from mematlas.core.telemetry import _ACTIVE_CALL, UsageLedger, UsageRecord, active_usage


class MeteredService:
    def __init__(self, delegate, ledger: UsageLedger, kind: str):
        self.delegate = delegate
        self.ledger = ledger
        self.kind = kind

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def construction_spec(self):
        from mematlas.core.serialization import service_spec
        return service_spec(self.delegate)

    async def _call(self, operation, *args, **kwargs):
        ledger = active_usage()[0] if active_usage() else self.ledger
        call_id = uuid4().hex
        token = _ACTIVE_CALL.set(call_id)
        failed = False
        try:
            return await getattr(self.delegate, operation)(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            if not any(record.call_id == call_id for record in ledger.all()):
                ledger.add(UsageRecord(provider=type(self.delegate).__name__,
                    model=str(getattr(self.delegate, "model", "unknown")),
                    usage_known=False, status="failed" if failed else "success",
                    resource_kind="embedding" if self.kind == "embedder" else "llm",
                    metadata={"kind": self.kind}))
            _ACTIVE_CALL.reset(token)

    async def complete(self, *args, **kwargs):
        return await self._call("complete", *args, **kwargs)

    async def embed(self, *args, **kwargs):
        return await self._call("embed", *args, **kwargs)

    async def encode_image(self, *args, **kwargs):
        return await self._call("encode_image", *args, **kwargs)

    async def embed_text(self, *args, **kwargs):
        return await self._call("embed_text", *args, **kwargs)


def instrument_services(services) -> None:
    for name in ("llm", "embedder"):
        service = getattr(services, name)
        if service is not None and not isinstance(service, MeteredService):
            setattr(services, name, MeteredService(service, services.usage, name))
