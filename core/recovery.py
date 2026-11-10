"""Preserve recoverability and cause identities across executor boundaries."""
TEMPORARY = {"LLMRequestPaused", "DoctorOutputPaused", "APIConnectionError", "APITimeoutError", "TimeoutError",
             "ConnectionError", "ConnectionResetError", "URLError"}


def error_info(exc):
    chain, seen = [], set()
    pending = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current),
                      "status_code": getattr(current, "status_code", getattr(current, "code",
                          getattr(getattr(current, "response", None), "status_code", None)))})
        remote = getattr(current, "remote_error", None)
        if remote:
            chain.extend(remote.get("chain", []))
        pending.append(current.__cause__ or current.__context__)
        pending.extend(reversed(getattr(current, "exceptions", ())))
    permanent = any(row.get("status_code") in {400, 401, 403, 404, 422} for row in chain)
    recoverable = not permanent and any(row["type"] in TEMPORARY or row.get("status_code") in {408, 429}
        or isinstance(row.get("status_code"), int) and row["status_code"] >= 500 for row in chain)
    return {"type": type(exc).__name__, "message": str(exc), "chain": chain, "recoverable": recoverable,
            "requested_stop": any(row["type"] == "HospitalPaused" for row in chain)}
