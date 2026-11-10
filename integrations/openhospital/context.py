"""Bound observable doctor input while retaining its beginning and end."""


def bounded_text(value: str, limit: int) -> str:
    if limit < 64:
        raise ValueError("Observable input budget must be at least 64 characters")
    if len(value) <= limit:
        return value
    marker = "\n[中间内容已裁剪]\n"
    remaining = limit - len(marker)
    head = remaining // 2
    return value[:head] + marker + value[-(remaining - head):]
