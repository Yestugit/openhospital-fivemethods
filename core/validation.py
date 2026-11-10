from __future__ import annotations

from typing import Any, Mapping


_TYPE_MAP = {
    "object": Mapping,
    "array": (list, tuple),
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def _matches_type(value: Any, expected_name: str | list[str] | None) -> bool:
    if isinstance(expected_name, list):
        return any(_matches_type(value, name) for name in expected_name)
    if expected_name == "null":
        return value is None
    expected = _TYPE_MAP.get(expected_name)
    if expected is None:
        return True
    if expected_name in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def validate_parameters(
    values: Mapping[str, Any], schema: Mapping[str, Any], *, path: str = "method.params"
) -> list[str]:
    if not schema:
        return []
    errors = []
    for name in schema.get("required", []):
        if name not in values or values[name] is None:
            errors.append(f"{path}.{name} is required")
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(values) - set(properties))
        if unknown:
            errors.append(f"{path} contains unknown fields: {', '.join(unknown)}")
    for name, value in values.items():
        field_schema = properties.get(name)
        if field_schema is None:
            continue
        expected_name = field_schema.get("type")
        if not _matches_type(value, expected_name):
            errors.append(f"{path}.{name} must be {expected_name}")
            continue
        if "enum" in field_schema and value not in field_schema["enum"]:
            errors.append(f"{path}.{name} must be one of {field_schema['enum']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in field_schema and value < field_schema["minimum"]:
                errors.append(f"{path}.{name} must be >= {field_schema['minimum']}")
            if "maximum" in field_schema and value > field_schema["maximum"]:
                errors.append(f"{path}.{name} must be <= {field_schema['maximum']}")
        if isinstance(value, str):
            if "minLength" in field_schema and len(value) < field_schema["minLength"]:
                errors.append(f"{path}.{name} must contain at least {field_schema['minLength']} characters")
            if "maxLength" in field_schema and len(value) > field_schema["maxLength"]:
                errors.append(f"{path}.{name} must contain at most {field_schema['maxLength']} characters")
        if isinstance(value, (list, tuple)):
            if "minItems" in field_schema and len(value) < field_schema["minItems"]:
                errors.append(f"{path}.{name} must contain at least {field_schema['minItems']} items")
            if "maxItems" in field_schema and len(value) > field_schema["maxItems"]:
                errors.append(f"{path}.{name} must contain at most {field_schema['maxItems']} items")
        if expected_name == "object" and isinstance(value, Mapping):
            errors.extend(validate_parameters(value, field_schema, path=f"{path}.{name}"))
    return errors
