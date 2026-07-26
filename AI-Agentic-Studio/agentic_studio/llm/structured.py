"""Structured output: get typed objects out of a language model.

Accepts a Pydantic model, a dataclass, or a raw JSON schema. Invalid output is
fed back to the model with the validation error attached, up to `retries` times.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, TypeVar

from agentic_studio.core.errors import StudioError
from agentic_studio.core.types import Message
from agentic_studio.observability.tracing import get_tracer

T = TypeVar("T")

_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class StructuredOutputError(StudioError):
    """The model could not produce output matching the schema."""


def json_schema_of(target: Any) -> dict[str, Any]:
    """Build a JSON schema from a Pydantic model, a dataclass, or pass one through."""
    if isinstance(target, dict):
        return target

    model_json_schema = getattr(target, "model_json_schema", None)
    if callable(model_json_schema):  # pydantic v2
        return model_json_schema()
    legacy_schema = getattr(target, "schema", None)
    if callable(legacy_schema) and hasattr(target, "__fields__"):  # pydantic v1
        return legacy_schema()

    if dataclasses.is_dataclass(target):
        properties: dict[str, Any] = {}
        required: list[str] = []
        for field in dataclasses.fields(target):
            properties[field.name] = {"type": _PY_TO_JSON.get(field.type, "string")} \
                if not isinstance(field.type, str) else {"type": _from_annotation(field.type)}
            if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:  # type: ignore[misc]
                required.append(field.name)
        return {"type": "object", "properties": properties, "required": required}

    raise StructuredOutputError(f"cannot derive a JSON schema from {target!r}")


def _from_annotation(annotation: str) -> str:
    lowered = annotation.lower()
    for needle, kind in (("str", "string"), ("bool", "boolean"), ("int", "integer"),
                         ("float", "number"), ("list", "array"), ("dict", "object")):
        if needle in lowered:
            return kind
    return "string"


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of arbitrary model prose."""
    candidate = text.strip()
    if not candidate:
        raise StructuredOutputError("model returned empty output")

    fenced = _FENCED.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth = 0
        for index in range(start, len(candidate)):
            char = candidate[index]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : index + 1])
                    except json.JSONDecodeError:
                        break
    raise StructuredOutputError(f"no valid JSON found in output: {text[:200]!r}")


def validate(data: Any, target: Any) -> Any:
    """Coerce parsed JSON into the requested type."""
    if isinstance(target, dict):
        _check_schema(data, target)
        return data

    validator = getattr(target, "model_validate", None)
    if callable(validator):  # pydantic v2
        return validator(data)
    legacy = getattr(target, "parse_obj", None)
    if callable(legacy):  # pydantic v1
        return legacy(data)

    if dataclasses.is_dataclass(target):
        names = {f.name for f in dataclasses.fields(target)}
        return target(**{k: v for k, v in data.items() if k in names})

    return data


def _check_schema(data: Any, schema: dict[str, Any]) -> None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(data, dict):
            raise StructuredOutputError(f"expected an object, got {type(data).__name__}")
        missing = [key for key in schema.get("required", []) if key not in data]
        if missing:
            raise StructuredOutputError(f"missing required field(s): {', '.join(missing)}")
    elif expected == "array" and not isinstance(data, list):
        raise StructuredOutputError(f"expected an array, got {type(data).__name__}")


def generate_structured(
    prompt: str | list[Message],
    target: Any,
    router: Any = None,
    system: str | None = None,
    retries: int = 2,
    **kwargs: Any,
) -> Any:
    """Ask the model for JSON matching `target` and return a validated object."""
    from agentic_studio.llm.router import get_router

    router = router or get_router()
    schema = json_schema_of(target)

    base: list[Message] = []
    if system:
        base.append(Message.system(system))
    base.append(Message.system(_instruction(schema)))
    if isinstance(prompt, str):
        base.append(Message.user(prompt))
    else:
        base.extend(prompt)

    conversation = list(base)
    last_error: Exception | None = None

    with get_tracer().span("llm.structured", kind="llm", schema_type=schema.get("type")) as span:
        for attempt in range(retries + 1):
            response = router.generate(conversation, use_cache=attempt == 0, **kwargs)
            try:
                parsed = validate(extract_json(response.text), target)
                span.set(attempts=attempt + 1)
                return parsed
            except (StructuredOutputError, Exception) as exc:  # noqa: B014
                last_error = exc
                conversation = list(base) + [
                    Message.assistant(response.text),
                    Message.user(
                        f"That was not valid. Error: {exc}. "
                        "Return ONLY valid JSON matching the schema, with no prose or code fences."
                    ),
                ]
        span.set(attempts=retries + 1, failed=True)

    raise StructuredOutputError(f"failed after {retries + 1} attempts: {last_error}")


def _instruction(schema: dict[str, Any]) -> str:
    return (
        "You must reply with a single JSON value and nothing else. "
        "No prose, no explanation, no markdown fences.\n"
        "It must validate against this schema:\n"
        f"```json-schema\n{json.dumps(schema, indent=2)}\n```"
    )
