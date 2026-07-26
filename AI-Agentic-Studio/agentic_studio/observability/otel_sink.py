"""Optional OpenTelemetry span forwarding.

Activated by setting OTEL_EXPORTER_OTLP_ENDPOINT and installing the
`tracing` extra. Falls back to a no-op when the SDK is absent.
"""

from __future__ import annotations

from typing import Any

_TRACER = None
_INITIALISED = False


def _init() -> Any:
    global _TRACER, _INITIALISED
    if _INITIALISED:
        return _TRACER
    _INITIALISED = True
    try:  # pragma: no cover - optional dependency
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "ai-agentic-studio"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("agentic_studio")
    except Exception:
        _TRACER = None
    return _TRACER


def send(payload: dict[str, Any]) -> bool:
    tracer = _init()
    if tracer is None:
        return False
    try:  # pragma: no cover - network path
        with tracer.start_as_current_span(payload["name"]) as otel_span:
            for key, value in (payload.get("attributes") or {}).items():
                otel_span.set_attribute(f"studio.{key}", str(value))
            otel_span.set_attribute("studio.duration_ms", payload.get("duration_ms", 0))
            otel_span.set_attribute("studio.status", payload.get("status", "ok"))
        return True
    except Exception:
        return False
