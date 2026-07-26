"""Exception hierarchy shared by every layer of the studio."""

from __future__ import annotations


class StudioError(Exception):
    """Base class for all studio errors."""


class ConfigError(StudioError):
    """Configuration is missing or invalid."""


class ProviderError(StudioError):
    """An LLM provider failed. The router treats this as retryable."""

    def __init__(self, provider: str, message: str, retryable: bool = True):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable


class AllProvidersFailed(StudioError):
    """Every provider in the routing chain failed."""

    def __init__(self, failures: dict[str, str]):
        detail = "; ".join(f"{name}: {err}" for name, err in failures.items())
        super().__init__(f"all providers failed -> {detail}")
        self.failures = failures


class ToolError(StudioError):
    """A tool raised, timed out, or was rejected."""

    def __init__(self, tool: str, message: str):
        super().__init__(f"tool '{tool}' failed: {message}")
        self.tool = tool


class ToolNotFound(ToolError):
    def __init__(self, tool: str):
        super().__init__(tool, "not registered or not allowed")


class GuardrailBlocked(StudioError):
    """A guardrail refused the input or output."""

    def __init__(self, rule: str, reason: str):
        super().__init__(f"blocked by '{rule}': {reason}")
        self.rule = rule
        self.reason = reason


class ApprovalRequired(StudioError):
    """A human-in-the-loop checkpoint was reached."""

    def __init__(self, request_id: str, tool: str, arguments: dict):
        super().__init__(f"approval required for '{tool}' (request {request_id})")
        self.request_id = request_id
        self.tool = tool
        self.arguments = arguments


class RetrievalError(StudioError):
    """The retrieval stack could not answer."""


class EvaluationError(StudioError):
    """The evaluation harness could not produce a score."""
