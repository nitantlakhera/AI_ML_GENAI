"""The guardrail policy applied at every trust boundary.

Boundaries covered:
  input   - what the user sends
  output  - what the model returns
  tool    - which tool an agent may call, with which arguments
  context - retrieved or fetched text before it enters a prompt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_studio.core.errors import GuardrailBlocked
from agentic_studio.guardrails.moderation import moderate, sanitize_retrieved
from agentic_studio.guardrails.pii import redact_pii
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.metrics import METRICS
from agentic_studio.settings import get_settings

logger = get_logger("guardrails.policy")


@dataclass
class GuardrailVerdict:
    allowed: bool = True
    text: str = ""
    notes: list[str] = field(default_factory=list)
    rule: str = ""

    def raise_if_blocked(self) -> str:
        if not self.allowed:
            raise GuardrailBlocked(self.rule or "policy", "; ".join(self.notes))
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "notes": self.notes, "rule": self.rule}


class GuardrailPolicy:
    def __init__(
        self,
        enabled: bool | None = None,
        pii_mode: str | None = None,
        moderation_mode: str | None = None,
        max_input_chars: int | None = None,
        blocked_tools: set[str] | None = None,
    ):
        settings = get_settings().guardrails
        self.enabled = settings.enabled if enabled is None else enabled
        self.pii_mode = (pii_mode or settings.pii_mode).lower()
        self.moderation_mode = (moderation_mode or settings.moderation_mode).lower()
        self.max_input_chars = max_input_chars or settings.max_input_chars
        self.blocked_tools = blocked_tools or set()

    # -- boundaries ---------------------------------------------------------

    def check_input(self, text: str) -> GuardrailVerdict:
        if not self.enabled:
            return GuardrailVerdict(allowed=True, text=text)

        notes: list[str] = []
        if len(text) > self.max_input_chars:
            METRICS.incr("guardrail_blocked", rule="input_too_long")
            return GuardrailVerdict(
                allowed=False,
                text=text,
                notes=[f"input is {len(text)} chars, limit is {self.max_input_chars}"],
                rule="input_too_long",
            )

        verdict = moderate(text)
        if verdict.flagged and self.moderation_mode == "block":
            METRICS.incr("guardrail_blocked", rule="moderation")
            return GuardrailVerdict(allowed=False, text=text, notes=[verdict.reason], rule="moderation")
        if verdict.flagged:
            notes.append(f"moderation warning: {verdict.reason}")
        if verdict.injection:
            notes.append("prompt-injection pattern detected in input")
            METRICS.incr("guardrail_injection_detected")

        cleaned = text
        if self.pii_mode in {"redact", "block"}:
            cleaned, matches = redact_pii(text)
            if matches:
                kinds = sorted({m.kind for m in matches})
                METRICS.incr("guardrail_pii_found", len(matches))
                if self.pii_mode == "block":
                    return GuardrailVerdict(
                        allowed=False, text=text, notes=[f"input contains PII: {', '.join(kinds)}"],
                        rule="pii",
                    )
                notes.append(f"redacted PII in input: {', '.join(kinds)}")

        return GuardrailVerdict(allowed=True, text=cleaned, notes=notes)

    def check_output(self, text: str) -> GuardrailVerdict:
        if not self.enabled:
            return GuardrailVerdict(allowed=True, text=text)

        notes: list[str] = []
        verdict = moderate(text, check_injection=False)
        if verdict.flagged and self.moderation_mode == "block":
            METRICS.incr("guardrail_blocked", rule="output_moderation")
            return GuardrailVerdict(
                allowed=False,
                text="I cannot provide that.",
                notes=[verdict.reason],
                rule="output_moderation",
            )

        cleaned = text
        if self.pii_mode == "redact":
            cleaned, matches = redact_pii(text)
            if matches:
                notes.append(f"redacted PII in output: {', '.join(sorted({m.kind for m in matches}))}")
        return GuardrailVerdict(allowed=True, text=cleaned, notes=notes)

    def check_tool(self, name: str, arguments: dict[str, Any], allowed: set[str] | None = None) -> GuardrailVerdict:
        if not self.enabled:
            return GuardrailVerdict(allowed=True, text=name)

        if name in self.blocked_tools:
            METRICS.incr("guardrail_blocked", rule="tool_blocked")
            return GuardrailVerdict(allowed=False, text=name, notes=[f"tool '{name}' is blocked"],
                                    rule="tool_blocked")
        if allowed and name not in allowed:
            METRICS.incr("guardrail_blocked", rule="tool_not_allowed")
            return GuardrailVerdict(
                allowed=False, text=name, notes=[f"tool '{name}' is not in the allowlist"],
                rule="tool_not_allowed",
            )

        rendered = " ".join(str(value) for value in arguments.values())
        verdict = moderate(rendered)
        if verdict.flagged and self.moderation_mode == "block":
            METRICS.incr("guardrail_blocked", rule="tool_arguments")
            return GuardrailVerdict(allowed=False, text=name, notes=[verdict.reason],
                                    rule="tool_arguments")
        return GuardrailVerdict(allowed=True, text=name)

    def clean_context(self, text: str) -> str:
        """Defang instruction-like content in retrieved documents."""
        if not self.enabled:
            return text
        return sanitize_retrieved(text)


_POLICY: GuardrailPolicy | None = None


def get_policy() -> GuardrailPolicy:
    global _POLICY
    if _POLICY is None:
        _POLICY = GuardrailPolicy()
    return _POLICY


def set_policy(policy: GuardrailPolicy) -> None:
    global _POLICY
    _POLICY = policy


def reset_policy() -> None:
    global _POLICY
    _POLICY = None
