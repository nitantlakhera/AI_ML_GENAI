"""Content moderation and prompt-injection detection.

A fast local pass runs always; a hosted moderation endpoint can be layered on
top when an OpenAI key is present. Injection detection matters most for agents,
where retrieved or fetched text can try to hijack the tool loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "self_harm": (
        re.compile(r"\b(kill|hurt|harm)\s+(myself|yourself)\b", re.I),
        re.compile(r"\bsuicide\s+(method|instructions|how)\b", re.I),
    ),
    "weapons": (
        re.compile(r"\bhow to (?:build|make|assemble)\s+a?\s*(bomb|explosive|firearm|silencer)\b", re.I),
        re.compile(r"\b(pipe bomb|nerve agent|sarin|ricin)\b", re.I),
    ),
    "malware": (
        re.compile(r"\b(write|generate|create)\s+(?:a\s+)?(ransomware|keylogger|botnet|rootkit)\b", re.I),
        re.compile(r"\bexploit\s+(?:the\s+)?(?:cve|zero[- ]day)\b", re.I),
    ),
    "illicit": (
        re.compile(r"\bhow to (?:synthesize|cook|manufacture)\s+(meth|methamphetamine|fentanyl)\b", re.I),
        re.compile(r"\b(launder money|counterfeit (?:currency|passport))\b", re.I),
    ),
    "credential_theft": (
        re.compile(r"\b(steal|dump|exfiltrate)\s+(?:the\s+)?(passwords?|credentials?|api keys?)\b", re.I),
    ),
}

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore (?:all |any )?(?:the )?(?:previous|prior|above) (?:instructions|prompts?)\b", re.I),
    re.compile(r"\bdisregard (?:the )?(?:system|previous) (?:prompt|instructions)\b", re.I),
    re.compile(r"\byou are now (?:in )?(?:developer|dan|god) mode\b", re.I),
    re.compile(r"\breveal (?:your )?(?:system prompt|instructions|hidden rules)\b", re.I),
    re.compile(r"\b(?:print|output|repeat) (?:everything )?above\b", re.I),
    re.compile(r"<\s*/?\s*(?:system|instructions)\s*>", re.I),
    re.compile(r"\bnew instructions?\s*:\s*", re.I),
)


@dataclass
class ModerationResult:
    flagged: bool = False
    categories: list[str] = field(default_factory=list)
    injection: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "flagged": self.flagged,
            "categories": self.categories,
            "injection": self.injection,
            "reason": self.reason,
        }


def moderate(text: str, check_injection: bool = True) -> ModerationResult:
    categories = [
        name for name, patterns in CATEGORY_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    ]
    injection = bool(check_injection and detect_injection(text))

    reasons: list[str] = []
    if categories:
        reasons.append(f"matched unsafe categories: {', '.join(categories)}")
    if injection:
        reasons.append("prompt-injection pattern detected")

    return ModerationResult(
        flagged=bool(categories),
        categories=categories,
        injection=injection,
        reason="; ".join(reasons),
    )


def detect_injection(text: str) -> list[str]:
    return [pattern.pattern for pattern in INJECTION_PATTERNS if pattern.search(text)]


def sanitize_retrieved(text: str) -> str:
    """Neutralise instruction-like content inside retrieved or fetched documents.

    Retrieved text is data, not instructions. Wrapping it and defanging the
    obvious hijack phrases stops indirect prompt injection through the corpus.
    """
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = pattern.sub("[redacted-instruction]", cleaned)
    return cleaned


def moderate_remote(text: str) -> ModerationResult | None:
    """Optional hosted moderation pass; returns None when unavailable."""
    from agentic_studio.settings import get_settings

    api_key = get_settings().llm.openai_api_key
    if not api_key:
        return None
    try:  # pragma: no cover - network path
        from agentic_studio.llm.providers._http import post_json

        data = post_json(
            "https://api.openai.com/v1/moderations",
            {"model": "omni-moderation-latest", "input": text[:4000]},
            {"Authorization": f"Bearer {api_key}"},
            10.0,
            "moderation",
        )
        result = (data.get("results") or [{}])[0]
        flagged_categories = [k for k, v in (result.get("categories") or {}).items() if v]
        return ModerationResult(
            flagged=bool(result.get("flagged")),
            categories=flagged_categories,
            reason="hosted moderation" if result.get("flagged") else "",
        )
    except Exception:
        return None
