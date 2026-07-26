"""PII detection and redaction.

Runs on user input before it reaches a hosted model, and on model output before
it reaches a user or a log. Patterns are deliberately conservative: a missed
detection is a leak, a false positive is only a redacted token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,5}[\s-]?\d{3,4}[\s-]?\d{0,4}(?!\d)"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "AWS_KEY": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "BEARER_TOKEN": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b"),
    "JWT": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
}

# Order matters: longer, more specific patterns first so a card number is not
# partially consumed by the phone pattern.
_ORDER = [
    "JWT", "AWS_KEY", "BEARER_TOKEN", "EMAIL", "SSN", "AADHAAR", "PAN",
    "CREDIT_CARD", "IBAN", "IPV4", "PHONE",
]


@dataclass
class PIIMatch:
    kind: str
    value: str
    start: int
    end: int

    def masked(self) -> str:
        if self.kind == "EMAIL" and "@" in self.value:
            local, _, domain = self.value.partition("@")
            return f"{local[:1]}***@{domain}"
        tail = self.value[-4:] if len(self.value) > 4 else ""
        return f"<{self.kind}:***{tail}>"


def luhn_valid(digits: str) -> bool:
    numbers = [int(c) for c in digits if c.isdigit()]
    if len(numbers) < 13:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(numbers)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def detect_pii(text: str, kinds: list[str] | None = None) -> list[PIIMatch]:
    """Find PII, skipping overlaps and rejecting card numbers that fail Luhn."""
    selected = kinds or _ORDER
    matches: list[PIIMatch] = []
    claimed: list[tuple[int, int]] = []

    for kind in selected:
        pattern = PATTERNS.get(kind)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            start, end = match.span()
            value = match.group(0)
            if kind == "CREDIT_CARD" and not luhn_valid(value):
                continue
            if kind == "PHONE" and len(re.sub(r"\D", "", value)) < 7:
                continue
            if any(start < claimed_end and end > claimed_start for claimed_start, claimed_end in claimed):
                continue
            claimed.append((start, end))
            matches.append(PIIMatch(kind=kind, value=value, start=start, end=end))

    return sorted(matches, key=lambda m: m.start)


def redact_pii(text: str, kinds: list[str] | None = None) -> tuple[str, list[PIIMatch]]:
    """Return the text with PII masked plus what was found."""
    matches = detect_pii(text, kinds)
    if not matches:
        return text, []
    output: list[str] = []
    cursor = 0
    for match in matches:
        output.append(text[cursor : match.start])
        output.append(match.masked())
        cursor = match.end
    output.append(text[cursor:])
    return "".join(output), matches


def contains_pii(text: str, kinds: list[str] | None = None) -> bool:
    return bool(detect_pii(text, kinds))
