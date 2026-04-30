"""Regex-based PII redaction. Run on every log entry before storage.

Not exhaustive — production would add NER and a manual review pass — but covers
the common surface (emails, phone numbers, card-shaped numbers, common ID
shapes) well enough to demo the principle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
ACCOUNT_ID = re.compile(r"\b(?:acct|account|customer|cust)[-_]?[A-Z0-9]{6,}\b", re.IGNORECASE)
INVOICE_ID = re.compile(r"\b(?:inv|invoice)[-_]?[A-Z0-9]{6,}\b", re.IGNORECASE)

PATTERNS = [
    ("EMAIL", EMAIL),
    ("PHONE", PHONE),
    ("CARD", CARD),
    ("SSN", SSN),
    ("ACCOUNT", ACCOUNT_ID),
    ("INVOICE", INVOICE_ID),
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int]


def redact(text: str) -> RedactionResult:
    counts: dict[str, int] = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[REDACTED_{label}]", text)
        if n:
            counts[label] = n
    return RedactionResult(text=text, counts=counts)
