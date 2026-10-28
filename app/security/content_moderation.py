"""L7 content moderation + PII redaction (Phase 6).

PII is found with regexes (user's choice: instant, no model): emails, card numbers (13-19
digits that pass the Luhn check), phone numbers (with a + country code or the 3-3-4 form),
JWTs, bearer tokens, API keys with known prefixes, Kubernetes Secret values and long base64
blobs, and public IPv4 addresses. Private / loopback IPs (10.x, 172.16-31.x, 192.168.x,
127.x) are kept: they're normal in Kubernetes answers ("the Service gets 10.96.0.1"). Names
and street addresses aren't caught (that needs an NER model).

`redact_pii` runs on the question too, before the agent (user's choice 2026-10-27), but there
`ips=False`: "why can't I reach 10.0.0.5" is a normal question.

Output moderation scores the answer with the input guard's toxicity model (loaded once),
in windows of ~2,000 characters (the model reads 512 tokens), against
`OUTPUT_TOXICITY_THRESHOLD`; a toxic answer is replaced as a whole.
"""

import ipaddress
import re

from loguru import logger

from app.config import settings
from app.security import input_guard

MODERATED_ANSWER = "I can't share that response. Please ask a Kubernetes or IT-operations question."
_WINDOW_CHARS = 2000

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_PHONE = re.compile(
    r"(?<![\w.+-])(?:\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}"
    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})(?![\w.-])"
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{20,}")
_API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|csk-[A-Za-z0-9]{16,}|tvly-[A-Za-z0-9-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,})"
)
# `key: <base64>` lines (a Secret's data:), and any long base64-looking run. Both need a
# character outside hex, so sha256 image digests and commit hashes are kept.
_NOT_HEX = r"(?=[A-Za-z0-9+/]*[G-Zg-z+/])"
_SECRET_VALUE = re.compile(
    rf"(?m)^(\s*[\w.-]+:\s+){_NOT_HEX}(?=[A-Za-z0-9+/]*\d)([A-Za-z0-9+/]{{16,}}={{0,2}})\s*$"
)
_BASE64_BLOB = re.compile(
    rf"(?<![A-Za-z0-9+/]){_NOT_HEX}(?=[A-Za-z0-9+/]*\d)(?=[A-Za-z0-9+/]*[A-Za-z])"
    r"[A-Za-z0-9+/]{40,}={0,2}"
)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return "[REDACTED_CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group()


def _ip(match: re.Match[str]) -> str:
    try:
        ip = ipaddress.IPv4Address(match.group())
    except ValueError:
        return match.group()  # not an address (e.g. 999.1.1.1 or a version number)
    return match.group() if ip.is_private or ip.is_loopback else "[REDACTED_IP]"


def redact_pii(text: str, ips: bool = True) -> tuple[str, list[str]]:
    """The text with PII replaced by [REDACTED_*], and the kinds found (for metadata/logs)."""
    found: list[str] = []

    def sub(pattern: re.Pattern[str], repl, kind: str) -> None:  # noqa: ANN001
        nonlocal text
        new = pattern.sub(repl, text)
        if new != text:
            found.append(kind)
            text = new

    sub(_JWT, "[REDACTED_TOKEN]", "token")
    sub(_BEARER, r"\1[REDACTED_TOKEN]", "token")
    sub(_API_KEY, "[REDACTED_KEY]", "api_key")
    sub(_EMAIL, "[REDACTED_EMAIL]", "email")
    sub(_CARD, _card, "card")
    sub(_PHONE, "[REDACTED_PHONE]", "phone")
    sub(_SECRET_VALUE, r"\1[REDACTED_SECRET]", "secret")
    sub(_BASE64_BLOB, "[REDACTED_SECRET]", "secret")
    if ips:
        sub(_IPV4, _ip, "ip")
    return text, list(dict.fromkeys(found))


def moderate_output(text: str) -> tuple[bool, str | None, float | None]:
    """(ok, reason, highest toxicity score). Sync: CPU-bound (run it in a thread)."""
    if not settings.security_models_enabled or not text.strip():
        return True, None, None
    models = input_guard.get_models()
    if models is None:
        if settings.security_fail_closed:
            return False, "guard_unavailable", None
        logger.warning("Output moderation unavailable; answer not checked (fail open)")
        return True, None, None
    try:
        windows = [text[i : i + _WINDOW_CHARS] for i in range(0, len(text), _WINDOW_CHARS)]
        score = max(models.toxicity_score(window) for window in windows)
    except Exception as exc:
        logger.error("Output moderation crashed ({})", type(exc).__name__)
        return (not settings.security_fail_closed), "guard_unavailable", None
    if score > settings.output_toxicity_threshold:
        return False, "toxic_output", round(score, 4)
    return True, None, round(score, 4)


def moderate_and_redact(text: str) -> tuple[bool, str, str | None, list[str]]:
    """(ok, clean text, reason, PII kinds redacted). A blocked answer becomes MODERATED_ANSWER."""
    ok, reason, _ = moderate_output(text)
    if not ok:
        return False, MODERATED_ANSWER, reason, []
    clean, found = redact_pii(text)
    return True, clean, None, found
