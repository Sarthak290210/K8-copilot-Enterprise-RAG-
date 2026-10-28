"""The Phase 6 layers around the agent, in one place so the API and the eval run the same ones.

    screen_question:  L5 restructure (token limit) -> L2 input guard (on the original text)
                      -> input PII redaction (the agent, Cerebras, the cache and the logs only
                      ever see the redacted question; user's choice 2026-10-27)
    [agent: route -> retrieve -> L8 spotlighting + L3 prompt -> LLM -> L9 -> L10 grounding]
    screen_response:  L7 output moderation + PII redaction (answer and chunk previews)

L1 (request validation) is the QueryRequest model; L6 (token budget) needs a user, so it's in
api/query.py. A blocked question raises InputBlocked (the API turns it into a 400).
"""

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from app.models import ChatResponse
from app.security.content_moderation import moderate_and_redact, redact_pii
from app.security.input_guard import check_input_safe
from app.security.input_restructuring import restructure_input

BLOCK_MESSAGES = {
    "banned_phrase": "This request looks like an attempt to override my instructions.",
    "prompt_injection": "This request looks like an attempt to override my instructions. "
    'If it\'s a real question, please rephrase it (e.g. "How do I ignore ...?").',
    "toxicity": "Please keep the request respectful and about Kubernetes or IT operations.",
    "guard_unavailable": "The safety check is unavailable right now, so the request wasn't "
    "processed. Please try again later.",
}


class InputBlocked(Exception):
    def __init__(self, reason: str, scores: dict[str, float] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.scores = scores or {}
        self.message = BLOCK_MESSAGES.get(reason, "This request was blocked.")


@dataclass
class ScreenedInput:
    question: str  # what the agent gets (restructured, PII redacted)
    action: str = "unchanged"  # L5: unchanged | truncated
    pii: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def report(self) -> dict[str, object]:
        return {"input_action": self.action, "input_pii": self.pii, "guard_scores": self.scores}


async def screen_question(question: str) -> ScreenedInput:
    text, action = restructure_input(question)
    verdict = await check_input_safe(text)
    if not verdict.safe:
        logger.warning("Input blocked: {} {}", verdict.reason, verdict.scores)
        raise InputBlocked(verdict.reason or "blocked", verdict.scores)
    redacted, pii = redact_pii(text, ips=False)
    return ScreenedInput(redacted, action, pii, verdict.scores)


async def screen_response(
    response: ChatResponse, screened: ScreenedInput | None = None
) -> ChatResponse:
    """L7 on everything the user will read; records what the layers did in metadata.security.
    A pending-SQL response is our own fixed text plus SQL generated from the redacted
    question, so only its report is added."""
    security = {**(screened.report() if screened else {}), **response.metadata.security}
    if response.pending_sql is None:
        ok, clean, reason, pii = await asyncio.to_thread(moderate_and_redact, response.answer)
        response.answer = clean
        security["moderation"] = "ok" if ok else reason
        security["output_pii"] = pii
        for preview in response.chunks:
            preview.snippet, _ = redact_pii(preview.snippet)
        if not ok:
            logger.warning("Answer blocked by output moderation: {}", reason)
    response.metadata.security = security
    return response
