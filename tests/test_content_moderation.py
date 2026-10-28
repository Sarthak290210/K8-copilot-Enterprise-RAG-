"""L7: regex PII redaction and output moderation (fake toxicity model)."""

import pytest

from app.config import settings
from app.security import content_moderation as cm
from app.security import input_guard


@pytest.mark.parametrize(
    ("text", "expected", "kinds"),
    [
        ("contact bob@corp.com", "contact [REDACTED_EMAIL]", ["email"]),
        ("card 4111 1111 1111 1111.", "card [REDACTED_CARD].", ["card"]),
        ("order 1234567812345678", "order 1234567812345678", []),  # fails the Luhn check
        ("call +1 415-555-0132", "call [REDACTED_PHONE]", ["phone"]),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234",
         "Authorization: Bearer [REDACTED_TOKEN]", ["token"]),
        ("key csk-abcdefghijklmnopqrst1234", "key [REDACTED_KEY]", ["api_key"]),
        ("data:\n  password: cGFzc3dvcmQxMjM0NTY3ODkw", "data:\n  password: [REDACTED_SECRET]",
         ["secret"]),
        ("public 8.8.8.8, ClusterIP 10.96.0.1", "public [REDACTED_IP], ClusterIP 10.96.0.1",
         ["ip"]),
    ],
)  # fmt: skip
def test_redacts_pii(text, expected, kinds):
    assert cm.redact_pii(text) == (expected, kinds)


@pytest.mark.parametrize(
    "text",
    [
        "kubectl get pods -n kube-system --field-selector=status.phase=Running",
        "image: nginx@sha256:3c3ea9d1a4b8c2e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9",
        "  serviceAccountName: myverylongserviceaccountname",
        "version 1.34.0 released 2025-08-27 on ports 8080 9090",
        "username: YWRtaW4=",
    ],
)
def test_keeps_normal_kubernetes_text(text):
    assert cm.redact_pii(text) == (text, [])


def test_question_redaction_keeps_ips():
    assert cm.redact_pii("why can't I reach 8.8.8.8?", ips=False) == (
        "why can't I reach 8.8.8.8?", [],
    )  # fmt: skip


class Toxic:
    def __init__(self, score):
        self.score = score

    def toxicity_score(self, text):
        return self.score


def test_toxic_answer_is_replaced(monkeypatch):
    monkeypatch.setattr(settings, "security_models_enabled", True)
    monkeypatch.setattr(input_guard, "get_models", lambda: Toxic(0.9))
    ok, clean, reason, _ = cm.moderate_and_redact("You are an idiot")
    assert (ok, clean, reason) == (False, cm.MODERATED_ANSWER, "toxic_output")


def test_clean_answer_is_redacted_only(monkeypatch):
    monkeypatch.setattr(settings, "security_models_enabled", True)
    monkeypatch.setattr(input_guard, "get_models", lambda: Toxic(0.01))
    ok, clean, reason, pii = cm.moderate_and_redact("Mail ops@corp.com [1]")
    assert (ok, clean, reason, pii) == (True, "Mail [REDACTED_EMAIL] [1]", None, ["email"])


def test_moderation_unavailable_follows_fail_closed(monkeypatch):
    monkeypatch.setattr(settings, "security_models_enabled", True)
    monkeypatch.setattr(input_guard, "get_models", lambda: None)
    monkeypatch.setattr(settings, "security_fail_closed", True)
    assert cm.moderate_output("A Pod is ...")[:2] == (False, "guard_unavailable")
    monkeypatch.setattr(settings, "security_fail_closed", False)
    assert cm.moderate_output("A Pod is ...")[0] is True
