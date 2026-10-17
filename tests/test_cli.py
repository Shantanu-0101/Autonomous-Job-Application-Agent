import builtins

import pytest

from main import ask_decision


def answer(monkeypatch, *responses: str) -> None:
    replies = iter(responses)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(replies))


@pytest.mark.parametrize(("reply", "action"), [("a", "approve"), ("APPROVE", "approve"), ("r", "reject")])
def test_simple_choices(monkeypatch, reply, action):
    answer(monkeypatch, reply)
    decision = ask_decision()
    assert decision is not None and decision.action == action


def test_change_request_asks_until_feedback_given(monkeypatch):
    answer(monkeypatch, "x", "c", "", "", "c", "Lead with FastAPI", "")
    decision = ask_decision()
    assert decision is not None
    assert decision.action == "request_changes"
    assert decision.resume_feedback == "Lead with FastAPI"


def test_no_input_means_no_decision(monkeypatch):
    def eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    assert ask_decision() is None
