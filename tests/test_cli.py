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


def test_approve_can_confirm_sending_by_typing_the_address(monkeypatch):
    answer(monkeypatch, "a", "not an email", "hr@acme.io")
    decision = ask_decision(email_hint="hr@acme.io")
    assert decision is not None
    assert (decision.action, decision.send_to) == ("approve", "hr@acme.io")


def test_approve_without_sending_when_enter_pressed(monkeypatch):
    answer(monkeypatch, "a", "")
    decision = ask_decision(email_hint="hr@acme.io")
    assert decision is not None and decision.send_to is None


def test_export_documents_writes_pdf_and_docx(tmp_path):
    from app.llm.models import ContactInfo, CoverLetter, ParsedResume, TailoredResume
    from main import export_documents

    result = {
        "tailored_resume": TailoredResume(resume=ParsedResume(contact=ContactInfo(name="Jane"))).model_dump(),
        "cover_letter": CoverLetter(greeting="Hi,", paragraphs=["Body."], sign_off="Best,").model_dump(),
        "job_details": {"company": "Acme Corp", "title": "ML Engineer"},
    }
    paths = export_documents(result, str(tmp_path))
    assert [p.name for p in paths] == ["resume.pdf", "resume.docx", "cover_letter.pdf", "cover_letter.docx"]
    assert all(p.parent.name == "Acme_Corp_ML_Engineer" and p.stat().st_size > 0 for p in paths)
