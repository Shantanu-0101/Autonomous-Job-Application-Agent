import re
import smtplib
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.llm.models import (
    ContactInfo,
    CoverLetter,
    HumanDecision,
    JobPosting,
    ParsedResume,
    TailoredResume,
)
from app.tools.email_sender import (
    Attachment,
    EmailNotConfiguredError,
    EmailSendError,
    build_email,
    email_configured,
    find_email,
    is_valid_email,
    send_email,
    verify_email_settings,
)
from app.tools.submitter import application_email, application_url, compose_application

PDF = Attachment("resume.pdf", b"%PDF-1.7 fake", "application/pdf")


@pytest.fixture
def smtp_settings(monkeypatch):
    for name, value in {
        "smtp_host": "smtp.example.com", "smtp_port": 587, "smtp_username": "me@example.com",
        "smtp_password": "app-password", "sender_email": "me@example.com", "sender_name": "Jane Doe",
    }.items():
        monkeypatch.setattr(settings, name, value)


@pytest.fixture
def no_sleep():
    with patch("app.tools.email_sender.time.sleep") as sleep:
        yield sleep


def smtp_mock(*send_effects):
    server = MagicMock()
    server.__enter__.return_value = server
    server.send_message.side_effect = list(send_effects) or None
    return server


# --- Addresses ---


def test_email_validation_and_extraction():
    assert is_valid_email(" jobs@acme.io ")
    assert not is_valid_email("jobs@acme")
    assert not is_valid_email("jobs@acme.io\nBcc: x@y.com")
    assert find_email("Send your CV to careers@acme.io by Friday") == "careers@acme.io"
    assert find_email("https://acme.io/apply") is None


def test_decision_validates_send_to():
    assert HumanDecision(action="approve", send_to=" hr@acme.io ").send_to == "hr@acme.io"
    with pytest.raises(ValueError, match="not a valid email"):
        HumanDecision(action="approve", send_to="hr at acme")
    with pytest.raises(ValueError, match="Only an approved"):
        HumanDecision(action="reject", send_to="hr@acme.io")


# --- Message building ---


def test_build_email_sets_headers_and_attachments(smtp_settings):
    message = build_email("hr@acme.io", "Application", "Hello", [PDF], reply_to="jane@personal.com")
    assert message["From"] == "Jane Doe <me@example.com>"
    assert message["To"] == "hr@acme.io"
    assert message["Reply-To"] == "jane@personal.com"
    assert message["Message-ID"].endswith("@example.com>")
    parts = list(message.iter_attachments())
    assert [(p.get_filename(), p.get_content_type()) for p in parts] == [("resume.pdf", "application/pdf")]
    assert parts[0].get_content() == b"%PDF-1.7 fake"


def test_build_email_strips_header_injection(smtp_settings):
    message = build_email("hr@acme.io", "Engineer\r\nBcc: victim@example.com", "Hi", [])
    assert message["Subject"] == "Engineer Bcc: victim@example.com"
    assert message["Bcc"] is None


def test_build_email_rejects_invalid_recipient(smtp_settings):
    with pytest.raises(EmailSendError, match="not a valid email"):
        build_email("not-an-address", "s", "b", [])


def test_compose_application_uses_letter_and_pdf_attachments(smtp_settings):
    tailored = TailoredResume(resume=ParsedResume(contact=ContactInfo(name="Jane Doe", email="jane@p.com")))
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["I build APIs."], sign_off="Best,",
                         signature="Jane Doe")
    job = JobPosting(title="Engineer", company="Acme", description="x", application_method="hr@acme.io")

    message = compose_application("hr@acme.io", job, tailored, letter)

    assert message["Subject"] == "Application for Engineer — Jane Doe"
    assert message["Reply-To"] == "jane@p.com"
    assert "I build APIs." in message.get_body().get_content()  # type: ignore[union-attr]
    names = [p.get_filename() for p in message.iter_attachments()]
    assert names == ["Jane_Doe_Resume.pdf", "Jane_Doe_Cover_Letter.pdf"]


def test_application_email_and_url():
    job = JobPosting(title="E", company="A", description="d", source_url="https://boards.io/1",
                     application_method="Email careers@acme.io")
    assert application_email(job) == "careers@acme.io"
    assert application_url(job) == "https://boards.io/1"
    job.application_method = "https://acme.io/apply"
    assert application_email(job) is None and application_url(job) == "https://acme.io/apply"


# --- Sending ---


def test_send_requires_configuration():
    assert not email_configured()
    with pytest.raises(EmailNotConfiguredError):
        send_email(MagicMock())


@patch("app.tools.email_sender.smtplib.SMTP")
def test_send_uses_starttls_and_login(mock_smtp, smtp_settings):
    server = smtp_mock()
    mock_smtp.return_value = server
    message = build_email("hr@acme.io", "s", "b", [])

    assert send_email(message) == message["Message-ID"]
    mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=30)
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("me@example.com", "app-password")
    server.send_message.assert_called_once_with(message)


@patch("app.tools.email_sender.smtplib.SMTP_SSL")
def test_port_465_uses_implicit_tls(mock_ssl, smtp_settings, monkeypatch):
    monkeypatch.setattr(settings, "smtp_port", 465)
    mock_ssl.return_value = smtp_mock()
    send_email(build_email("hr@acme.io", "s", "b", []))
    assert mock_ssl.call_args.args == ("smtp.example.com", 465)


@patch("app.tools.email_sender.smtplib.SMTP")
def test_retries_temporary_failures(mock_smtp, smtp_settings, no_sleep):
    server = smtp_mock(smtplib.SMTPServerDisconnected("dropped"),
                       smtplib.SMTPResponseException(421, b"try later"), None)
    mock_smtp.return_value = server
    send_email(build_email("hr@acme.io", "s", "b", []))
    assert server.send_message.call_count == 3
    assert no_sleep.call_count == 2


@patch("app.tools.email_sender.smtplib.SMTP")
def test_gives_up_after_max_attempts(mock_smtp, smtp_settings, no_sleep):
    mock_smtp.return_value = smtp_mock(*[smtplib.SMTPServerDisconnected("dropped")] * 3)
    with pytest.raises(EmailSendError, match="Could not send"):
        send_email(build_email("hr@acme.io", "s", "b", []))


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (smtplib.SMTPAuthenticationError(535, b"bad creds"), "rejected the login"),
        (smtplib.SMTPRecipientsRefused({"hr@acme.io": (550, b"no such user")}), "refused the recipient"),
        (smtplib.SMTPResponseException(554, b"spam"), "Could not send"),
    ],
)
@patch("app.tools.email_sender.smtplib.SMTP")
def test_permanent_failures_are_not_retried(mock_smtp, error, message, smtp_settings, no_sleep):
    server = smtp_mock(error)
    mock_smtp.return_value = server
    with pytest.raises(EmailSendError, match=message):
        send_email(build_email("hr@acme.io", "s", "b", []))
    assert server.send_message.call_count == 1
    no_sleep.assert_not_called()


@patch("app.tools.email_sender.smtplib.SMTP", side_effect=ConnectionRefusedError("refused"))
def test_network_errors_are_retried(mock_smtp, smtp_settings, no_sleep):
    with pytest.raises(EmailSendError, match="refused"):
        send_email(build_email("hr@acme.io", "s", "b", []))
    assert mock_smtp.call_count == 3


@patch("app.tools.email_sender.smtplib.SMTP")
def test_failure_message_names_the_step_and_suggests_a_fix(mock_smtp, smtp_settings, no_sleep):
    server = smtp_mock()
    server.login.side_effect = smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
    mock_smtp.return_value = server

    with pytest.raises(EmailSendError) as raised:
        send_email(build_email("hr@acme.io", "s", "b", []))

    message = str(raised.value)
    assert "failed while logging in: Connection unexpectedly closed" in message
    assert "phone hotspot" in message


def test_verify_requires_configuration():
    with pytest.raises(EmailNotConfiguredError):
        verify_email_settings()


@patch("app.tools.email_sender.smtplib.SMTP")
def test_verify_logs_in_without_sending(mock_smtp, smtp_settings):
    server = smtp_mock()
    mock_smtp.return_value = server
    verify_email_settings()
    server.login.assert_called_once_with("me@example.com", "app-password")
    server.send_message.assert_not_called()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (smtplib.SMTPAuthenticationError(535, b"bad"), "rejected the login"),
        (smtplib.SMTPServerDisconnected("closed"), "Could not connect to smtp.example.com:587 (closed)"),
    ],
)
@patch("app.tools.email_sender.smtplib.SMTP")
def test_verify_explains_failures(mock_smtp, error, message, smtp_settings):
    server = smtp_mock()
    server.login.side_effect = error
    mock_smtp.return_value = server
    with pytest.raises(EmailSendError, match=re.escape(message)):
        verify_email_settings()
