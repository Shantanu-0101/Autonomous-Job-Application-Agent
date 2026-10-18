import logging
import re
import smtplib
import ssl
import time
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

from app.config import Settings, settings
from app.llm.models import EMAIL_PATTERN

logger = logging.getLogger("job_agent.email")

MAX_ATTEMPTS = 3


class EmailNotConfiguredError(RuntimeError):
    """SMTP settings are missing."""


class EmailSendError(RuntimeError):
    """Sending failed. The message is safe to show users."""


@dataclass(frozen=True)
class Attachment:
    filename: str
    data: bytes
    mime_type: str  # e.g. "application/pdf"


def is_valid_email(address: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(address.strip()))


def find_email(text: str) -> str | None:
    """The first email address in text, e.g. a job posting's application_method."""
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def email_configured(cfg: Settings | None = None) -> bool:
    cfg = cfg or settings
    return bool(cfg.smtp_host and cfg.smtp_username and cfg.smtp_password and cfg.sender_email)


def _single_line(text: str) -> str:
    # Header values come partly from scraped job postings; never let them inject new headers.
    return re.sub(r"[\r\n]+", " ", text).strip()


def build_email(
    to: str,
    subject: str,
    body: str,
    attachments: list[Attachment],
    reply_to: str = "",
    cfg: Settings | None = None,
) -> EmailMessage:
    cfg = cfg or settings
    if not is_valid_email(to):
        raise EmailSendError(f'"{to}" is not a valid email address.')

    message = EmailMessage()
    message["From"] = formataddr((_single_line(cfg.sender_name), cfg.sender_email))
    message["To"] = to.strip()
    if reply_to and is_valid_email(reply_to) and reply_to.strip() != cfg.sender_email:
        message["Reply-To"] = reply_to.strip()
    message["Subject"] = _single_line(subject)
    message["Message-ID"] = make_msgid(domain=cfg.sender_email.split("@")[-1])
    message.set_content(body)
    for attachment in attachments:
        maintype, subtype = attachment.mime_type.split("/", 1)
        message.add_attachment(
            attachment.data, maintype=maintype, subtype=subtype, filename=attachment.filename
        )
    return message


def _connect(cfg: Settings) -> smtplib.SMTP:
    context = ssl.create_default_context()
    if cfg.smtp_port == 465:
        return smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=30, context=context)
    server = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30)
    server.starttls(context=context)
    return server


LOGIN_REJECTED = (
    "The email server rejected the login. Check SMTP_USERNAME and SMTP_PASSWORD "
    "(Gmail needs an App Password)."
)
NETWORK_HINT = (
    "Your network may be blocking outgoing email — common on college or office Wi-Fi, and with "
    "antivirus 'mail shield' features. Try another network, such as a phone hotspot."
)


def _require_configuration(cfg: Settings) -> None:
    if not email_configured(cfg):
        raise EmailNotConfiguredError(
            "Email sending isn't set up. Add SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and SENDER_EMAIL."
        )


def verify_email_settings(cfg: Settings | None = None) -> None:
    """Connect and log in without sending anything. Raises EmailSendError with a fix to try."""
    cfg = cfg or settings
    _require_configuration(cfg)
    try:
        with _connect(cfg) as server:
            server.login(cfg.smtp_username, cfg.smtp_password)
    except smtplib.SMTPAuthenticationError as e:
        raise EmailSendError(LOGIN_REJECTED) from e
    except (smtplib.SMTPException, OSError) as e:
        raise EmailSendError(
            f"Could not connect to {cfg.smtp_host}:{cfg.smtp_port} ({e}). {NETWORK_HINT}"
        ) from e


def send_email(message: EmailMessage, cfg: Settings | None = None) -> str:
    """Send with retries for temporary failures. Returns the Message-ID."""
    cfg = cfg or settings
    _require_configuration(cfg)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        error: Exception
        temporary: bool
        stage = "connecting to the mail server"
        try:
            with _connect(cfg) as server:
                stage = "logging in"
                server.login(cfg.smtp_username, cfg.smtp_password)
                stage = "sending the message"
                server.send_message(message)
            logger.info("Sent email to %s", message["To"])
            return str(message["Message-ID"])
        except smtplib.SMTPAuthenticationError as e:
            raise EmailSendError(LOGIN_REJECTED) from e
        except smtplib.SMTPRecipientsRefused as e:
            raise EmailSendError(f"The email server refused the recipient {message['To']}.") from e
        # SMTP errors subclass OSError, so they must be handled before the network-error clause.
        except smtplib.SMTPServerDisconnected as e:
            temporary, error = True, e
        except smtplib.SMTPResponseException as e:
            temporary = 400 <= e.smtp_code < 500  # 4xx = try again later; 5xx = permanent
            error = e
        except smtplib.SMTPException as e:
            temporary, error = False, e
        except OSError as e:  # connection refused, DNS failure, timeout
            temporary, error = True, e

        if not temporary or attempt == MAX_ATTEMPTS:
            hint = f" {NETWORK_HINT}" if temporary else ""
            raise EmailSendError(f"Could not send the email (failed while {stage}: {error}).{hint}") from error
        logger.warning("Email attempt %d failed while %s (%s); retrying", attempt, stage, error)
        time.sleep(2**attempt)

    raise AssertionError("unreachable")
