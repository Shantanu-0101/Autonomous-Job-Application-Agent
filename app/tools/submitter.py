import logging
from datetime import UTC, datetime
from email.message import EmailMessage

from app.generators.export import cover_letter_pdf, resume_pdf, safe_filename
from app.generators.render import render_cover_letter
from app.llm.models import CoverLetter, JobPosting, Submission, TailoredResume
from app.tools.email_sender import Attachment, build_email, find_email, send_email

logger = logging.getLogger("job_agent.submitter")


def application_email(job: JobPosting) -> str | None:
    """The address the posting asks applications to be sent to, if any."""
    return find_email(job.application_method)


def application_url(job: JobPosting) -> str:
    method = job.application_method.strip()
    return method if method.startswith(("http://", "https://")) else job.source_url


def application_attachments(tailored: TailoredResume, letter: CoverLetter) -> list[Attachment]:
    contact = tailored.resume.contact
    name = safe_filename(contact.name) or "Applicant"
    return [
        Attachment(f"{name}_Resume.pdf", resume_pdf(tailored.resume), "application/pdf"),
        Attachment(f"{name}_Cover_Letter.pdf", cover_letter_pdf(letter, contact), "application/pdf"),
    ]


def compose_application(
    to: str, job: JobPosting, tailored: TailoredResume, letter: CoverLetter
) -> EmailMessage:
    contact = tailored.resume.contact
    subject = f"Application for {job.title}" + (f" — {contact.name}" if contact.name else "")
    return build_email(
        to=to,
        subject=subject,
        body=render_cover_letter(letter),
        attachments=application_attachments(tailored, letter),
        reply_to=contact.email,
    )


def submit_by_email(
    to: str, job: JobPosting, tailored: TailoredResume, letter: CoverLetter
) -> Submission:
    message_id = send_email(compose_application(to, job, tailored, letter))
    logger.info("Application for %s at %s sent", job.title, job.company)
    return Submission(
        method="email",
        status="sent",
        recipient=to,
        message_id=message_id,
        sent_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
