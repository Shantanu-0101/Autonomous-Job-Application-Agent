import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from app.config import settings
from app.llm.models import (
    EMAIL_PATTERN,
    CoverLetter,
    HumanDecision,
    JobPosting,
    Submission,
    TailoredResume,
)
from app.logging_config import logger
from app.memory.tracker import STATUSES

OUTPUT_KEYS = (
    "parsed_resume",
    "job_details",
    "company_research",
    "tailored_resume",
    "cover_letter",
    "resume_warnings",
    "fit_assessment",
    "application_review",
    "approval_status",
    "submission",
    "completed_steps",
    "errors",
)


def health_check() -> None:
    from app.llm.models import LLMHealthCheck
    from app.llm.provider import invoke_structured

    result = invoke_structured(LLMHealthCheck, "Say hello and confirm you are working.")
    print(f"Status:   {result.status}\nProvider: {result.provider}\nMessage:  {result.message}")


def print_report(result: dict[str, Any]) -> None:
    from app.generators.render import render_cover_letter, render_markdown
    from app.llm.models import ApplicationReview, CoverLetter, FitAssessment, TailoredResume

    job = result.get("job_details") or {}
    if job:
        print(f"Job: {job.get('title')} at {job.get('company')}\n")

    history_message = (result.get("history") or {}).get("message")
    if history_message:
        print(f"===== History =====\n  ! {history_message}\n")

    resume_warnings = result.get("resume_warnings") or []
    if resume_warnings:
        print("===== Resume checks =====\n" + "\n".join(f"  ! {w}" for w in resume_warnings) + "\n")

    fit_data = result.get("fit_assessment")
    if fit_data:
        fit = FitAssessment.model_validate(fit_data)
        score = "n/a" if fit.score is None else f"{fit.score}/100"
        print(f"===== Fit: {score} — {fit.verdict} =====")
        marks = {"met": "[x]", "partial": "[~]", "not_met": "[ ]", "unknown": "[?]"}
        for r in fit.requirements:
            tag = "" if r.importance == "required" else " (nice to have)"
            print(f"  {marks[r.status]} {r.requirement}{tag}\n        {r.evidence}")
        for w in fit.warnings:
            print(f"  ! {w}")
        print("  Legend: [x] met  [~] partial  [ ] not met  [?] confirm yourself\n")

    review_data = result.get("application_review")
    if review_data:
        review = ApplicationReview.model_validate(review_data)
        print(
            f"===== Review: resume {review.resume_score}/100, cover letter "
            f"{review.cover_letter_score}/100 (after {review.revision} revision(s)) ====="
        )
        for note in review.notes:
            print(f"  - {note}")
        print()

    tailored_data = result.get("tailored_resume")
    if tailored_data:
        tailored = TailoredResume.model_validate(tailored_data)
        print(render_markdown(tailored.resume))
        for title, items in (
            ("Matched keywords", tailored.matched_keywords),
            ("Missing keywords (not added to resume)", tailored.missing_keywords),
            ("Changes", tailored.changes),
            ("Fabrication checks", tailored.warnings),
        ):
            if items:
                print(f"{title}:\n" + "\n".join(f"  - {i}" for i in items) + "\n")

    letter_data = result.get("cover_letter")
    if letter_data:
        letter = CoverLetter.model_validate(letter_data)
        print(f"===== Cover letter ({letter.word_count} words) =====\n")
        print(render_cover_letter(letter))
        if letter.warnings:
            print("Cover letter checks:\n" + "\n".join(f"  - {w}" for w in letter.warnings) + "\n")

    for error in result.get("errors") or []:
        print(f"ERROR: {error}")

    status = result.get("approval_status")
    if status:
        print(f"Status: {status.replace('_', ' ').capitalize()}")

    submission_data = result.get("submission")
    if submission_data:
        submission = Submission.model_validate(submission_data)
        if submission.status == "sent":
            print(f"Emailed to {submission.recipient} at {submission.sent_at}")
        elif submission.status == "failed":
            print(f"Not sent: {submission.error}")
        elif submission.apply_url:
            print(f"Apply here with your exported documents: {submission.apply_url}")


def write_report(result: dict[str, Any], as_json: bool, output: str | None) -> None:
    # Writing the file ourselves avoids shell redirects re-encoding UTF-8 (e.g. PowerShell's ">").
    with open(output, "w", encoding="utf-8") if output else contextlib.nullcontext(sys.stdout) as out:
        with contextlib.redirect_stdout(out):
            if as_json:
                print(json.dumps({k: result.get(k) for k in OUTPUT_KEYS}, indent=2, ensure_ascii=False))
            else:
                print_report(result)
    if output:
        logger.info("Wrote report to %s", output)


def ask_send_to(email_hint: str) -> str | None:
    """Sending is irreversible, so the user must type the recipient address to confirm."""
    while True:
        typed = input(
            f"This job takes applications by email at {email_hint}.\n"
            "Type the address to send your application now, or press Enter to skip: "
        ).strip()
        if not typed:
            return None
        if EMAIL_PATTERN.fullmatch(typed):
            return typed
        print(f'"{typed}" is not a valid email address.')


def ask_decision(email_hint: str | None = None) -> HumanDecision | None:
    try:
        while True:
            choice = input("\nApprove this application? [a]pprove / [c]hange / [r]eject: ").strip().lower()
            if choice in ("a", "approve"):
                send_to = ask_send_to(email_hint) if email_hint else None
                return HumanDecision(action="approve", send_to=send_to)
            if choice in ("r", "reject"):
                return HumanDecision(action="reject")
            if choice in ("c", "change"):
                resume_fb = input("What should change in the resume? (Enter to skip) ")
                letter_fb = input("What should change in the cover letter? (Enter to skip) ")
                if resume_fb.strip() or letter_fb.strip():
                    return HumanDecision(
                        action="request_changes",
                        resume_feedback=resume_fb,
                        cover_letter_feedback=letter_fb,
                    )
                print("Please describe at least one change.")
    except EOFError:
        return None


def export_documents(result: dict[str, Any], export_dir: str) -> list[Path]:
    from app.generators.export import (
        cover_letter_docx,
        cover_letter_pdf,
        resume_docx,
        resume_pdf,
        safe_filename,
    )

    tailored = TailoredResume.model_validate(result["tailored_resume"])
    letter = CoverLetter.model_validate(result["cover_letter"])
    job = result.get("job_details") or {}
    folder = Path(export_dir) / safe_filename(job.get("company", ""), job.get("title", ""))
    folder.mkdir(parents=True, exist_ok=True)
    contact = tailored.resume.contact
    files = {
        "resume.pdf": resume_pdf(tailored.resume),
        "resume.docx": resume_docx(tailored.resume),
        "cover_letter.pdf": cover_letter_pdf(letter, contact),
        "cover_letter.docx": cover_letter_docx(letter, contact),
    }
    paths = []
    for name, data in files.items():
        path = folder / name
        path.write_bytes(data)
        paths.append(path)
    return paths


def email_hint(state: dict[str, Any]) -> str | None:
    """The posting's application address, if sending is set up; otherwise explain how."""
    from app.tools.email_sender import email_configured
    from app.tools.submitter import application_email

    job_data = state.get("job_details")
    address = application_email(JobPosting.model_validate(job_data)) if job_data else None
    if address and not email_configured():
        print(f"This job takes applications at {address}. Set the SMTP_* values in .env to send from here.")
        return None
    return address


def print_applications() -> None:
    from app.memory.tracker import get_tracker

    records = get_tracker().list()
    if not records:
        print("No applications yet.")
        return
    print(f"{'ID':>4}  {'Status':<12} {'Fit':>4}  {'Updated':<10}  Role")
    for r in records:
        fit = "-" if r.fit_score is None else str(r.fit_score)
        print(f"{r.id:>4}  {r.status:<12} {fit:>4}  {r.updated_at[:10]:<10}  {r.title} at {r.company}")


def set_status(application_id: str, status: str, notes: str | None) -> int:
    from app.memory.tracker import get_tracker

    if status not in STATUSES:
        print(f"Unknown status '{status}'. Use one of: {', '.join(STATUSES)}")
        return 2
    try:
        record = get_tracker().update_status(int(application_id), status, notes)
    except (ValueError, KeyError) as e:
        print(f"Could not update application {application_id}: {e}")
        return 2
    print(f"#{record.id} {record.title} at {record.company} is now {record.status}.")
    return 0


def run_pipeline(
    resume_path: str,
    job_url: str,
    job_text: str,
    tone: str,
    max_revisions: int,
    as_json: bool,
    output: str | None = None,
    auto_approve: bool = False,
    send_to: str | None = None,
    export_dir: str = "data/outputs",
    allow_duplicate: bool = False,
) -> int:
    from app.graph.builder import build_graph
    from app.graph.runner import awaiting_approval, current_state, new_run_config, resume_command

    inputs: dict[str, Any] = {
        "resume_path": resume_path,
        "job_url": job_url,
        "cover_letter_tone": tone,
        "max_revisions": max_revisions,
        "allow_duplicate": allow_duplicate,
    }
    if job_text:
        inputs["job_description"] = job_text

    graph = build_graph()
    config = new_run_config()
    graph.invoke(inputs, config)

    while awaiting_approval(graph, config):
        state = current_state(graph, config)
        write_report(state, as_json, output)
        if auto_approve:
            decision: HumanDecision | None = HumanDecision(action="approve", send_to=send_to)
        else:
            decision = ask_decision(email_hint(state))
        if decision is None:
            print("No decision given; nothing was approved. Re-run with --yes to approve automatically.")
            return 2
        graph.invoke(resume_command(decision), config)

    result = current_state(graph, config)
    write_report(result, as_json, output)
    if result.get("approval_status") == "approved":
        paths = export_documents(result, export_dir)
        print("Saved:\n" + "\n".join(f"  {p}" for p in paths))
    return 1 if result.get("errors") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous Job Application Agent")
    parser.add_argument("--resume", help="Path to resume (PDF, DOCX, or TXT)")
    parser.add_argument("--job-url", default="", help="Job posting URL")
    parser.add_argument("--job-file", help="Text file with a pasted job description")
    parser.add_argument("--provider", choices=["gemini", "openai", "groq"], help="Override LLM_PROVIDER")
    parser.add_argument(
        "--tone", choices=["professional", "enthusiastic", "concise"], default="professional",
        help="Cover letter tone",
    )
    parser.add_argument(
        "--max-revisions", type=int, choices=range(0, 4), default=settings.max_revisions,
        help="Self-evaluation revision rounds (0 disables)",
    )
    parser.add_argument("--json", action="store_true", help="Print raw pipeline state as JSON")
    parser.add_argument("--output", help="Write the report to this file (UTF-8) instead of stdout")
    parser.add_argument("--yes", action="store_true", help="Approve automatically instead of asking")
    parser.add_argument(
        "--send-to", help="With --yes: email the approved application to this address"
    )
    parser.add_argument(
        "--export-dir", default="data/outputs", help="Where approved PDF/DOCX files are saved"
    )
    parser.add_argument(
        "--allow-duplicate", action="store_true", help="Continue even if you already applied to this job"
    )
    parser.add_argument("--list", action="store_true", help="Show your application history")
    parser.add_argument(
        "--set-status", nargs=2, metavar=("ID", "STATUS"),
        help=f"Update an application's status ({', '.join(STATUSES)})",
    )
    parser.add_argument("--notes", help="With --set-status: notes to save")
    parser.add_argument("--health-check", action="store_true", help="Verify the LLM connection")
    args = parser.parse_args()

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.provider:
        settings.llm_provider = args.provider
    logger.info("LLM provider: %s", settings.llm_provider)

    if args.list:
        print_applications()
        return 0
    if args.set_status:
        return set_status(args.set_status[0], args.set_status[1], args.notes)
    if args.health_check:
        health_check()
        return 0
    if not args.resume or not (args.job_url or args.job_file):
        parser.error("--resume and one of --job-url / --job-file are required (or use --health-check)")
    if args.send_to and not args.yes:
        parser.error("--send-to only applies with --yes; without it you'll be asked interactively")
    if args.send_to and not EMAIL_PATTERN.fullmatch(args.send_to.strip()):
        parser.error(f'--send-to "{args.send_to}" is not a valid email address')

    job_text = Path(args.job_file).read_text(encoding="utf-8") if args.job_file else ""
    return run_pipeline(
        args.resume,
        args.job_url,
        job_text,
        args.tone,
        args.max_revisions,
        args.json,
        args.output,
        auto_approve=args.yes,
        send_to=args.send_to.strip() if args.send_to else None,
        export_dir=args.export_dir,
        allow_duplicate=args.allow_duplicate,
    )


if __name__ == "__main__":
    sys.exit(main())
