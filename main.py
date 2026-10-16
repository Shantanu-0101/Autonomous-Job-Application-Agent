import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

from app.config import settings
from app.logging_config import logger

OUTPUT_KEYS = (
    "parsed_resume",
    "job_details",
    "company_research",
    "tailored_resume",
    "cover_letter",
    "fit_assessment",
    "application_review",
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


def run_pipeline(
    resume_path: str, job_url: str, job_text: str, tone: str, max_revisions: int, as_json: bool
) -> int:
    from app.graph.builder import build_graph

    inputs: dict[str, Any] = {
        "resume_path": resume_path,
        "job_url": job_url,
        "cover_letter_tone": tone,
        "max_revisions": max_revisions,
    }
    if job_text:
        inputs["job_description"] = job_text
    result = build_graph().invoke(inputs)

    if as_json:
        print(json.dumps({k: result.get(k) for k in OUTPUT_KEYS}, indent=2, ensure_ascii=False))
    else:
        print_report(result)
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
    parser.add_argument("--health-check", action="store_true", help="Verify the LLM connection")
    args = parser.parse_args()

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.provider:
        settings.llm_provider = args.provider
    logger.info("LLM provider: %s", settings.llm_provider)

    if args.health_check:
        health_check()
        return 0
    if not args.resume or not (args.job_url or args.job_file):
        parser.error("--resume and one of --job-url / --job-file are required (or use --health-check)")

    job_text = Path(args.job_file).read_text(encoding="utf-8") if args.job_file else ""
    return run_pipeline(
        args.resume, args.job_url, job_text, args.tone, args.max_revisions, args.json
    )


if __name__ == "__main__":
    sys.exit(main())
