import argparse
import io
import json
import sys

from app.config import settings
from app.logging_config import logger


def health_check() -> None:
    from app.llm.models import LLMHealthCheck
    from app.llm.provider import invoke_structured

    result = invoke_structured(LLMHealthCheck, "Say hello and confirm you are working.")
    print(f"Status:   {result.status}\nProvider: {result.provider}\nMessage:  {result.message}")


def run_pipeline(resume_path: str, job_url: str) -> int:
    from app.graph.builder import build_graph

    result = build_graph().invoke({"resume_path": resume_path, "job_url": job_url})

    output = {
        key: result.get(key)
        for key in ("parsed_resume", "job_details", "company_research", "completed_steps", "errors")
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 1 if result.get("errors") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous Job Application Agent")
    parser.add_argument("--resume", help="Path to resume (PDF, DOCX, or TXT)")
    parser.add_argument("--job-url", help="Job posting URL")
    parser.add_argument("--provider", choices=["gemini", "openai", "groq"], help="Override LLM_PROVIDER")
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
    if not (args.resume and args.job_url):
        parser.error("--resume and --job-url are required (or use --health-check)")
    return run_pipeline(args.resume, args.job_url)


if __name__ == "__main__":
    sys.exit(main())
