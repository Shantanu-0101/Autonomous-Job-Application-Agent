import json
import re
from typing import Any

from app.llm.models import CompanyResearch, JobPosting, ParsedResume

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def to_json(data: Any) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False)


def resume_payload(resume: ParsedResume) -> dict[str, Any]:
    """Resume facts for prompts, with experience, projects, and bullets indexed for reference."""

    def bullets(items: list[str]) -> list[dict[str, Any]]:
        return [{"index": j, "text": b} for j, b in enumerate(items)]

    return {
        "summary": resume.summary,
        "skills": resume.skills,
        "experience": [
            {
                "index": i,
                "title": e.title,
                "company": e.company,
                "dates": f"{e.start_date} - {e.end_date}".strip(" -"),
                "bullets": bullets(e.bullets),
            }
            for i, e in enumerate(resume.experience)
        ],
        "projects": [
            {"index": i, "name": p.name, "technologies": p.technologies, "bullets": bullets(p.bullets)}
            for i, p in enumerate(resume.projects)
        ],
        "education": [e.model_dump(exclude={"highlights"}) for e in resume.education],
        "certifications": resume.certifications,
        "achievements": resume.achievements,
    }


def job_payload(job: JobPosting) -> dict[str, Any]:
    return job.model_dump(include={
        "title", "company", "description", "requirements", "nice_to_have", "responsibilities"
    })


def company_payload(research: CompanyResearch | None) -> dict[str, Any]:
    if research is None:
        return {}
    return research.model_dump(include={"name", "mission", "values", "recent_news", "summary"})
