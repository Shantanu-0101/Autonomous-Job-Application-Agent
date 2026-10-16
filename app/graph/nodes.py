import logging

from app.graph.state import AgentState
from app.parsers.resume_parser import (
    ResumeParseError,
    extract_text,
    load_resume_file,
    structure_resume,
)
from app.generators.resume_tailor import tailor_resume
from app.llm.models import CompanyResearch, JobPosting, ParsedResume
from app.tools.researcher import research_company
from app.tools.scraper import parse_job_text, scrape_job_posting

logger = logging.getLogger("job_agent.graph")


def parse_resume_node(state: AgentState) -> dict:
    if state.get("parsed_resume"):
        logger.info("Node: parse_resume — using pre-parsed resume")
        return {"completed_steps": ["parse_resume"]}

    path = state.get("resume_path")
    if not path:
        return {"errors": ["Resume parsing failed: no resume provided"]}

    logger.info("Node: parse_resume — parsing %s", path)
    try:
        data, filename = load_resume_file(path)
        text = extract_text(data, filename)
        parsed = structure_resume(text)
    except ResumeParseError as e:
        logger.warning("Resume parsing failed: %s", e)
        return {"errors": [f"Resume parsing failed: {e}"]}
    except Exception as e:
        logger.exception("Unexpected error while parsing resume")
        return {"errors": [f"Resume parsing failed: {e}"]}

    return {
        "resume_text": text,
        "parsed_resume": parsed.model_dump(),
        "completed_steps": ["parse_resume"],
    }


def scrape_job_node(state: AgentState) -> dict:
    description = state.get("job_description")
    url = state.get("job_url", "")
    if not (description or url):
        return {"errors": ["Scraping failed: no job URL or description provided"]}

    try:
        if description:
            logger.info("Node: scrape_job — parsing pasted job description")
            posting = parse_job_text(description, source_url=url)
        else:
            logger.info("Node: scrape_job — scraping %s", url)
            posting = scrape_job_posting(url)
    except Exception as e:
        logger.error("Scraping failed: %s", e)
        return {"errors": [f"Scraping failed: {e}"]}

    return {"job_details": posting.model_dump(), "completed_steps": ["scrape_job"]}


def research_company_node(state: AgentState) -> dict:
    job = state.get("job_details")
    if not job:
        return {"errors": ["Cannot research company: no job details available"]}

    company_name = job.get("company", "")
    if not company_name:
        return {"errors": ["Cannot research company: company name not found in job details"]}

    logger.info("Node: research_company — researching %s", company_name)
    try:
        research = research_company(company_name)
        return {
            "company_research": research.model_dump(),
            "completed_steps": ["research_company"],
        }
    except Exception as e:
        logger.error("Company research failed: %s", e)
        return {"errors": [f"Company research failed: {e}"]}


def tailor_resume_node(state: AgentState) -> dict:
    resume_data = state.get("parsed_resume")
    job_data = state.get("job_details")
    if not (resume_data and job_data):
        return {"errors": ["Tailoring skipped: resume or job details are missing"]}

    research_data = state.get("company_research")
    research = CompanyResearch.model_validate(research_data) if research_data else None

    logger.info("Node: tailor_resume")
    try:
        tailored = tailor_resume(
            ParsedResume.model_validate(resume_data),
            JobPosting.model_validate(job_data),
            research,
            feedback=state.get("tailoring_feedback"),
        )
    except Exception as e:
        logger.error("Resume tailoring failed: %s", e)
        return {"errors": [f"Resume tailoring failed: {e}"]}

    return {"tailored_resume": tailored.model_dump(), "completed_steps": ["tailor_resume"]}
