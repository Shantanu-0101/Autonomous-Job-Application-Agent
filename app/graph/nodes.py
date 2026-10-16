import logging

from app.graph.state import AgentState
from app.parsers.resume_parser import (
    ResumeParseError,
    extract_text,
    load_resume_file,
    structure_resume,
)
from app.tools.researcher import research_company
from app.tools.scraper import scrape_job_posting

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
    logger.info("Node: scrape_job — scraping %s", state["job_url"])
    try:
        posting = scrape_job_posting(state["job_url"])
        return {
            "job_details": posting.model_dump(),
            "completed_steps": ["scrape_job"],
        }
    except Exception as e:
        logger.error("Scraping failed: %s", e)
        return {"errors": [f"Scraping failed: {e}"]}


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
