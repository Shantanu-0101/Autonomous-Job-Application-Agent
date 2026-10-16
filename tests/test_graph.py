from unittest.mock import patch

from app.graph.builder import build_graph
from app.graph.nodes import parse_resume_node
from app.llm.models import CompanyResearch, ContactInfo, JobPosting, ParsedResume

JOB = JobPosting(title="Engineer", company="Acme", description="Build things")
RESEARCH = CompanyResearch(name="Acme")
RESUME = ParsedResume(contact=ContactInfo(name="Jane Doe"))


def test_parse_resume_node_skips_when_already_parsed():
    result = parse_resume_node({"parsed_resume": RESUME.model_dump()})
    assert result == {"completed_steps": ["parse_resume"]}


def test_parse_resume_node_errors_without_path():
    result = parse_resume_node({})
    assert "no resume provided" in result["errors"][0]


@patch("app.graph.nodes.research_company", return_value=RESEARCH)
@patch("app.graph.nodes.scrape_job_posting", return_value=JOB)
@patch("app.graph.nodes.structure_resume", return_value=RESUME)
@patch("app.graph.nodes.extract_text", return_value="resume text")
@patch("app.graph.nodes.load_resume_file", return_value=(b"data", "r.pdf"))
def test_graph_runs_parallel_branches(*_mocks):
    result = build_graph().invoke({"resume_path": "r.pdf", "job_url": "https://x.com/job"})

    assert result["parsed_resume"]["contact"]["name"] == "Jane Doe"
    assert result["resume_text"] == "resume text"
    assert result["job_details"]["company"] == "Acme"
    assert result["company_research"]["name"] == "Acme"
    assert sorted(result["completed_steps"]) == ["parse_resume", "research_company", "scrape_job"]
    assert not result.get("errors")


@patch("app.graph.nodes.scrape_job_posting", side_effect=RuntimeError("site down"))
@patch("app.graph.nodes.load_resume_file", side_effect=RuntimeError("disk error"))
def test_graph_collects_errors_from_parallel_failures(*_mocks):
    result = build_graph().invoke({"resume_path": "r.pdf", "job_url": "https://x.com/job"})

    assert len(result["errors"]) == 3
    assert any("disk error" in e for e in result["errors"])
    assert any("site down" in e for e in result["errors"])
    assert any("no job details" in e for e in result["errors"])
