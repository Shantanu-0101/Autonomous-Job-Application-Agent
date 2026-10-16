from unittest.mock import patch

from app.graph.builder import build_graph
from app.graph.nodes import parse_resume_node, scrape_job_node, tailor_resume_node
from app.llm.models import (
    CompanyResearch,
    ContactInfo,
    JobPosting,
    ParsedResume,
    TailoredResume,
)

JOB = JobPosting(title="Engineer", company="Acme", description="Build things")
RESEARCH = CompanyResearch(name="Acme")
RESUME = ParsedResume(contact=ContactInfo(name="Jane Doe"))
TAILORED = TailoredResume(resume=RESUME, matched_keywords=["Python"])


def test_parse_resume_node_skips_when_already_parsed():
    result = parse_resume_node({"parsed_resume": RESUME.model_dump()})
    assert result == {"completed_steps": ["parse_resume"]}


def test_parse_resume_node_errors_without_path():
    result = parse_resume_node({})
    assert "no resume provided" in result["errors"][0]


def test_scrape_job_node_errors_without_input():
    assert "no job URL or description" in scrape_job_node({})["errors"][0]


@patch("app.graph.nodes.scrape_job_posting")
@patch("app.graph.nodes.parse_job_text", return_value=JOB)
def test_scrape_job_node_prefers_pasted_description(mock_parse_text, mock_scrape):
    result = scrape_job_node({"job_description": "full text", "job_url": "https://x.com/job"})
    mock_parse_text.assert_called_once_with("full text", source_url="https://x.com/job")
    mock_scrape.assert_not_called()
    assert result["job_details"]["company"] == "Acme"


def test_tailor_node_errors_without_inputs():
    assert "missing" in tailor_resume_node({"job_details": JOB.model_dump()})["errors"][0]


@patch("app.graph.nodes.tailor_resume", return_value=TAILORED)
def test_tailor_node_passes_research_and_feedback(mock_tailor):
    result = tailor_resume_node({
        "parsed_resume": RESUME.model_dump(),
        "job_details": JOB.model_dump(),
        "company_research": RESEARCH.model_dump(),
        "tailoring_feedback": ["more APIs"],
    })
    args, kwargs = mock_tailor.call_args
    assert args == (RESUME, JOB, RESEARCH)
    assert kwargs == {"feedback": ["more APIs"]}
    assert result["tailored_resume"]["matched_keywords"] == ["Python"]


@patch("app.graph.nodes.tailor_resume", return_value=TAILORED)
@patch("app.graph.nodes.research_company", return_value=RESEARCH)
@patch("app.graph.nodes.scrape_job_posting", return_value=JOB)
@patch("app.graph.nodes.structure_resume", return_value=RESUME)
@patch("app.graph.nodes.extract_text", return_value="resume text")
@patch("app.graph.nodes.load_resume_file", return_value=(b"data", "r.pdf"))
def test_graph_runs_branches_then_tailors_once(*mocks):
    mock_tailor = mocks[-1]
    result = build_graph().invoke({"resume_path": "r.pdf", "job_url": "https://x.com/job"})

    assert result["parsed_resume"]["contact"]["name"] == "Jane Doe"
    assert result["job_details"]["company"] == "Acme"
    assert result["company_research"]["name"] == "Acme"
    assert result["tailored_resume"]["matched_keywords"] == ["Python"]
    assert sorted(result["completed_steps"]) == [
        "parse_resume", "research_company", "scrape_job", "tailor_resume"
    ]
    assert result["completed_steps"][-1] == "tailor_resume"
    mock_tailor.assert_called_once()
    assert not result.get("errors")


@patch("app.graph.nodes.scrape_job_posting", side_effect=RuntimeError("site down"))
@patch("app.graph.nodes.load_resume_file", side_effect=RuntimeError("disk error"))
def test_graph_collects_errors_from_parallel_failures(*_mocks):
    result = build_graph().invoke({"resume_path": "r.pdf", "job_url": "https://x.com/job"})

    errors = result["errors"]
    assert len(errors) == 4
    assert any("disk error" in e for e in errors)
    assert any("site down" in e for e in errors)
    assert any("no job details" in e for e in errors)
    assert any("Tailoring skipped" in e for e in errors)
