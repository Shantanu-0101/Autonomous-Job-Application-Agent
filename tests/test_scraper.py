from unittest.mock import patch

import pytest

from app.llm.models import JobPosting
from app.tools.scraper import scrape_job_posting


SAMPLE_JOB_CONTENT = """
Software Engineer - Backend
Company: TechCorp Inc.
Location: San Francisco, CA (Hybrid)
Salary: $120,000 - $160,000

About the role:
We are looking for a backend engineer to build scalable APIs.

Requirements:
- 3+ years Python experience
- Experience with FastAPI or Django
- Knowledge of PostgreSQL
- Familiar with Docker and Kubernetes

Responsibilities:
- Design and implement RESTful APIs
- Write unit and integration tests
- Participate in code reviews
- Collaborate with frontend team

Apply at: careers@techcorp.com
"""

SAMPLE_JOB_POSTING = JobPosting(
    title="Software Engineer - Backend",
    company="TechCorp Inc.",
    location="San Francisco, CA (Hybrid)",
    description="Backend engineer to build scalable APIs",
    requirements=[
        "3+ years Python experience",
        "Experience with FastAPI or Django",
        "Knowledge of PostgreSQL",
        "Familiar with Docker and Kubernetes",
    ],
    responsibilities=[
        "Design and implement RESTful APIs",
        "Write unit and integration tests",
        "Participate in code reviews",
        "Collaborate with frontend team",
    ],
    salary_range="$120,000 - $160,000",
    application_method="careers@techcorp.com",
    source_url="https://example.com/job/123",
)


@patch("app.tools.scraper.invoke_structured")
@patch("app.tools.scraper._fetch_with_tavily")
def test_scrape_uses_tavily_when_key_present(mock_tavily, mock_invoke):
    mock_tavily.return_value = SAMPLE_JOB_CONTENT
    mock_invoke.return_value = SAMPLE_JOB_POSTING.model_copy()

    result = scrape_job_posting("https://example.com/job/123")

    mock_tavily.assert_called_once_with("https://example.com/job/123")
    assert result.title == "Software Engineer - Backend"
    assert result.company == "TechCorp Inc."
    assert result.source_url == "https://example.com/job/123"


@patch("app.tools.scraper.invoke_structured")
@patch("app.tools.scraper._fetch_with_selenium")
@patch("app.tools.scraper._fetch_with_tavily")
def test_falls_back_to_selenium_on_tavily_failure(
    mock_tavily, mock_selenium, mock_invoke
):
    mock_tavily.side_effect = Exception("Tavily error")
    mock_selenium.return_value = SAMPLE_JOB_CONTENT
    mock_invoke.return_value = SAMPLE_JOB_POSTING.model_copy()

    result = scrape_job_posting("https://example.com/job/123")

    mock_selenium.assert_called_once()
    assert result.company == "TechCorp Inc."


@patch("app.tools.scraper._fetch_with_selenium")
@patch("app.tools.scraper._fetch_with_tavily")
def test_raises_when_both_fail(mock_tavily, mock_selenium):
    mock_tavily.side_effect = Exception("Tavily error")
    mock_selenium.side_effect = Exception("Selenium error")

    with pytest.raises(RuntimeError, match="Both Tavily and Selenium failed"):
        scrape_job_posting("https://example.com/job/123")


@patch("app.tools.scraper.invoke_structured")
@patch("app.tools.scraper._fetch_with_tavily")
def test_raises_on_empty_content(mock_tavily, mock_invoke):
    mock_tavily.return_value = "   "

    with pytest.raises(ValueError, match="No content extracted"):
        scrape_job_posting("https://example.com/job/123")
    mock_invoke.assert_not_called()


def test_job_posting_model_defaults():
    posting = JobPosting(
        title="Test",
        company="TestCo",
        description="A test job",
    )
    assert posting.requirements == []
    assert posting.responsibilities == []
    assert posting.salary_range == "Not specified"
    assert posting.location == ""


@patch("app.tools.scraper.invoke_structured")
@patch("app.tools.scraper._fetch_with_tavily")
def test_content_truncated_for_llm(mock_tavily, mock_invoke):
    mock_tavily.return_value = "x" * 20000
    mock_invoke.return_value = SAMPLE_JOB_POSTING.model_copy()

    scrape_job_posting("https://example.com/job/123")

    prompt = mock_invoke.call_args[0][1]
    assert "x" * 15000 in prompt and "x" * 15001 not in prompt
