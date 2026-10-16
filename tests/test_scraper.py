from unittest.mock import patch

import pytest

from app.llm.models import JobPosting
from app.tools.scraper import (
    JobNotFoundError,
    fetch_job_content,
    html_to_text,
    parse_job_text,
    scrape_job_posting,
    validate_job_url,
)

SAMPLE_JOB_CONTENT = """
Software Engineer - Backend
Company: TechCorp Inc.
Location: San Francisco, CA (Hybrid)
Salary: $120,000 - $160,000

Requirements:
- 3+ years Python experience
- Experience with FastAPI or Django
- Knowledge of PostgreSQL

Responsibilities:
- Design and implement RESTful APIs
- Write unit and integration tests
"""

SAMPLE_JOB_POSTING = JobPosting(
    title="Software Engineer - Backend",
    company="TechCorp Inc.",
    description="Backend engineer to build scalable APIs",
    requirements=["3+ years Python experience"],
    responsibilities=["Design and implement RESTful APIs"],
)


def posting() -> JobPosting:
    return SAMPLE_JOB_POSTING.model_copy()


# --- URL validation ---


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/job", "not a url", "http://localhost:8000/job", "http://10.0.0.5/job",
     "http://169.254.169.254/latest/meta-data", "http://printer.local/job"],
)
def test_rejects_unsafe_or_invalid_urls(url):
    with pytest.raises(ValueError):
        validate_job_url(url)


def test_accepts_public_urls():
    validate_job_url("https://jobs.lever.co/acme/123")
    validate_job_url("http://8.8.8.8/job")


# --- HTML conversion ---


def test_html_to_text_keeps_list_structure():
    markup = "<h4>Requirements</h4><ul><li>Python&nbsp;3</li><li>Docker</li></ul><p>Apply&amp;go</p>"
    assert html_to_text(markup) == "Requirements\n\n- Python 3\n\n- Docker\n\nApply&go"


# --- ATS APIs ---


@patch("app.tools.scraper._get_json")
def test_greenhouse_uses_public_api(mock_get_json):
    mock_get_json.return_value = {
        "title": "Engineer",
        "company_name": "Anduril",
        "location": {"name": "Costa Mesa"},
        "absolute_url": "https://job-boards.greenhouse.io/anduril/jobs/42",
        "content": "&lt;h4&gt;REQUIRED&lt;/h4&gt;&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;/ul&gt;",
    }

    text = fetch_job_content("https://job-boards.greenhouse.io/anduril/jobs/42?gh_src=x")

    mock_get_json.assert_called_once_with("https://boards-api.greenhouse.io/v1/boards/anduril/jobs/42")
    assert "Company: Anduril" in text
    assert "Apply URL: https://job-boards.greenhouse.io/anduril/jobs/42" in text
    assert "- Python" in text


@patch("app.tools.scraper._get_json")
def test_lever_uses_public_api(mock_get_json):
    mock_get_json.return_value = {
        "text": "Data Scientist",
        "categories": {"location": "Remote", "commitment": "Full-time"},
        "applyUrl": "https://jobs.lever.co/acme/abc/apply",
        "descriptionPlain": "Join us.",
        "lists": [{"text": "Requirements", "content": "<li>SQL</li>"}],
        "salaryRange": {"min": 100000, "max": 150000, "currency": "USD", "interval": "per-year"},
    }

    text = fetch_job_content("https://jobs.lever.co/acme/abc")

    mock_get_json.assert_called_once_with("https://api.lever.co/v0/postings/acme/abc")
    assert "Job title: Data Scientist" in text
    assert "Salary: 100000 - 150000 USD per-year" in text
    assert "- SQL" in text


@patch("app.tools.scraper._get_json")
def test_ashby_finds_job_in_board(mock_get_json):
    mock_get_json.return_value = {
        "jobs": [
            {"id": "other", "title": "Designer"},
            {"id": "abc", "title": "ML Engineer", "descriptionPlain": "Train models.",
             "applyUrl": "https://jobs.ashbyhq.com/acme/abc/application"},
        ]
    }

    text = fetch_job_content("https://jobs.ashbyhq.com/acme/abc")

    assert "Job title: ML Engineer" in text and "Train models." in text
    assert "Designer" not in text


@patch("app.tools.scraper._fetch_with_tavily")
@patch("app.tools.scraper._get_json", side_effect=JobNotFoundError("This job posting is no longer available."))
def test_expired_ats_job_does_not_fall_back(_mock_get_json, mock_tavily):
    with pytest.raises(JobNotFoundError, match="no longer available"):
        fetch_job_content("https://job-boards.greenhouse.io/acme/jobs/1")
    mock_tavily.assert_not_called()


@patch("app.tools.scraper._fetch_with_tavily", return_value=SAMPLE_JOB_CONTENT)
@patch("app.tools.scraper._get_json", side_effect=RuntimeError("503"))
def test_ats_outage_falls_back_to_tavily(_mock_get_json, mock_tavily):
    assert fetch_job_content("https://jobs.lever.co/acme/abc") == SAMPLE_JOB_CONTENT
    mock_tavily.assert_called_once()


# --- Generic page fetching ---


@patch("app.tools.scraper._fetch_with_selenium")
@patch("app.tools.scraper._fetch_with_tavily", return_value=SAMPLE_JOB_CONTENT)
def test_non_ats_url_uses_tavily(mock_tavily, mock_selenium):
    assert fetch_job_content("https://careers.example.com/job/1") == SAMPLE_JOB_CONTENT
    mock_selenium.assert_not_called()


@patch("app.tools.scraper._fetch_with_selenium", return_value=SAMPLE_JOB_CONTENT)
@patch("app.tools.scraper._fetch_with_tavily", side_effect=ValueError("too little content"))
def test_falls_back_to_selenium(_mock_tavily, mock_selenium):
    assert fetch_job_content("https://careers.example.com/job/1") == SAMPLE_JOB_CONTENT
    mock_selenium.assert_called_once()


@patch("app.tools.scraper._fetch_with_selenium", side_effect=RuntimeError("no chrome"))
@patch("app.tools.scraper._fetch_with_tavily", side_effect=ValueError("too little content"))
def test_raises_when_all_fetchers_fail(_mock_tavily, _mock_selenium):
    with pytest.raises(RuntimeError, match="Could not fetch job posting.*Tavily.*Selenium"):
        fetch_job_content("https://careers.example.com/job/1")


# --- LLM structuring ---


@patch("app.tools.scraper.invoke_structured", return_value=SAMPLE_JOB_POSTING)
@patch("app.tools.scraper.fetch_job_content", return_value=SAMPLE_JOB_CONTENT)
def test_scrape_job_posting_sets_source_url(_mock_fetch, mock_invoke):
    mock_invoke.return_value = posting()
    result = scrape_job_posting("https://careers.example.com/job/1")
    assert result.source_url == "https://careers.example.com/job/1"
    assert result.company == "TechCorp Inc."


@patch("app.tools.scraper.invoke_structured")
def test_parse_job_text_wraps_content_and_truncates(mock_invoke):
    mock_invoke.return_value = posting()
    parse_job_text("x" * 20_000)
    prompt = mock_invoke.call_args[0][1]
    assert "<job_posting>\n" + "x" * 15_000 + "\n</job_posting>" in prompt


@patch("app.tools.scraper.invoke_structured")
def test_parse_job_text_rejects_short_text(mock_invoke):
    with pytest.raises(ValueError, match="too short"):
        parse_job_text("Engineer wanted")
    mock_invoke.assert_not_called()


def test_job_posting_model_defaults():
    p = JobPosting(title="Test", company="TestCo", description="A test job")
    assert p.requirements == [] and p.nice_to_have == [] and p.responsibilities == []
    assert p.salary_range == "Not specified"
