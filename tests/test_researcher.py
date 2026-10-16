from unittest.mock import patch

import pytest

from app.llm.models import CompanyResearch
from app.tools.researcher import research_company

SAMPLE_RESEARCH = CompanyResearch(
    name="TechCorp",
    industry="Software",
    mission="To build great software",
    values=["Innovation", "Collaboration", "Integrity"],
    recent_news=["Raised $50M Series C", "Launched new AI product"],
    culture_notes="Fast-paced, collaborative environment",
    tech_stack=["Python", "React", "PostgreSQL", "AWS"],
    size="~500 employees",
    headquarters="San Francisco, CA",
    summary="TechCorp is an innovative software company focused on AI solutions.",
)


@patch("app.tools.researcher.invoke_structured", return_value=SAMPLE_RESEARCH)
@patch("app.tools.researcher._search_tavily", return_value="TechCorp is a great company...")
def test_research_company_success(_mock_search, mock_invoke):
    result = research_company("TechCorp")

    assert mock_invoke.call_args[0][0] is CompanyResearch
    assert result.name == "TechCorp"
    assert len(result.values) == 3
    assert len(result.recent_news) == 2
    assert len(result.tech_stack) == 4


@patch("app.tools.researcher.invoke_structured", return_value=SAMPLE_RESEARCH)
@patch("app.tools.researcher._search_tavily", return_value="Some content")
def test_search_queries_and_prompt_include_company_name(mock_search, mock_invoke):
    research_company("Acme Corp")

    queries = mock_search.call_args[0][0]
    assert all("Acme Corp" in q for q in queries)
    assert "Acme Corp" in mock_invoke.call_args[0][1]


@patch("app.tools.researcher._search_tavily", side_effect=ValueError("No search results found"))
def test_raises_on_no_search_results(_mock_search):
    with pytest.raises(ValueError, match="No search results"):
        research_company("NonexistentCorp12345")


@patch("app.tools.researcher.settings")
def test_raises_without_tavily_key(mock_settings):
    mock_settings.tavily_api_key = ""

    with pytest.raises(ValueError, match="TAVILY_API_KEY is required"):
        research_company("TechCorp")


def test_company_research_model_defaults():
    research = CompanyResearch(name="Test")
    assert research.values == []
    assert research.recent_news == []
    assert research.tech_stack == []
    assert research.industry == ""
    assert research.summary == ""


@patch("app.tools.researcher.invoke_structured", return_value=SAMPLE_RESEARCH)
@patch("app.tools.researcher._search_tavily", return_value="x" * 30000)
def test_search_results_truncated(_mock_search, mock_invoke):
    research_company("TechCorp")

    prompt = mock_invoke.call_args[0][1]
    assert "x" * 20000 in prompt and "x" * 20001 not in prompt
