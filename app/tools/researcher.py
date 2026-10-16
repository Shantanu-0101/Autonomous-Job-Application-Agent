import logging

from langchain_core.tools import tool

from app.config import settings
from app.llm.models import CompanyResearch
from app.llm.provider import invoke_structured

logger = logging.getLogger("job_agent.researcher")

RESEARCH_PROMPT = """You are a company research analyst. Based on the search results below, create a structured research brief about "{company}".

Instructions:
- name: The official company name
- industry: Their primary industry or sector
- mission: Their mission statement or core purpose (quote if possible)
- values: List their stated core values or cultural principles. Extract each one separately.
- recent_news: List 2-5 recent news items, product launches, funding rounds, or notable events. Each should be one sentence.
- culture_notes: Summarize the work culture — what employees say, work-life balance, management style
- tech_stack: List technologies, programming languages, frameworks, and tools they use
- size: Employee count or scale description (e.g., "~5000 employees", "Series B startup")
- headquarters: HQ city and country
- summary: Write a 2-3 sentence overview useful for a cover letter — what makes this company distinctive

Only include information supported by the search results. Do not fabricate details.

Search results:
{search_results}
"""

SEARCH_QUERIES = [
    '"{company}" company mission values culture about',
    '"{company}" recent news announcements 2026',
    '"{company}" tech stack engineering blog technology',
    '"{company}" glassdoor employee reviews work culture',
]


def _search_tavily(queries: list[str]) -> str:
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    all_results = []

    for query in queries:
        try:
            response = client.search(query=query, max_results=3)
            for r in response.get("results", []):
                content = r.get("content", "")
                url = r.get("url", "")
                if content:
                    all_results.append(f"[Source: {url}]\n{content}")
        except Exception as e:
            logger.warning("Search failed for query '%s': %s", query, e)

    if not all_results:
        raise ValueError("No search results found for any query")

    return "\n\n---\n\n".join(all_results)


def research_company(company_name: str) -> CompanyResearch:
    logger.info("Researching company: %s", company_name)

    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY is required for company research")

    queries = [q.format(company=company_name) for q in SEARCH_QUERIES]
    search_results = _search_tavily(queries)

    # Truncate to avoid token limits
    search_results = search_results[:20000]
    logger.info(
        "Collected %d chars of research, sending to LLM", len(search_results)
    )

    result = invoke_structured(
        CompanyResearch,
        RESEARCH_PROMPT.format(company=company_name, search_results=search_results),
    )

    logger.info(
        "Research complete: %s (%s) — %d values, %d news items, %d tech items",
        result.name,
        result.industry,
        len(result.values),
        len(result.recent_news),
        len(result.tech_stack),
    )
    return result


@tool
def research_company_tool(company_name: str) -> dict:
    """Research a company and return a structured brief about their mission, culture, news, and tech stack."""
    research = research_company(company_name)
    return research.model_dump()
