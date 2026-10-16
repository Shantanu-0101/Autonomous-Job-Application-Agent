import logging

from langchain_core.tools import tool

from app.config import settings
from app.llm.models import JobPosting
from app.llm.provider import invoke_structured

logger = logging.getLogger("job_agent.scraper")

EXTRACTION_PROMPT = """You are a job posting parser. Extract structured details from the web page content below.

Instructions:
- title: The exact job title
- company: The hiring company name
- location: City/state/country or "Remote" — extract whatever is stated
- description: A 2-3 sentence summary of what the role does
- requirements: A list of ALL required skills, qualifications, years of experience, education, etc. Extract each as a separate item. Look for sections like "Requirements", "Qualifications", "What you need", "Must have", "Skills", or bullet points describing what the candidate should have.
- responsibilities: A list of ALL job duties. Look for sections like "Responsibilities", "What you'll do", "The role", "Day to day", or bullet points describing the work.
- salary_range: Salary/compensation if mentioned, otherwise "Not specified"
- application_method: Email address or URL to apply, if mentioned

IMPORTANT: requirements and responsibilities should each have multiple items. Parse every bullet point or listed item individually. If the posting lists qualifications inline (not as bullets), split them into separate items.

Do NOT fabricate information. Only extract what is explicitly stated.

Web page content:
{content}
"""


def _fetch_with_tavily(url: str) -> str:
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)

    # Try extract first (best for static pages)
    try:
        response = client.extract(urls=[url])
        if response.get("results"):
            content = response["results"][0].get("raw_content", "")
            if content and len(content.strip()) > 200:
                return content
    except Exception:
        pass

    # Fall back to search with the URL as context
    logger.info("Tavily extract insufficient, using search for URL content")
    response = client.search(
        query=f"site:{url} job posting full description requirements responsibilities",
        max_results=3,
        include_raw_content=True,
    )
    if not response.get("results"):
        raise ValueError(f"Tavily returned no results for {url}")

    # Combine all result content for richer extraction
    parts = []
    for r in response["results"]:
        raw = r.get("raw_content") or r.get("content", "")
        if raw:
            parts.append(raw)
    combined = "\n\n---\n\n".join(parts)
    if not combined.strip():
        raise ValueError(f"Tavily returned empty content for {url}")
    return combined


def _fetch_with_selenium(url: str) -> str:
    import time

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        # Extra wait for JS-rendered content
        time.sleep(3)
        body = driver.find_element(By.TAG_NAME, "body")
        return body.text
    finally:
        driver.quit()


def scrape_job_posting(url: str) -> JobPosting:
    logger.info("Scraping job posting from %s", url)

    raw_content = ""
    try:
        if settings.tavily_api_key:
            logger.info("Using Tavily to extract content")
            raw_content = _fetch_with_tavily(url)
        else:
            logger.warning("No Tavily API key — falling back to Selenium")
            raw_content = _fetch_with_selenium(url)
    except Exception as e:
        logger.warning("Tavily failed (%s), trying Selenium fallback", e)
        try:
            raw_content = _fetch_with_selenium(url)
        except Exception as selenium_err:
            raise RuntimeError(
                f"Both Tavily and Selenium failed for {url}: {selenium_err}"
            ) from selenium_err

    if not raw_content.strip():
        raise ValueError(f"No content extracted from {url}")

    # Truncate to avoid token limits (keep first ~15k chars)
    content = raw_content[:15000]
    logger.info("Extracted %d chars, sending to LLM for structuring", len(content))

    result = invoke_structured(JobPosting, EXTRACTION_PROMPT.format(content=content))
    result.source_url = url

    logger.info("Parsed job: %s at %s", result.title, result.company)
    return result


@tool
def scrape_job_tool(url: str) -> dict:
    """Scrape a job posting URL and extract structured job details."""
    posting = scrape_job_posting(url)
    return posting.model_dump()
