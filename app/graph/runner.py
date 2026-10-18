import uuid
from collections.abc import Iterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.llm.models import HumanDecision

STEP_LABELS = {
    "parse_resume": "Read your resume",
    "scrape_job": "Read the job posting",
    "research_company": "Researched the company",
    "tailor_resume": "Tailored your resume",
    "write_cover_letter": "Wrote the cover letter",
    "assess_fit": "Assessed your fit",
    "review_application": "Reviewed the application",
    "revise_application": "Revised the documents",
    "human_review": "Recorded your decision",
    "submit_application": "Prepared your application",
}


def new_run_config() -> RunnableConfig:
    """Each run gets its own thread so its checkpoints can be paused and resumed."""
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def stream_steps(
    graph: CompiledStateGraph, payload: dict[str, Any] | Command, config: RunnableConfig
) -> Iterator[str]:
    """Run (or resume) the graph, yielding the name of each step as it finishes."""
    for update in graph.stream(payload, config, stream_mode="updates"):
        for step in update:
            if step in STEP_LABELS:
                yield step


def resume_command(decision: HumanDecision) -> Command:
    return Command(resume=decision.model_dump())


def awaiting_approval(graph: CompiledStateGraph, config: RunnableConfig) -> bool:
    return "human_review" in graph.get_state(config).next


def current_state(graph: CompiledStateGraph, config: RunnableConfig) -> dict[str, Any]:
    return dict(graph.get_state(config).values)
