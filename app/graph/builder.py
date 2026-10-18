from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.edges import route_after_human, route_after_review
from app.graph.nodes import (
    assess_fit_node,
    human_review_node,
    parse_resume_node,
    research_company_node,
    review_application_node,
    revise_application_node,
    scrape_job_node,
    submit_application_node,
    tailor_resume_node,
    write_cover_letter_node,
)
from app.graph.state import AgentState

DRAFTING_NODES = ["tailor_resume", "write_cover_letter", "assess_fit"]


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the agent. A checkpointer is required so runs can pause for human approval."""
    graph = StateGraph(AgentState)

    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("scrape_job", scrape_job_node)
    graph.add_node("research_company", research_company_node)
    graph.add_node("tailor_resume", tailor_resume_node)
    graph.add_node("write_cover_letter", write_cover_letter_node)
    graph.add_node("assess_fit", assess_fit_node)
    graph.add_node("review_application", review_application_node)
    graph.add_node("revise_application", revise_application_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("submit_application", submit_application_node)

    # Resume parsing is independent of the job branch, so both start in parallel.
    graph.add_edge(START, "parse_resume")
    graph.add_edge(START, "scrape_job")
    graph.add_edge("scrape_job", "research_company")

    # Drafting and fit assessment each wait for both branches, then run in parallel.
    for node in DRAFTING_NODES:
        graph.add_edge(["parse_resume", "research_company"], node)

    # Self-evaluation loop, then a pause for the user's decision.
    graph.add_edge(DRAFTING_NODES, "review_application")
    graph.add_conditional_edges(
        "review_application", route_after_review, ["revise_application", "human_review", END]
    )
    graph.add_edge("revise_application", "review_application")
    graph.add_conditional_edges(
        "human_review", route_after_human, ["revise_application", "submit_application", END]
    )
    graph.add_edge("submit_application", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
