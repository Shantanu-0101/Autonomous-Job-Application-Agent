from langgraph.graph import END, START, StateGraph

from app.graph.edges import route_after_review
from app.graph.nodes import (
    assess_fit_node,
    parse_resume_node,
    research_company_node,
    review_application_node,
    revise_application_node,
    scrape_job_node,
    tailor_resume_node,
    write_cover_letter_node,
)
from app.graph.state import AgentState

DRAFTING_NODES = ["tailor_resume", "write_cover_letter", "assess_fit"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("scrape_job", scrape_job_node)
    graph.add_node("research_company", research_company_node)
    graph.add_node("tailor_resume", tailor_resume_node)
    graph.add_node("write_cover_letter", write_cover_letter_node)
    graph.add_node("assess_fit", assess_fit_node)
    graph.add_node("review_application", review_application_node)
    graph.add_node("revise_application", revise_application_node)

    # Resume parsing is independent of the job branch, so both start in parallel.
    graph.add_edge(START, "parse_resume")
    graph.add_edge(START, "scrape_job")
    graph.add_edge("scrape_job", "research_company")

    # Drafting and fit assessment each wait for both branches, then run in parallel.
    for node in DRAFTING_NODES:
        graph.add_edge(["parse_resume", "research_company"], node)

    # Self-evaluation loop: review, revise weak documents, review again.
    graph.add_edge(DRAFTING_NODES, "review_application")
    graph.add_conditional_edges(
        "review_application", route_after_review, ["revise_application", END]
    )
    graph.add_edge("revise_application", "review_application")

    return graph.compile()
