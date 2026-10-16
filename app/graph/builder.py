from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    parse_resume_node,
    research_company_node,
    scrape_job_node,
    tailor_resume_node,
)
from app.graph.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("scrape_job", scrape_job_node)
    graph.add_node("research_company", research_company_node)
    graph.add_node("tailor_resume", tailor_resume_node)

    # Resume parsing is independent of the job branch, so both start in parallel.
    graph.add_edge(START, "parse_resume")
    graph.add_edge(START, "scrape_job")
    graph.add_edge("scrape_job", "research_company")
    # Tailoring waits for both branches to finish.
    graph.add_edge(["parse_resume", "research_company"], "tailor_resume")
    graph.add_edge("tailor_resume", END)

    return graph.compile()
