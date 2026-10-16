from app.graph.state import AgentState


def should_continue(state: AgentState) -> str:
    if (state.get("fit_score") or 0) >= 75:
        return "human_review"
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return "human_review"
    return "tailor_resume"


def is_approved(state: AgentState) -> str:
    if state.get("human_approved"):
        return "submit_application"
    return "end"
