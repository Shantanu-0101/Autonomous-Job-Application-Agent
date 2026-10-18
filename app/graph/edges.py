from langgraph.graph import END

from app.config import settings
from app.evaluator.scorer import documents_to_revise
from app.graph.state import AgentState
from app.llm.models import ApplicationReview


def route_after_review(state: AgentState) -> str:
    """Revise weak documents within the revision budget, then hand off to the user."""
    review_data = state.get("application_review")
    if not review_data:
        return END  # nothing to approve: an earlier step failed
    max_revisions = state.get("max_revisions", settings.max_revisions)
    if state.get("revision_count", 0) < max_revisions:
        review = ApplicationReview.model_validate(review_data)
        if documents_to_revise(review, settings.review_quality_threshold):
            return "revise_application"
    return "human_review"


def route_after_human(state: AgentState) -> str:
    status = state.get("approval_status")
    if status == "changes_requested":
        return "revise_application"
    if status == "approved":
        return "submit_application"
    return END
