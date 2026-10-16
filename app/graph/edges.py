from langgraph.graph import END

from app.config import settings
from app.evaluator.scorer import documents_to_revise
from app.graph.state import AgentState
from app.llm.models import ApplicationReview


def route_after_review(state: AgentState) -> str:
    """Revise weak documents until they pass review or the revision budget runs out."""
    review_data = state.get("application_review")
    if not review_data:
        return END
    max_revisions = state.get("max_revisions", settings.max_revisions)
    if state.get("revision_count", 0) >= max_revisions:
        return END
    review = ApplicationReview.model_validate(review_data)
    if documents_to_revise(review, settings.review_quality_threshold):
        return "revise_application"
    return END
