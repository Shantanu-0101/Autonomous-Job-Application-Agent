import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langgraph.graph import END

from app.graph.builder import build_graph
from app.graph.edges import route_after_human, route_after_review
from app.graph.runner import awaiting_approval, current_state, new_run_config, resume_command
from app.graph.nodes import (
    parse_resume_node,
    review_application_node,
    revise_application_node,
    scrape_job_node,
    tailor_resume_node,
    write_cover_letter_node,
)
from app.graph.state import AgentState
from app.llm.models import (
    ApplicationReview,
    BulletChange,
    CompanyResearch,
    ContactInfo,
    CoverLetter,
    Experience,
    FitAssessment,
    HumanDecision,
    JobPosting,
    ParsedResume,
    TailoredResume,
)

JOB = JobPosting(title="Engineer", company="Acme", description="Build things")
RESEARCH = CompanyResearch(name="Acme")
RESUME = ParsedResume(contact=ContactInfo(name="Jane Doe"))
TAILORED = TailoredResume(resume=RESUME, matched_keywords=["Python"])
TAILORED_V2 = TailoredResume(resume=RESUME, matched_keywords=["Python", "AWS"])
LETTER = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["Hi."], sign_off="Best,")
LETTER_V2 = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["Hello again."], sign_off="Best,")
FIT = FitAssessment(score=70, verdict="Good match")


def review(resume_score: int, letter_score: int, revision: int = 0) -> ApplicationReview:
    return ApplicationReview(
        resume_score=resume_score,
        resume_feedback=["Lead with Python"] if resume_score < 80 else [],
        cover_letter_score=letter_score,
        cover_letter_feedback=["Cite the mission"] if letter_score < 80 else [],
        revision=revision,
    )


def base_state(**extra) -> AgentState:
    state: AgentState = {
        "parsed_resume": RESUME.model_dump(),
        "job_details": JOB.model_dump(),
        "company_research": RESEARCH.model_dump(),
        "tailored_resume": TAILORED.model_dump(),
        "cover_letter": LETTER.model_dump(),
        "fit_assessment": FIT.model_dump(),
    }
    state.update(extra)  # type: ignore[typeddict-item]
    return state


# --- First-pass nodes ---


def test_parse_resume_node_skips_when_already_parsed():
    assert parse_resume_node({"parsed_resume": RESUME.model_dump()}) == {"completed_steps": ["parse_resume"]}


def test_parse_resume_node_errors_without_path():
    assert "no resume provided" in parse_resume_node({})["errors"][0]


def test_scrape_job_node_errors_without_input():
    assert "no job URL or description" in scrape_job_node({})["errors"][0]


@patch("app.graph.nodes.scrape_job_posting")
@patch("app.graph.nodes.parse_job_text", return_value=JOB)
def test_scrape_job_node_prefers_pasted_description(mock_parse_text, mock_scrape):
    result = scrape_job_node({"job_description": "full text", "job_url": "https://x.com/job"})
    mock_parse_text.assert_called_once_with("full text", source_url="https://x.com/job")
    mock_scrape.assert_not_called()
    assert result["job_details"]["company"] == "Acme"


def test_tailor_node_errors_without_inputs():
    assert "missing" in tailor_resume_node({"job_details": JOB.model_dump()})["errors"][0]


@patch("app.graph.nodes.tailor_resume", return_value=TAILORED)
def test_tailor_node_passes_research(mock_tailor):
    result = tailor_resume_node(base_state())
    assert mock_tailor.call_args.args == (RESUME, JOB, RESEARCH)
    assert result["tailored_resume"]["matched_keywords"] == ["Python"]


def test_cover_letter_node_rejects_unknown_tone():
    assert "unknown tone" in write_cover_letter_node(base_state(cover_letter_tone="sarcastic"))["errors"][0]


@patch("app.graph.nodes.generate_cover_letter", return_value=LETTER)
def test_cover_letter_node_passes_tone_and_research(mock_generate):
    result = write_cover_letter_node(base_state(cover_letter_tone="concise"))
    assert mock_generate.call_args.args == (RESUME, JOB, RESEARCH)
    assert mock_generate.call_args.kwargs == {"tone": "concise"}
    assert result["cover_letter"]["greeting"] == "Dear Hiring Manager,"


# --- Review, revise, and routing ---


@patch("app.graph.nodes.review_application", return_value=review(70, 90))
def test_review_node_first_pass(mock_review):
    result = review_application_node(base_state())
    assert mock_review.call_args.args[2] == FIT
    assert result["application_review"]["resume_score"] == 70
    assert "tailored_resume" not in result


@patch("app.graph.nodes.review_application", return_value=review(85, 60, revision=1))
def test_review_node_accepts_better_and_rejects_worse_revisions(_mock_review):
    state = base_state(
        application_review=review(70, 75).model_dump(),
        candidate_tailored_resume=TAILORED_V2.model_dump(),
        candidate_cover_letter=LETTER_V2.model_dump(),
        revision_count=1,
    )
    result = review_application_node(state)

    assert result["tailored_resume"] == TAILORED_V2.model_dump()  # 85 >= 70: accepted
    assert "cover_letter" not in result  # 60 < 75: previous letter kept
    final = ApplicationReview.model_validate(result["application_review"])
    assert (final.resume_score, final.cover_letter_score, final.revision) == (85, 75, 1)
    assert final.cover_letter_feedback == ["Cite the mission"]
    assert final.notes == ["Kept the previous cover letter: the revision scored lower."]
    assert result["candidate_tailored_resume"] is None and result["candidate_cover_letter"] is None


def test_review_node_skips_without_documents():
    assert "Review skipped" in review_application_node(base_state(tailored_resume=None))["errors"][0]


@patch("app.graph.nodes.generate_cover_letter", return_value=LETTER_V2)
@patch("app.graph.nodes.tailor_resume", return_value=TAILORED_V2)
def test_revise_node_reruns_only_weak_documents_with_feedback(mock_tailor, mock_letter):
    result = revise_application_node(base_state(application_review=review(70, 90).model_dump()))

    assert mock_tailor.call_args.kwargs == {"feedback": ["Lead with Python"]}
    mock_letter.assert_not_called()
    assert result["candidate_tailored_resume"] == TAILORED_V2.model_dump()
    assert "candidate_cover_letter" not in result
    assert result["revision_count"] == 1


@patch("app.graph.nodes.generate_cover_letter", side_effect=RuntimeError("rate limit"))
@patch("app.graph.nodes.tailor_resume", return_value=TAILORED_V2)
def test_revise_node_reports_partial_failure(_mock_tailor, _mock_letter):
    result = revise_application_node(base_state(application_review=review(70, 60).model_dump()))
    assert "candidate_tailored_resume" in result
    assert result["errors"] == ["Revision failed (cover_letter): rate limit"]


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({}, END),  # no review: an earlier step failed, nothing to approve
        ({"application_review": review(90, 90).model_dump()}, "human_review"),
        ({"application_review": review(60, 90).model_dump()}, "revise_application"),
        ({"application_review": review(60, 90).model_dump(), "revision_count": 1, "max_revisions": 1},
         "human_review"),
        ({"application_review": review(60, 90).model_dump(), "max_revisions": 0}, "human_review"),
    ],
)
def test_route_after_review(extra, expected):
    assert route_after_review(extra) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [("changes_requested", "revise_application"), ("approved", END), ("rejected", END)],
)
def test_route_after_human(status, expected):
    assert route_after_human({"approval_status": status}) == expected


# --- Whole graph ---

RESUME_EXP = ParsedResume(
    contact=ContactInfo(name="Jane Doe"),
    experience=[Experience(title="Engineer", company="Acme", bullets=["Designed APIs"])],
)
TAILORED_CHANGED = TailoredResume(
    resume=RESUME_EXP,
    bullet_changes=[BulletChange(
        section="experience", entry=0, position=0, original="Built APIs", rewritten="Designed APIs"
    )],
)

FIRST_PASS = {
    "app.graph.nodes.load_resume_file": (b"data", "r.pdf"),
    "app.graph.nodes.extract_text": "resume text",
    "app.graph.nodes.structure_resume": RESUME,
    "app.graph.nodes.scrape_job_posting": JOB,
    "app.graph.nodes.research_company": RESEARCH,
    "app.graph.nodes.assess_fit": FIT,
}


@pytest.fixture
def agent():
    """Start a run with LLM-backed steps mocked; patches stay active for resumes."""
    with contextlib.ExitStack() as stack:
        for target, value in FIRST_PASS.items():
            stack.enter_context(patch(target, return_value=value))

        def start(reviews, tailors, letters, **inputs):
            mocks = SimpleNamespace(
                review=stack.enter_context(
                    patch("app.graph.nodes.review_application", side_effect=reviews)
                ),
                tailor=stack.enter_context(patch("app.graph.nodes.tailor_resume", side_effect=tailors)),
                letter=stack.enter_context(
                    patch("app.graph.nodes.generate_cover_letter", side_effect=letters)
                ),
            )
            graph, config = build_graph(), new_run_config()
            graph.invoke({"resume_path": "r.pdf", "job_url": "https://x.com/job", **inputs}, config)
            return graph, config, mocks

        yield start


def test_graph_pauses_for_approval_after_review(agent):
    graph, config, mocks = agent([review(90, 90)], [TAILORED], [LETTER])

    assert awaiting_approval(graph, config)
    state = current_state(graph, config)
    assert state["fit_assessment"]["score"] == 70
    assert state["resume_warnings"] == []
    assert state["application_review"]["resume_score"] == 90
    assert "approval_status" not in state
    assert sorted(state["completed_steps"]) == sorted([
        "parse_resume", "scrape_job", "research_company",
        "tailor_resume", "write_cover_letter", "assess_fit", "review_application",
    ])
    assert mocks.review.call_count == mocks.tailor.call_count == mocks.letter.call_count == 1


def test_approve_finishes_the_run(agent):
    graph, config, _ = agent([review(90, 90)], [TAILORED], [LETTER])
    graph.invoke(resume_command(HumanDecision(action="approve")), config)

    state = current_state(graph, config)
    assert not awaiting_approval(graph, config)
    assert state["approval_status"] == "approved"
    assert state["completed_steps"][-1] == "human_review"
    assert state["tailored_resume"] == TAILORED.model_dump()


def test_approve_applies_reverted_bullets_and_letter_edits(agent):
    graph, config, _ = agent([review(90, 90)], [TAILORED_CHANGED], [LETTER])
    decision = HumanDecision(
        action="approve", reverted_bullets=[0], cover_letter_text="First para.\n\n  Second para.  "
    )
    graph.invoke(resume_command(decision), config)

    state = current_state(graph, config)
    tailored = TailoredResume.model_validate(state["tailored_resume"])
    assert tailored.resume.experience[0].bullets == ["Built APIs"]
    assert tailored.bullet_changes == []
    letter = CoverLetter.model_validate(state["cover_letter"])
    assert letter.paragraphs == ["First para.", "Second para."]
    assert letter.word_count == 4


def test_request_changes_revises_with_user_feedback_and_pauses_again(agent):
    # The reviewer scores the user's version lower (70 < 90) — it is still applied.
    graph, config, mocks = agent([review(90, 90), review(70, 90)], [TAILORED, TAILORED_V2], [LETTER])
    decision = HumanDecision(action="request_changes", resume_feedback="Mention FastAPI first")
    graph.invoke(resume_command(decision), config)

    assert awaiting_approval(graph, config)
    state = current_state(graph, config)
    assert mocks.tailor.call_args.kwargs == {"feedback": ["Mention FastAPI first"]}
    assert mocks.letter.call_count == 1  # no letter feedback, so no new letter
    assert state["tailored_resume"] == TAILORED_V2.model_dump()
    assert state["application_review"]["resume_score"] == 70
    assert state["application_review"]["notes"] == ["Applied the changes you requested."]
    assert state["human_feedback"] is None and state["user_requested_revision"] is False
    assert state["completed_steps"][-3:] == ["human_review", "revise_application", "review_application"]


def test_reject_ends_the_run(agent):
    graph, config, _ = agent([review(90, 90)], [TAILORED], [LETTER])
    graph.invoke(resume_command(HumanDecision(action="reject")), config)
    assert current_state(graph, config)["approval_status"] == "rejected"
    assert not awaiting_approval(graph, config)


def test_graph_revises_weak_resume_before_asking(agent):
    graph, config, mocks = agent(
        [review(70, 90), review(85, 90)], [TAILORED, TAILORED_V2], [LETTER], max_revisions=2
    )

    state = current_state(graph, config)
    assert awaiting_approval(graph, config)
    assert mocks.review.call_count == 2 and mocks.tailor.call_count == 2 and mocks.letter.call_count == 1
    assert mocks.tailor.call_args.kwargs == {"feedback": ["Lead with Python"]}
    assert state["tailored_resume"] == TAILORED_V2.model_dump()
    assert state["revision_count"] == 1 and state["application_review"]["revision"] == 1


def test_graph_respects_revision_budget(agent):
    graph, config, mocks = agent(
        [review(60, 90), review(65, 90)], [TAILORED, TAILORED_V2], [LETTER], max_revisions=1
    )
    assert awaiting_approval(graph, config)
    assert mocks.review.call_count == 2 and mocks.tailor.call_count == 2
    assert current_state(graph, config)["revision_count"] == 1


@patch("app.graph.nodes.scrape_job_posting", side_effect=RuntimeError("site down"))
@patch("app.graph.nodes.load_resume_file", side_effect=RuntimeError("disk error"))
def test_graph_ends_without_approval_when_steps_fail(*_mocks):
    graph, config = build_graph(), new_run_config()
    graph.invoke({"resume_path": "r.pdf", "job_url": "https://x.com/job"}, config)

    assert not awaiting_approval(graph, config)
    errors = current_state(graph, config)["errors"]
    for expected in ("disk error", "site down", "no job details", "Tailoring skipped",
                     "Cover letter skipped", "Fit assessment skipped", "Review skipped"):
        assert any(expected in e for e in errors), expected
    assert len(errors) == 7


def test_request_changes_requires_feedback():
    with pytest.raises(ValueError, match="Describe the changes"):
        HumanDecision(action="request_changes", resume_feedback="  ")
