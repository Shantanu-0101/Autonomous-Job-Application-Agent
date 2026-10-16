from unittest.mock import patch

import pytest
from langgraph.graph import END

from app.graph.builder import build_graph
from app.graph.edges import route_after_review
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
    CompanyResearch,
    ContactInfo,
    CoverLetter,
    FitAssessment,
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
        ({}, END),  # no review yet
        ({"application_review": review(90, 90).model_dump()}, END),
        ({"application_review": review(60, 90).model_dump()}, "revise_application"),
        ({"application_review": review(60, 90).model_dump(), "revision_count": 1, "max_revisions": 1}, END),
        ({"application_review": review(60, 90).model_dump(), "max_revisions": 0}, END),
    ],
)
def test_route_after_review(extra, expected):
    assert route_after_review(extra) == expected


# --- Whole graph ---

FIRST_PASS_PATCHES = [
    patch("app.graph.nodes.load_resume_file", return_value=(b"data", "r.pdf")),
    patch("app.graph.nodes.extract_text", return_value="resume text"),
    patch("app.graph.nodes.structure_resume", return_value=RESUME),
    patch("app.graph.nodes.scrape_job_posting", return_value=JOB),
    patch("app.graph.nodes.research_company", return_value=RESEARCH),
    patch("app.graph.nodes.assess_fit", return_value=FIT),
]


def run_graph(review_results, tailor_results, letter_results, **inputs):
    for p in FIRST_PASS_PATCHES:
        p.start()
    try:
        with patch("app.graph.nodes.review_application", side_effect=review_results) as mock_review, \
             patch("app.graph.nodes.tailor_resume", side_effect=tailor_results) as mock_tailor, \
             patch("app.graph.nodes.generate_cover_letter", side_effect=letter_results) as mock_letter:
            result = build_graph().invoke(
                {"resume_path": "r.pdf", "job_url": "https://x.com/job", **inputs}
            )
            return result, mock_review, mock_tailor, mock_letter
    finally:
        for p in FIRST_PASS_PATCHES:
            p.stop()


def test_graph_ends_after_review_when_documents_pass():
    result, mock_review, mock_tailor, mock_letter = run_graph(
        [review(90, 90)], [TAILORED], [LETTER]
    )

    assert result["fit_assessment"]["score"] == 70
    assert result["application_review"]["resume_score"] == 90
    assert sorted(result["completed_steps"]) == sorted([
        "parse_resume", "scrape_job", "research_company",
        "tailor_resume", "write_cover_letter", "assess_fit", "review_application",
    ])
    assert mock_review.call_count == mock_tailor.call_count == mock_letter.call_count == 1
    assert not result.get("errors")


def test_graph_revises_weak_resume_then_stops():
    result, mock_review, mock_tailor, mock_letter = run_graph(
        [review(70, 90), review(85, 90)], [TAILORED, TAILORED_V2], [LETTER], max_revisions=2
    )

    assert mock_review.call_count == 2 and mock_tailor.call_count == 2 and mock_letter.call_count == 1
    assert mock_tailor.call_args.kwargs == {"feedback": ["Lead with Python"]}
    assert result["tailored_resume"] == TAILORED_V2.model_dump()
    assert result["revision_count"] == 1
    assert result["application_review"]["revision"] == 1
    assert result["completed_steps"][-2:] == ["revise_application", "review_application"]


def test_graph_respects_revision_budget():
    result, mock_review, mock_tailor, _ = run_graph(
        [review(60, 90), review(65, 90)], [TAILORED, TAILORED_V2], [LETTER], max_revisions=1
    )
    assert mock_review.call_count == 2 and mock_tailor.call_count == 2
    assert result["revision_count"] == 1


@patch("app.graph.nodes.scrape_job_posting", side_effect=RuntimeError("site down"))
@patch("app.graph.nodes.load_resume_file", side_effect=RuntimeError("disk error"))
def test_graph_collects_errors_from_parallel_failures(*_mocks):
    result = build_graph().invoke({"resume_path": "r.pdf", "job_url": "https://x.com/job"})

    errors = result["errors"]
    for expected in ("disk error", "site down", "no job details", "Tailoring skipped",
                     "Cover letter skipped", "Fit assessment skipped", "Review skipped"):
        assert any(expected in e for e in errors), expected
    assert len(errors) == 7
