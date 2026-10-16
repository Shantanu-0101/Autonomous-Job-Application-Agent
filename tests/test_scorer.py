from typing import Literal
from unittest.mock import patch

import pytest

from app.evaluator.scorer import (
    assess_fit,
    documents_to_revise,
    fit_score,
    fit_verdict,
    review_application,
)
from app.llm.models import (
    ApplicationReview,
    ApplicationReviewResult,
    ContactInfo,
    CoverLetter,
    Experience,
    FitAssessment,
    FitCheckResult,
    JobPosting,
    ParsedResume,
    RequirementCheck,
    RequirementKind,
    RequirementMatch,
    RequirementStatus,
    TailoredResume,
)

RESUME = ParsedResume(
    contact=ContactInfo(name="Jane Doe"),
    skills=["Python", "AWS"],
    experience=[Experience(title="Engineer", company="Acme", bullets=["Built Python APIs on AWS"])],
)
JOB = JobPosting(
    title="Backend Engineer",
    company="Globex",
    description="Build APIs",
    requirements=["Python", "Go", "US work authorization"],
    nice_to_have=["AWS"],
    responsibilities=["Ship features"],
)


def match(
    status: RequirementStatus, importance: Literal["required", "nice_to_have"] = "required"
) -> RequirementMatch:
    return RequirementMatch(requirement="x", importance=importance, status=status)


def check(
    id_: str,
    status: RequirementStatus,
    evidence: str = "",
    kind: RequirementKind = "skill_or_experience",
) -> RequirementCheck:
    return RequirementCheck(id=id_, kind=kind, status=status, evidence=evidence)


# --- Score maths ---


def test_fit_score_weights_required_double_and_ignores_unknown():
    # required: met (2) + not_met (0) of 4; nice: partial (0.5) of 1; unknown excluded
    reqs = [match("met"), match("not_met"), match("unknown"), match("partial", "nice_to_have")]
    assert fit_score(reqs) == round(100 * 2.5 / 5)


def test_fit_score_none_when_nothing_scorable():
    assert fit_score([]) is None
    assert fit_score([match("unknown")]) is None


@pytest.mark.parametrize(
    ("score", "verdict"),
    [(None, "Not enough information"), (90, "Strong match"), (75, "Strong match"),
     (60, "Good match"), (40, "Stretch"), (10, "Long shot")],
)
def test_fit_verdict_thresholds(score, verdict):
    assert fit_verdict(score) == verdict


# --- Fit assessment ---


@patch("app.evaluator.scorer.invoke_structured")
def test_assess_fit_maps_checks_to_requirements(mock_invoke):
    mock_invoke.return_value = FitCheckResult(checks=[
        check("R1", "met", "Built Python APIs"),
        check("R2", "not_met", "No Go experience"),
        check("R3", "unknown", "Resume does not state work authorization", kind="eligibility"),
        check("N1", "met", "AWS listed"),
    ])

    fit = assess_fit(RESUME, JOB)

    assert [(r.requirement, r.importance, r.kind, r.status) for r in fit.requirements] == [
        ("Python", "required", "skill_or_experience", "met"),
        ("Go", "required", "skill_or_experience", "not_met"),
        ("US work authorization", "required", "eligibility", "unknown"),
        ("AWS", "nice_to_have", "skill_or_experience", "met"),
    ]
    assert fit.score == 60  # (2 + 0 + 1) / (2 + 2 + 1)
    assert fit.verdict == "Good match"
    assert fit.warnings == ['Confirm before applying: "US work authorization"']
    prompt = mock_invoke.call_args[0][1]
    assert '"id": "N1"' in prompt and "<resume>" in prompt


@patch("app.evaluator.scorer.invoke_structured")
def test_unknown_traits_and_optional_eligibility_do_not_warn(mock_invoke):
    job = JOB.model_copy(update={
        "requirements": ["Passion for real-world impact"], "nice_to_have": ["Based in Austin"]
    })
    mock_invoke.return_value = FitCheckResult(checks=[
        check("R1", "unknown", kind="personal_trait"),
        check("N1", "unknown", kind="eligibility"),
    ])
    fit = assess_fit(RESUME, job)
    assert fit.warnings == [] and fit.score is None


@patch("app.evaluator.scorer.invoke_structured")
def test_assess_fit_warns_about_unassessed_requirements(mock_invoke):
    mock_invoke.return_value = FitCheckResult(checks=[check("R1", "met")])
    fit = assess_fit(RESUME, JOB)
    assert len(fit.requirements) == 1 and fit.score == 100
    assert len(fit.warnings) == 3 and 'Could not assess requirement: "Go"' in fit.warnings


@patch("app.evaluator.scorer.invoke_structured")
def test_assess_fit_falls_back_to_responsibilities(mock_invoke):
    mock_invoke.return_value = FitCheckResult(checks=[check("R1", "partial")])
    job = JOB.model_copy(update={"requirements": [], "nice_to_have": []})
    fit = assess_fit(RESUME, job)
    assert fit.requirements[0].requirement == "Ship features"


@patch("app.evaluator.scorer.invoke_structured")
def test_assess_fit_without_any_requirements_skips_llm(mock_invoke):
    job = JOB.model_copy(update={"requirements": [], "nice_to_have": [], "responsibilities": []})
    fit = assess_fit(RESUME, job)
    assert fit.score is None and fit.warnings
    mock_invoke.assert_not_called()


# --- Application review ---


@patch("app.evaluator.scorer.invoke_structured")
def test_review_cleans_and_caps_feedback(mock_invoke):
    mock_invoke.return_value = ApplicationReviewResult(
        resume_score=70,
        resume_feedback=["Lead with AWS", " ", "a", "b", "c", "d"],
        cover_letter_score=90,
        cover_letter_feedback=[],
    )
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["Hello Globex."], sign_off="Best,")
    fit = FitAssessment(score=60, verdict="Good match")

    review = review_application(RESUME, JOB, fit, TailoredResume(resume=RESUME), letter)

    assert review.resume_feedback == ["Lead with AWS", "a", "b", "c"]
    assert review.cover_letter_score == 90
    prompt = mock_invoke.call_args[0][1]
    assert "# Jane Doe" in prompt and "Hello Globex." in prompt and '"score": 60' in prompt


def test_review_scores_are_bounded():
    with pytest.raises(ValueError):
        ApplicationReviewResult(
            resume_score=120, resume_feedback=[], cover_letter_score=50, cover_letter_feedback=[]
        )


@pytest.mark.parametrize(
    ("resume_score", "resume_fb", "letter_score", "letter_fb", "expected"),
    [
        (60, ["fix"], 60, ["fix"], ["resume", "cover_letter"]),
        (60, [], 60, ["fix"], ["cover_letter"]),  # low score but nothing actionable
        (85, ["fix"], 79, ["fix"], ["cover_letter"]),
        (80, ["fix"], 95, [], []),
    ],
)
def test_documents_to_revise(resume_score, resume_fb, letter_score, letter_fb, expected):
    review = ApplicationReview(
        resume_score=resume_score, resume_feedback=resume_fb,
        cover_letter_score=letter_score, cover_letter_feedback=letter_fb,
    )
    assert documents_to_revise(review, threshold=80) == expected
