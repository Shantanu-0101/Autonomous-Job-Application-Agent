from unittest.mock import patch

import pytest

from app.generators.cover_letter import (
    build_prompt,
    find_issues,
    generate_cover_letter,
    remove_unsupported_numbers,
)
from app.generators.render import render_cover_letter
from app.llm.models import (
    ClaimCheck,
    CompanyResearch,
    ContactInfo,
    CoverLetter,
    CoverLetterDraft,
    Experience,
    FactCheckResult,
    JobPosting,
    ParsedResume,
)

RESUME = ParsedResume(
    contact=ContactInfo(name="Jane Doe"),
    skills=["Python", "FastAPI"],
    experience=[
        Experience(
            title="Software Engineer",
            company="Acme",
            bullets=["Built REST APIs serving 2M requests per day"],
        )
    ],
)
JOB = JobPosting(title="Backend Engineer", company="Globex", description="Build APIs")
RESEARCH = CompanyResearch(name="Globex", mission="Make APIs delightful", recent_news=["Raised $40M"])

GOOD = [
    "I'm applying for the Backend Engineer role at Globex.",
    "At Acme I built REST APIs serving 2M requests per day.",
    "Globex's mission to make APIs delightful is why I want to join.",
]


def draft(paragraphs: list[str]) -> CoverLetterDraft:
    return CoverLetterDraft(greeting="Dear Hiring Manager,", paragraphs=paragraphs, sign_off="Best,")


def verdicts(*supported: bool, issue: str = "") -> FactCheckResult:
    return FactCheckResult(checks=[
        ClaimCheck(id=k, supported=ok, issue="" if ok else issue)
        for k, ok in enumerate(supported, start=1)
    ])


def prompts(mock_invoke) -> list[str]:
    return [call[0][1] for call in mock_invoke.call_args_list]


# --- Prompt ---


def test_prompt_marks_inputs_as_data_and_sets_tone():
    prompt = build_prompt(RESUME, JOB, RESEARCH, tone="concise")
    for tag in ("<resume>", "<job_posting>", "<company_notes>"):
        assert tag in prompt
    assert "direct and brief" in prompt and "150-220 words" in prompt
    assert "Raised $40M" in prompt
    assert "fact-checker rejected" not in prompt


def test_prompt_includes_feedback():
    prompt = build_prompt(RESUME, JOB, feedback=["Paragraph 2: invents Kubernetes"])
    assert "fact-checker rejected" in prompt and "- Paragraph 2: invents Kubernetes" in prompt


# --- Issue detection ---


@patch("app.generators.cover_letter.invoke_structured", return_value=verdicts(True, True, True))
def test_find_issues_passes_supported_letter(_mock_invoke):
    assert find_issues(GOOD, RESUME, JOB, RESEARCH, "professional") == []


@patch("app.generators.cover_letter.invoke_structured", return_value=verdicts(True, True))
def test_find_issues_flags_unknown_numbers_and_missing_verdicts(_mock_invoke):
    paragraphs = ["Hi.", "I cut costs by 35%.", "I raised $40M? No — Globex did."]
    issues = find_issues(paragraphs, RESUME, JOB, RESEARCH, "professional")
    assert any("Paragraph 2 uses numbers" in i and "35" in i for i in issues)
    assert not any("Paragraph 3 uses numbers" in i for i in issues)  # 40 comes from research
    assert "Paragraph 3 could not be verified." in issues


@patch("app.generators.cover_letter.invoke_structured")
def test_find_issues_reports_fact_check_verdicts_and_length(mock_invoke):
    mock_invoke.return_value = verdicts(False, issue="claims Kubernetes experience")
    issues = find_issues(["word " * 300], RESUME, JOB, None, "concise")
    assert "Paragraph 1: claims Kubernetes experience" in issues
    assert any("300 words" in i for i in issues)


@patch("app.generators.cover_letter.invoke_structured", side_effect=RuntimeError("rate limit"))
def test_find_issues_reports_failed_fact_check(_mock_invoke):
    issues = find_issues(GOOD, RESUME, JOB, RESEARCH, "professional")
    assert issues == ["Automatic fact-check could not run. Review every claim before sending."]


# --- Generation flow ---


@patch("app.generators.cover_letter.invoke_structured")
def test_generate_accepts_clean_first_draft(mock_invoke):
    mock_invoke.side_effect = [draft(GOOD), verdicts(True, True, True)]

    letter = generate_cover_letter(RESUME, JOB, RESEARCH)

    assert mock_invoke.call_count == 2
    assert letter.paragraphs == GOOD
    assert letter.signature == "Jane Doe"
    assert letter.sign_off == "Best,"
    assert letter.word_count == sum(len(p.split()) for p in GOOD)
    assert letter.warnings == []


@patch("app.generators.cover_letter.invoke_structured")
def test_generate_revises_with_fact_check_feedback(mock_invoke):
    bad = GOOD[:1] + ["I led Kubernetes migrations at Acme."]
    mock_invoke.side_effect = [
        draft(bad), verdicts(True, False, issue="invents Kubernetes"),
        draft(GOOD), verdicts(True, True, True),
    ]

    letter = generate_cover_letter(RESUME, JOB, RESEARCH)

    assert mock_invoke.call_count == 4
    assert "- Paragraph 2: invents Kubernetes" in prompts(mock_invoke)[2]
    assert letter.paragraphs == GOOD
    assert letter.warnings == []


@patch("app.generators.cover_letter.invoke_structured")
def test_generate_keeps_unresolved_issues_as_warnings(mock_invoke):
    bad = ["I led Kubernetes migrations at Acme."]
    mock_invoke.side_effect = [
        draft(bad), verdicts(False, issue="invents Kubernetes"),
        draft(bad), verdicts(False, issue="still invents Kubernetes"),
    ]

    letter = generate_cover_letter(RESUME, JOB)

    assert letter.warnings == ["Paragraph 1: still invents Kubernetes"]


@patch("app.generators.cover_letter.invoke_structured")
def test_generate_drops_blank_paragraphs_and_rejects_empty_letter(mock_invoke):
    mock_invoke.side_effect = [draft(["  ", GOOD[0]]), verdicts(True)]
    assert generate_cover_letter(RESUME, JOB).paragraphs == [GOOD[0]]

    mock_invoke.side_effect = [draft(["", "  "])]
    with pytest.raises(ValueError, match="empty"):
        generate_cover_letter(RESUME, JOB)


def test_render_cover_letter():
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["One.", "Two."],
                         sign_off="Sincerely,", signature="Jane Doe")
    assert render_cover_letter(letter) == (
        "Dear Hiring Manager,\n\nOne.\n\nTwo.\n\nSincerely,\nJane Doe\n"
    )


def test_remove_unsupported_numbers_drops_only_offending_sentences():
    paragraphs = [
        "I cut costs by 30%. I also improved latency by 33 ms. I love APIs!",
        "Only 99 problems here.",
        "Globex raised $5.4 billion.",
    ]
    cleaned, removed = remove_unsupported_numbers(paragraphs, known={"30", "5.4"})
    assert cleaned == ["I cut costs by 30%. I love APIs!", "Globex raised $5.4 billion."]
    assert removed == ["I also improved latency by 33 ms.", "Only 99 problems here."]


@patch("app.generators.cover_letter.invoke_structured")
def test_generate_removes_sentences_with_invented_numbers_after_revision(mock_invoke):
    bad = [GOOD[0], "At Acme I built REST APIs serving 2M requests per day. I cut latency by 33 ms."]
    mock_invoke.side_effect = [
        draft(bad), verdicts(True, True),
        draft(bad), verdicts(True, True),
    ]

    letter = generate_cover_letter(RESUME, JOB, RESEARCH)

    assert letter.paragraphs == [GOOD[0], "At Acme I built REST APIs serving 2M requests per day."]
    assert letter.warnings == [
        'Removed a sentence with a number not found in your resume or the job posting: '
        '"I cut latency by 33 ms."'
    ]
