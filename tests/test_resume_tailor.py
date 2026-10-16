from unittest.mock import patch

from app.generators.render import render_markdown
from app.generators.resume_tailor import (
    apply_tailoring,
    build_prompt,
    fact_check,
    tailor_resume,
)
from app.llm.models import (
    ClaimCheck,
    CompanyResearch,
    ContactInfo,
    Education,
    Experience,
    FactCheckResult,
    JobPosting,
    ParsedResume,
    Project,
    RewrittenBullet,
    TailoredEntry,
    TailoringResult,
)

ORIGINAL = ParsedResume(
    contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
    summary="Backend engineer with 2 years of experience.",
    skills=["Python", "FastAPI", "PostgreSQL", "Photoshop"],
    experience=[
        Experience(
            title="Software Engineer",
            company="Acme",
            start_date="Jan 2023",
            end_date="Present",
            bullets=["Built REST APIs serving 2M requests per day", "Organized team lunches"],
        ),
        Experience(
            title="Intern",
            company="Beta",
            bullets=["Wrote tests that raised coverage from 45% to 78%"],
        ),
    ],
    projects=[Project(name="AutoApply", technologies=["LangGraph"], bullets=["Built an agent"])],
    education=[Education(institution="UT Austin", degree="B.S.")],
)

JOB = JobPosting(
    title="Backend Engineer",
    company="Globex",
    description="Build APIs",
    requirements=["Python", "PostgreSQL", "Kubernetes"],
)

GOOD_REWRITE = "Designed Python REST APIs serving 2M requests per day"


def entry(index: int, *bullets: tuple[int, str]) -> TailoredEntry:
    return TailoredEntry(index=index, bullets=[RewrittenBullet(source=s, text=t) for s, t in bullets])


def result(**overrides) -> TailoringResult:
    data = {
        "summary": "Backend engineer with 2 years of experience building Python APIs.",
        "skills": ["PostgreSQL", "Python", "FastAPI"],
        "experience": [entry(0, (0, GOOD_REWRITE))],
        "matched_keywords": ["Python", "PostgreSQL"],
        "missing_keywords": ["Kubernetes"],
        "changes": ["Reordered skills"],
    }
    data.update(overrides)
    return TailoringResult(**data)


def all_supported(ids: range) -> FactCheckResult:
    return FactCheckResult(checks=[ClaimCheck(id=i, supported=True) for i in ids])


# --- Deterministic merge ---


def test_applies_valid_rewrites_and_keeps_metadata():
    tailored = apply_tailoring(ORIGINAL, result())

    exp = tailored.resume.experience[0]
    assert exp.bullets == [GOOD_REWRITE]  # irrelevant "team lunches" bullet omitted
    assert (exp.title, exp.company, exp.start_date) == ("Software Engineer", "Acme", "Jan 2023")
    assert tailored.resume.experience[1].bullets == ORIGINAL.experience[1].bullets
    assert tailored.resume.skills == ["PostgreSQL", "Python", "FastAPI"]
    assert tailored.resume.summary.endswith("building Python APIs.")
    assert tailored.resume.contact == ORIGINAL.contact
    assert tailored.resume.education == ORIGINAL.education
    assert [(c.entry, c.position, c.rewritten) for c in tailored.bullet_changes] == [(0, 0, GOOD_REWRITE)]
    assert tailored.warnings == []


def test_does_not_mutate_original():
    before = ORIGINAL.model_dump()
    apply_tailoring(ORIGINAL, result())
    assert ORIGINAL.model_dump() == before


def test_keeps_original_wording_when_numbers_change():
    tailored = apply_tailoring(
        ORIGINAL,
        result(experience=[
            entry(1, (0, "Raised test coverage to 78%")),  # dropped 45
            entry(0, (0, "Built REST APIs serving 5M requests per day")),  # changed 2 -> 5
        ]),
    )
    assert tailored.resume.experience[1].bullets == ORIGINAL.experience[1].bullets
    assert tailored.resume.experience[0].bullets == [ORIGINAL.experience[0].bullets[0]]
    assert tailored.bullet_changes == []
    assert len(tailored.warnings) == 2 and "Intern at Beta" in tailored.warnings[0]


def test_reorders_bullets_and_tracks_positions():
    tailored = apply_tailoring(
        ORIGINAL, result(experience=[entry(0, (1, "Organized team events"), (0, GOOD_REWRITE))])
    )
    assert tailored.resume.experience[0].bullets == ["Organized team events", GOOD_REWRITE]
    assert [(c.position, c.original) for c in tailored.bullet_changes] == [
        (0, "Organized team lunches"),
        (1, "Built REST APIs serving 2M requests per day"),
    ]


def test_ignores_invalid_indices_and_duplicates():
    tailored = apply_tailoring(
        ORIGINAL,
        result(
            experience=[
                entry(5, (0, "Made up job")),
                entry(0, (7, "Made up bullet"), (0, "  ")),
                entry(1, (0, "Improved test coverage from 45% to 78%"), (0, "Duplicate source")),
                entry(1, (0, "Second rewrite of entry is ignored")),
            ],
            projects=[entry(0, (0, "Built a LangGraph agent"))],
        ),
    )
    assert len(tailored.resume.experience) == 2
    assert tailored.resume.experience[0].bullets == ORIGINAL.experience[0].bullets
    assert tailored.resume.experience[1].bullets == ["Improved test coverage from 45% to 78%"]
    assert tailored.resume.projects[0].bullets == ["Built a LangGraph agent"]


def test_ignores_invented_skills_and_matches_case():
    tailored = apply_tailoring(ORIGINAL, result(skills=["python", "Kubernetes", "Python"]))
    assert tailored.resume.skills == ["Python"]
    assert any("Kubernetes" in w for w in tailored.warnings)


def test_keeps_original_skills_if_none_survive():
    assert apply_tailoring(ORIGINAL, result(skills=["Kubernetes"])).resume.skills == ORIGINAL.skills


def test_rejects_summary_with_invented_numbers():
    tailored = apply_tailoring(ORIGINAL, result(summary="Engineer with 7 years of experience."))
    assert tailored.resume.summary == ORIGINAL.summary
    assert any("summary" in w for w in tailored.warnings)


# --- LLM fact-check ---


@patch("app.generators.resume_tailor.invoke_structured")
def test_fact_check_reverts_unsupported_bullets_and_summary(mock_invoke):
    tailored = apply_tailoring(
        ORIGINAL,
        result(experience=[entry(0, (0, GOOD_REWRITE), (1, "Led cross-team culture initiatives"))]),
    )
    mock_invoke.return_value = FactCheckResult(checks=[
        ClaimCheck(id=0, supported=True),
        ClaimCheck(id=1, supported=False, issue="adds leadership not in original"),
        ClaimCheck(id=2, supported=False, issue="claims API design expertise"),
    ])

    checked = fact_check(ORIGINAL, tailored)

    assert checked.resume.experience[0].bullets == [GOOD_REWRITE, "Organized team lunches"]
    assert [c.rewritten for c in checked.bullet_changes] == [GOOD_REWRITE]
    assert checked.resume.summary == ORIGINAL.summary
    assert "Software Engineer at Acme: adds leadership" in checked.warnings[0]
    assert "claims API design expertise" in checked.warnings[1]
    assert tailored.resume.summary != ORIGINAL.summary  # input left untouched


@patch("app.generators.resume_tailor.invoke_structured")
def test_fact_check_treats_missing_verdicts_as_unsupported(mock_invoke):
    tailored = apply_tailoring(ORIGINAL, result())
    mock_invoke.return_value = FactCheckResult(checks=[])

    checked = fact_check(ORIGINAL, tailored)

    assert checked.resume.experience[0].bullets == [ORIGINAL.experience[0].bullets[0]]
    assert checked.resume.summary == ORIGINAL.summary
    assert all("could not be verified" in w for w in checked.warnings)


@patch("app.generators.resume_tailor.invoke_structured", side_effect=RuntimeError("rate limited"))
def test_fact_check_failure_keeps_rewrites_with_warning(_mock_invoke):
    tailored = apply_tailoring(ORIGINAL, result())
    checked = fact_check(ORIGINAL, tailored)
    assert checked.resume == tailored.resume
    assert "could not run" in checked.warnings[-1]


@patch("app.generators.resume_tailor.invoke_structured")
def test_fact_check_skips_llm_when_nothing_changed(mock_invoke):
    tailored = apply_tailoring(ORIGINAL, result(summary="", experience=[]))
    assert fact_check(ORIGINAL, tailored) is tailored
    mock_invoke.assert_not_called()


@patch("app.generators.resume_tailor.invoke_structured")
def test_fact_check_prompt_contains_pairs(mock_invoke):
    mock_invoke.return_value = all_supported(range(2))
    fact_check(ORIGINAL, apply_tailoring(ORIGINAL, result()))
    prompt = mock_invoke.call_args[0][1]
    assert GOOD_REWRITE in prompt and '"original": "<full resume>"' in prompt


# --- Prompt and end-to-end ---


def test_prompt_marks_inputs_as_data_and_includes_feedback():
    prompt = build_prompt(
        ORIGINAL, JOB, CompanyResearch(name="Globex", mission="Win"), feedback=["Stress APIs"]
    )
    for tag in ("<resume>", "<job_posting>", "<company_notes>"):
        assert tag in prompt
    assert '"text": "Organized team lunches"' in prompt
    assert "- Stress APIs" in prompt
    assert "Kubernetes" in prompt


def test_prompt_without_feedback_or_research():
    prompt = build_prompt(ORIGINAL, JOB)
    assert "Reviewer feedback" not in prompt
    assert "<company_notes>\n{}\n</company_notes>" in prompt


@patch("app.generators.resume_tailor.invoke_structured")
def test_tailor_resume_rewrites_then_fact_checks(mock_invoke):
    mock_invoke.side_effect = [result(), all_supported(range(2))]
    tailored = tailor_resume(ORIGINAL, JOB)

    schemas = [call[0][0] for call in mock_invoke.call_args_list]
    assert schemas == [TailoringResult, FactCheckResult]
    assert tailored.resume.experience[0].bullets == [GOOD_REWRITE]
    assert tailored.matched_keywords == ["Python", "PostgreSQL"]


def test_render_markdown_includes_all_sections():
    md = render_markdown(ORIGINAL)
    assert md.startswith("# Jane Doe\njane@example.com\n")
    for heading in ("## Summary", "## Skills", "## Experience", "## Projects", "## Education"):
        assert heading in md
    assert "### Software Engineer — Acme\nJan 2023 – Present\n- Built REST APIs" in md
    assert "### AutoApply (LangGraph)" in md
    assert "\n\n\n" not in md


@patch("app.generators.resume_tailor.invoke_structured")
def test_fact_check_gives_entry_context(mock_invoke):
    mock_invoke.return_value = all_supported(range(3))
    tailored = apply_tailoring(
        ORIGINAL, result(projects=[entry(0, (0, "Built a LangGraph agent"))])
    )
    fact_check(ORIGINAL, tailored)
    prompt = mock_invoke.call_args[0][1]
    assert '"context": "Software Engineer at Acme"' in prompt
    assert '"context": "Project AutoApply; technologies: LangGraph"' in prompt
