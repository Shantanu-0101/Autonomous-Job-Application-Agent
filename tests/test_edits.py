import pytest

from app.generators.edits import edit_cover_letter, revert_bullets
from app.llm.models import (
    BulletChange,
    ContactInfo,
    CoverLetter,
    Experience,
    ParsedResume,
    Project,
    TailoredResume,
)

TAILORED = TailoredResume(
    resume=ParsedResume(
        contact=ContactInfo(name="Jane"),
        experience=[Experience(title="Eng", company="Acme", bullets=["New A", "New B"])],
        projects=[Project(name="Bot", bullets=["New C"])],
    ),
    bullet_changes=[
        BulletChange(section="experience", entry=0, position=0, original="Old A", rewritten="New A"),
        BulletChange(section="experience", entry=0, position=1, original="Old B", rewritten="New B"),
        BulletChange(section="projects", entry=0, position=0, original="Old C", rewritten="New C"),
    ],
)
LETTER = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["One.", "Two."], sign_off="Best,",
                     word_count=2, warnings=["Paragraph 2: check this"])


def test_revert_bullets_restores_selected_originals_only():
    result = revert_bullets(TAILORED, [0, 2, 9])
    assert result.resume.experience[0].bullets == ["Old A", "New B"]
    assert result.resume.projects[0].bullets == ["Old C"]
    assert [c.rewritten for c in result.bullet_changes] == ["New B"]
    assert TAILORED.resume.experience[0].bullets == ["New A", "New B"]  # input untouched


def test_edit_cover_letter_splits_paragraphs_and_flags_edit():
    edited = edit_cover_letter(LETTER, "One, rewritten.\n\n\n  Two.  \n")
    assert edited.paragraphs == ["One, rewritten.", "Two."]
    assert edited.word_count == 3
    assert edited.warnings == ["You edited this letter. Automatic fact-checks only covered the earlier draft."]


def test_unchanged_cover_letter_keeps_its_checks():
    assert edit_cover_letter(LETTER, "One.\n\nTwo.") is LETTER


def test_empty_cover_letter_edit_rejected():
    with pytest.raises(ValueError, match="empty"):
        edit_cover_letter(LETTER, "  \n\n ")
