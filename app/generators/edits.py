import re

from app.llm.models import CoverLetter, TailoredResume


def revert_bullets(tailored: TailoredResume, indices: list[int]) -> TailoredResume:
    """Restore the original wording of the given bullet changes."""
    result = tailored.model_copy(deep=True)
    undo = {i for i in indices if 0 <= i < len(result.bullet_changes)}
    kept = []
    for i, change in enumerate(result.bullet_changes):
        if i not in undo:
            kept.append(change)
            continue
        entries = result.resume.experience if change.section == "experience" else result.resume.projects
        entries[change.entry].bullets[change.position] = change.original
    result.bullet_changes = kept
    return result


def edit_cover_letter(letter: CoverLetter, body: str) -> CoverLetter:
    """Replace the letter body with the user's edit. Paragraphs are separated by blank lines."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        raise ValueError("The edited cover letter is empty.")
    if paragraphs == letter.paragraphs:
        return letter
    return letter.model_copy(update={
        "paragraphs": paragraphs,
        "word_count": sum(len(p.split()) for p in paragraphs),
        # The user takes responsibility for their own wording; earlier checks no longer apply.
        "warnings": ["You edited this letter. Automatic fact-checks only covered the earlier draft."],
    })
