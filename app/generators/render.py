from app.llm.models import CoverLetter, ParsedResume


def _dates(start: str, end: str) -> str:
    return " – ".join(d for d in (start, end) if d)


def _join(*parts: str, sep: str = " | ") -> str:
    return sep.join(p for p in parts if p)


def render_markdown(resume: ParsedResume) -> str:
    c = resume.contact
    lines = [f"# {c.name}" if c.name else ""]
    lines.append(_join(c.email, c.phone, c.location))
    lines.append(_join(c.linkedin, c.github, c.portfolio, *c.other_links))

    if resume.summary:
        lines += ["", "## Summary", resume.summary]

    if resume.skills:
        lines += ["", "## Skills", ", ".join(resume.skills)]

    if resume.experience:
        lines += ["", "## Experience"]
        for e in resume.experience:
            lines += ["", f"### {e.title} — {e.company}", _join(e.location, _dates(e.start_date, e.end_date))]
            lines += [f"- {b}" for b in e.bullets]

    if resume.projects:
        lines += ["", "## Projects"]
        for p in resume.projects:
            heading = f"### {p.name}" + (f" ({', '.join(p.technologies)})" if p.technologies else "")
            lines += ["", heading, _join(p.description, p.link)]
            lines += [f"- {b}" for b in p.bullets]

    if resume.education:
        lines += ["", "## Education"]
        for ed in resume.education:
            degree = _join(ed.degree, ed.field_of_study, sep=", ")
            lines += ["", f"### {ed.institution}", _join(degree, _dates(ed.start_date, ed.end_date), ed.grade)]
            lines += [f"- {h}" for h in ed.highlights]

    for title, items in (
        ("Certifications", resume.certifications),
        ("Achievements", resume.achievements),
        ("Languages", resume.languages),
    ):
        if items:
            lines += ["", f"## {title}"] + [f"- {i}" for i in items]

    # Drop empty lines produced by missing optional fields, but keep section spacing.
    output: list[str] = []
    for line in lines:
        if line or (output and output[-1]):
            output.append(line)
    return "\n".join(output).strip() + "\n"


def render_cover_letter(letter: CoverLetter) -> str:
    closing = "\n".join(part for part in (letter.sign_off, letter.signature) if part)
    return "\n\n".join([letter.greeting, *letter.paragraphs, closing]) + "\n"
