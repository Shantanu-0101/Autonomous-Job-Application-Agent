import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class LLMHealthCheck(BaseModel):
    """Verifies structured output works."""

    status: str = Field(description="Should be 'ok'")
    provider: str = Field(description="Which LLM provider responded")
    message: str = Field(description="A short greeting")


class JobPosting(BaseModel):
    """Structured representation of a job posting."""

    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(default="", description="Job location or 'Remote'")
    description: str = Field(description="Full job description summary")
    requirements: list[str] = Field(
        default_factory=list,
        description="Required skills, qualifications, and experience",
    )
    nice_to_have: list[str] = Field(
        default_factory=list,
        description="Preferred or bonus qualifications",
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Key job responsibilities and duties",
    )
    salary_range: str = Field(
        default="Not specified",
        description="Salary range if mentioned",
    )
    application_method: str = Field(
        default="",
        description="How to apply: email address, application URL, or 'form'",
    )
    source_url: str = Field(default="", description="Original job posting URL")


class CompanyResearch(BaseModel):
    """Structured research brief about a company."""

    name: str = Field(description="Company name")
    industry: str = Field(default="", description="Primary industry or sector")
    mission: str = Field(default="", description="Company mission statement or purpose")
    values: list[str] = Field(
        default_factory=list,
        description="Core company values or cultural principles",
    )
    recent_news: list[str] = Field(
        default_factory=list,
        description="Recent news, announcements, or notable events",
    )
    culture_notes: str = Field(
        default="",
        description="Summary of work culture, employee reviews, work environment",
    )
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Technologies, tools, and frameworks the company uses",
    )
    size: str = Field(
        default="",
        description="Company size: number of employees or scale (startup, mid-size, enterprise)",
    )
    headquarters: str = Field(default="", description="HQ location")
    summary: str = Field(
        default="",
        description="2-3 sentence overview of the company for use in a cover letter",
    )


class ContactInfo(BaseModel):
    name: str = Field(default="", description="Candidate's full name")
    email: str = Field(default="", description="Email address")
    phone: str = Field(default="", description="Phone number as written")
    location: str = Field(default="", description="City, state/country as written")
    linkedin: str = Field(default="", description="LinkedIn profile URL")
    github: str = Field(default="", description="GitHub profile URL")
    portfolio: str = Field(default="", description="Personal website or portfolio URL")
    other_links: list[str] = Field(
        default_factory=list,
        description="Any other profile URLs (LeetCode, Kaggle, Medium, etc.)",
    )


class Experience(BaseModel):
    title: str = Field(description="Job title or role")
    company: str = Field(description="Employer name")
    location: str = Field(default="", description="Work location if stated")
    start_date: str = Field(default="", description="Start date as written, e.g. 'Jan 2023'")
    end_date: str = Field(default="", description="End date as written, or 'Present' if current")
    employment_type: str = Field(
        default="", description="Full-time, Internship, Contract, Part-time, etc. if stated"
    )
    bullets: list[str] = Field(
        default_factory=list,
        description="Each responsibility/achievement bullet, copied verbatim",
    )


class Education(BaseModel):
    institution: str = Field(description="School, college, or university name")
    degree: str = Field(default="", description="Degree, e.g. 'B.Tech', 'Master of Science'")
    field_of_study: str = Field(default="", description="Major or specialization")
    location: str = Field(default="", description="City or campus if stated")
    start_date: str = Field(default="", description="Start date as written")
    end_date: str = Field(default="", description="End or expected graduation date as written")
    grade: str = Field(default="", description="GPA, CGPA, or percentage if stated")
    highlights: list[str] = Field(
        default_factory=list, description="Relevant coursework, honors, or activities"
    )


class Project(BaseModel):
    name: str = Field(description="Project name")
    description: str = Field(default="", description="One-line description if given")
    technologies: list[str] = Field(default_factory=list, description="Technologies used")
    link: str = Field(default="", description="Repository or demo URL if given")
    bullets: list[str] = Field(
        default_factory=list, description="Each project bullet, copied verbatim"
    )


class ParsedResume(BaseModel):
    """Structured representation of a candidate's resume."""

    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = Field(default="", description="Professional summary or objective, verbatim")
    skills: list[str] = Field(
        default_factory=list,
        description="Every individual skill, tool, language, or framework listed",
    )
    experience: list[Experience] = Field(
        default_factory=list, description="Work experience, most recent first"
    )
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(
        default_factory=list, description="Awards, honors, publications, competition results"
    )
    languages: list[str] = Field(
        default_factory=list, description="Spoken languages, with proficiency if stated"
    )


class RewrittenBullet(BaseModel):
    source: int = Field(description="Index of the original bullet (within this entry) being rewritten")
    text: str = Field(description="The rewritten bullet")


class TailoredEntry(BaseModel):
    index: int = Field(description="Index of the experience or project in the original resume")
    bullets: list[RewrittenBullet] = Field(
        description="Rewritten bullets for this entry, most relevant first"
    )


class TailoringResult(BaseModel):
    """LLM output: rewritten content that references original resume entries by index."""

    summary: str = Field(description="2-3 sentence professional summary targeted at this job")
    skills: list[str] = Field(
        description="Skills from the original resume, most relevant to the job first"
    )
    experience: list[TailoredEntry] = Field(default_factory=list)
    projects: list[TailoredEntry] = Field(default_factory=list)
    matched_keywords: list[str] = Field(
        default_factory=list,
        description="Job requirements and keywords the resume genuinely demonstrates",
    )
    missing_keywords: list[str] = Field(
        default_factory=list,
        description="Important job requirements the resume shows no evidence of",
    )
    changes: list[str] = Field(
        default_factory=list, description="Short notes on what was changed and why"
    )


class BulletChange(BaseModel):
    """An accepted bullet rewrite, kept so reviewers can see before/after."""

    section: Literal["experience", "projects"]
    entry: int
    position: int
    original: str
    rewritten: str


class TailoredResume(BaseModel):
    resume: ParsedResume
    bullet_changes: list[BulletChange] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Rewrites that were rejected by fabrication checks",
    )


class ClaimCheck(BaseModel):
    id: int = Field(description="The item id being checked")
    supported: bool = Field(description="True only if every claim is backed by the original")
    issue: str = Field(default="", description="What is unsupported, if anything")


class FactCheckResult(BaseModel):
    # Required (no default) so strict JSON-schema mode can't return an empty object.
    checks: list[ClaimCheck] = Field(description="Exactly one check for every item id")


Tone = Literal["professional", "enthusiastic", "concise"]


class CoverLetterDraft(BaseModel):
    """LLM output for a cover letter, before fact-checking."""

    greeting: str = Field(description='e.g. "Dear Hiring Manager,"')
    paragraphs: list[str] = Field(description="Body paragraphs, in order")
    sign_off: str = Field(default="Sincerely,", description='e.g. "Sincerely,"')


class CoverLetter(BaseModel):
    greeting: str
    paragraphs: list[str]
    sign_off: str
    signature: str = Field(default="", description="Candidate's name")
    tone: Tone = "professional"
    word_count: int = 0
    warnings: list[str] = Field(
        default_factory=list,
        description="Problems the fact-checker could not resolve; review before sending",
    )


RequirementStatus = Literal["met", "partial", "not_met", "unknown"]
RequirementKind = Literal["skill_or_experience", "eligibility", "personal_trait"]


class RequirementCheck(BaseModel):
    id: str = Field(description="The requirement id, e.g. R1 or N2")
    kind: RequirementKind = Field(
        description="eligibility = work authorization, citizenship, clearance, location, or start "
        "date; personal_trait = motivation, attitude, or values; otherwise skill_or_experience"
    )
    status: RequirementStatus = Field(
        description="met, partial, not_met, or unknown (the resume cannot answer it)"
    )
    evidence: str = Field(description="Resume fact supporting the status, or why it is unmet/unknown")


class FitCheckResult(BaseModel):
    """LLM output: one verdict per requirement id."""

    checks: list[RequirementCheck] = Field(description="Exactly one check for every requirement id")


class RequirementMatch(BaseModel):
    requirement: str
    importance: Literal["required", "nice_to_have"]
    kind: RequirementKind = "skill_or_experience"
    status: RequirementStatus
    evidence: str = ""


class FitAssessment(BaseModel):
    """How well the candidate's real background matches the job. Rewriting cannot change this."""

    score: int | None = Field(description="0-100, computed from requirement statuses; None if unscorable")
    verdict: str
    requirements: list[RequirementMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ApplicationReviewResult(BaseModel):
    """LLM output: how well the documents present the candidate's real qualifications."""

    resume_score: int = Field(ge=0, le=100, description="Presentation quality of the tailored resume")
    resume_feedback: list[str] = Field(
        description="Up to 4 specific fixes using only facts in the original resume; empty if none"
    )
    cover_letter_score: int = Field(ge=0, le=100, description="Quality of the cover letter")
    cover_letter_feedback: list[str] = Field(
        description="Up to 4 specific fixes using only facts in the original resume; empty if none"
    )


class ApplicationReview(ApplicationReviewResult):
    revision: int = Field(default=0, description="How many revision rounds produced these documents")
    notes: list[str] = Field(default_factory=list, description="Decisions made during revision")


class HumanDecision(BaseModel):
    """What the user decided at the approval step. Sent to resume a paused run."""

    action: Literal["approve", "request_changes", "reject"]
    resume_feedback: str = ""
    cover_letter_feedback: str = ""
    # Applied on approve: indices into TailoredResume.bullet_changes to undo
    reverted_bullets: list[int] = Field(default_factory=list)
    # Applied on approve: the user's edited cover letter body, paragraphs separated by blank lines
    cover_letter_text: str | None = None
    # On approve: email the application to this address. The user confirms it explicitly.
    send_to: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "HumanDecision":
        if self.action == "request_changes" and not (
            self.resume_feedback.strip() or self.cover_letter_feedback.strip()
        ):
            raise ValueError("Describe the changes you want to the resume or the cover letter.")
        if self.send_to is not None:
            self.send_to = self.send_to.strip()
            if self.action != "approve":
                raise ValueError("Only an approved application can be sent.")
            if not EMAIL_PATTERN.fullmatch(self.send_to):
                raise ValueError(f'"{self.send_to}" is not a valid email address.')
        return self


class Submission(BaseModel):
    method: Literal["email", "manual"]
    status: Literal["sent", "failed", "ready"]  # ready = the user applies on the job's own page
    recipient: str = ""
    apply_url: str = ""
    message_id: str = ""
    sent_at: str = ""
    error: str = ""
