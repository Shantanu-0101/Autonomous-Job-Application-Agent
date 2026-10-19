"""
Job Application Agent — Standalone UI Walkthrough
==================================================
This is a self-contained Streamlit file that demonstrates all the
UI components used in the project. No API keys, LLM providers,
or project imports needed.

Usage:
    pip install streamlit pandas
    streamlit run Job_Application_Agent_StreamlitApp.py
"""

import hmac
import time

import pandas as pd
import streamlit as st


# ============================================================
# Constants (same as the real app)
# ============================================================

STATUSES = ("ready", "sent", "skipped", "applied", "interviewing", "offer", "rejected", "withdrawn")

STATUS_LABELS = {
    "met": ":green[Met]",
    "partial": ":orange[Partial]",
    "not_met": ":red[Missing]",
    "unknown": ":gray[Not on resume]",
}

STEP_LABELS = {
    "parse_resume": "Read your resume",
    "scrape_job": "Read the job posting",
    "check_history": "Checked your application history",
    "research_company": "Researched the company",
    "tailor_resume": "Tailored your resume",
    "write_cover_letter": "Wrote the cover letter",
    "assess_fit": "Assessed your fit",
    "review_application": "Reviewed the application",
}


# ============================================================
# 1. PASSWORD GATE
# Uses hmac.compare_digest for timing-safe comparison.
# In the real app, the password comes from APP_ACCESS_PASSWORD env var.
# ============================================================

def access_gate() -> bool:
    if st.session_state.get("authenticated"):
        return True
    with st.form("login"):
        attempt = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open")
    if submitted:
        expected = "demo"
        if hmac.compare_digest(attempt.encode(), expected.encode()):
            st.session_state["authenticated"] = True
            st.rerun()
        time.sleep(1)
        st.error("Wrong password.")
    return False


# ============================================================
# 2. INPUT SCREEN
# File uploader + job URL/text tabs + settings sidebar.
# In the real app, the uploaded file is saved, parsed via LLM,
# and the pipeline starts on button click.
# ============================================================

def input_screen() -> None:
    st.subheader("1. Your resume")
    upload = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

    st.subheader("2. The job")
    url_tab, text_tab = st.tabs(["Job link", "Paste the description"])
    with url_tab:
        job_url = st.text_input("Job posting URL", placeholder="https://job-boards.greenhouse.io/...")
        st.caption(
            "Greenhouse, Lever, and Ashby links work best. LinkedIn and Naukri block automated "
            "reading — paste the description instead."
        )
    with text_tab:
        job_text = st.text_area("Job description", height=250)

    ready = upload is not None and bool(job_url.strip() or job_text.strip())
    if st.button("Tailor my application", type="primary", disabled=not ready):
        simulate_progress("Working on your application — this takes 1-3 minutes")
        st.session_state["demo_stage"] = "review"
        st.rerun()


# ============================================================
# 3. PROGRESS DISPLAY
# st.status with expanding step list. In the real app, each
# step is yielded by LangGraph's stream_steps() as it completes.
# ============================================================

def simulate_progress(label: str) -> None:
    with st.status(label, expanded=True) as status:
        for step_label in STEP_LABELS.values():
            time.sleep(0.3)
            st.write(f"✓ {step_label}")
        status.update(label="Done", state="complete", expanded=False)


# ============================================================
# 4. FIT ASSESSMENT SECTION
# Metric card + requirement checklist in expander.
# In the real app, data comes from FitAssessment Pydantic model.
# ============================================================

def fit_section() -> None:
    score_col, text_col = st.columns([1, 3])
    score_col.metric("Fit", "—/100")
    text_col.markdown("**Verdict appears here**")
    text_col.caption(
        "Based on which requirements your resume shows. Rewriting can't change this score — "
        "only real experience can."
    )
    with st.expander("Requirement checklist"):
        st.caption("Each job requirement is shown with its match status (Met / Partial / Missing / Not on resume).")
        for status_key, label in STATUS_LABELS.items():
            st.markdown(f"{label} — Example requirement text")


# ============================================================
# 5. REVIEW SCREEN — TABS
# Four tabs: Tailored Resume, Changes, Cover Letter, Job & Company.
# This is the core interactive screen where users review and edit.
# ============================================================

def review_screen() -> None:
    st.header("Role Title at Company Name")
    fit_section()

    st.caption("AI review — resume —/100, cover letter —/100")

    resume_t, changes_t, letter_t, job_t = st.tabs([
        "Tailored resume",
        "Changes (—)",
        "Cover letter",
        "Job & company",
    ])

    # --- Tab 1: Tailored Resume ---
    with resume_t:
        st.markdown("*The full tailored resume renders here in markdown.*")
        st.markdown("**Matched:** keywords that appear in both resume and job")
        st.markdown("**Missing (not added to your resume):** keywords the job needs but the resume lacks")
        with st.expander("Fact-check notes"):
            st.caption("Any warnings from the fact-checker appear here (e.g. reverted fabricated claims).")

    # --- Tab 2: Before/After Changes ---
    with changes_t:
        st.caption("Tick any rewrite you'd rather undo. Your original wording will be used instead.")
        for i in range(2):
            before_col, after_col = st.columns(2)
            before_col.markdown(f"**Before**\n\nOriginal bullet text #{i+1}")
            after_col.markdown(f"**After**\n\nRewritten bullet text #{i+1}")
            st.checkbox("Keep my original", key=f"revert_{i}")
            st.divider()

    # --- Tab 3: Editable Cover Letter ---
    with letter_t:
        st.caption("Greeting … Sign-off, Name")
        st.text_area(
            "Cover letter (edit freely — paragraphs are separated by blank lines)",
            value="",
            height=400,
            key="letter_edit",
            placeholder="The generated cover letter appears here. Users can edit it freely before approving.",
        )

    # --- Tab 4: Job & Company Research ---
    with job_t:
        st.markdown("**Role Title** at **Company Name** — Location")
        st.caption("Salary: range or 'Not specified'")
        st.markdown("**Requirements**\n- Requirement 1\n- Requirement 2")
        st.markdown("**Nice to have**\n- Nice-to-have 1")
        st.markdown("**Responsibilities**\n- Responsibility 1")
        st.markdown("**About Company**\n\nCompany research summary from Tavily web search.")
        st.markdown("**Recent news**\n- News item 1\n- News item 2")

    # --- Decision Buttons ---
    decision_section()


# ============================================================
# 6. DECISION SECTION
# Approve / Reject buttons + Request Changes form.
# In the real app, these create a HumanDecision and resume
# the LangGraph pipeline via interrupt().
# ============================================================

def decision_section() -> None:
    st.subheader("Your decision")

    # Email opt-in (shown when the job has an application email)
    st.checkbox("Also email my application to careers@example.com", key="send_email")
    if st.session_state.get("send_email"):
        st.text_input(
            "Recipient (check it against the job posting)",
            value="careers@example.com",
            key="send_to",
        )
        st.caption("Your resume and cover letter PDFs will be sent. This can't be undone.")

    approve_col, reject_col = st.columns(2)
    if approve_col.button("Approve", type="primary"):
        st.session_state["demo_stage"] = "finished_approved"
        st.rerun()
    if reject_col.button("Reject — don't apply"):
        st.session_state["demo_stage"] = "finished_rejected"
        st.rerun()

    with st.form("request_changes"):
        st.markdown("**Or ask for changes**")
        st.text_area("What should change in the resume?", height=80)
        st.text_area("What should change in the cover letter?", height=80)
        if st.form_submit_button("Revise"):
            simulate_progress("Revising — this takes about a minute")
            st.rerun()


# ============================================================
# 7. FINISHED SCREEN
# Download buttons for PDF/DOCX exports.
# In the real app, these call resume_pdf(), resume_docx(),
# cover_letter_pdf(), cover_letter_docx() from export.py.
# ============================================================

def finished_screen() -> None:
    st.header("Role Title at Company Name")
    fit_section()

    status = st.session_state.get("demo_stage")
    if status == "finished_rejected":
        st.info("You chose not to apply to this job.")
        return

    st.success("Approved. Download your documents and apply on the job's page.")
    st.link_button("Open the application page", "https://example.com")

    resume_col, letter_col = st.columns(2)
    resume_col.download_button(
        "Resume (PDF)", b"placeholder", "Resume.pdf", "application/pdf"
    )
    resume_col.download_button(
        "Resume (Word)", b"placeholder", "Resume.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    letter_col.download_button(
        "Cover letter (PDF)", b"placeholder", "Cover_Letter.pdf", "application/pdf"
    )
    letter_col.download_button(
        "Cover letter (Word)", b"placeholder", "Cover_Letter.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ============================================================
# 8. APPLICATION HISTORY SCREEN
# Editable data table with status dropdown.
# In the real app, data comes from SQLite via Tracker class.
# ============================================================

def applications_screen() -> None:
    st.info(
        "This screen shows all past applications from the SQLite tracker. "
        "Users can update Status and Notes directly — changes save automatically."
    )

    counts = {"Sent": 0, "Applied": 0, "Interviewing": 0, "Offer": 0}
    for col, (label, count) in zip(st.columns(len(counts)), counts.items()):
        col.metric(label, count)

    st.caption("Update the status and notes as you hear back. Changes save automatically.")

    table = pd.DataFrame(columns=["Company", "Role", "Status", "Fit", "Updated", "Notes", "Link"])
    st.data_editor(
        table,
        key="applications_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["Company", "Role", "Fit", "Updated", "Link"],
        column_config={
            "Status": st.column_config.SelectboxColumn(options=list(STATUSES), required=True),
            "Link": st.column_config.LinkColumn(display_text="Open"),
        },
    )


# ============================================================
# 9. SIDEBAR — Settings
# Tone selector, revision slider, LLM provider info,
# email check button, start-over button.
# ============================================================

def sidebar() -> None:
    with st.sidebar:
        st.header("Settings")
        st.selectbox("Cover letter tone", ["professional", "enthusiastic", "concise"], key="tone")
        st.slider(
            "Automatic revision rounds", 0, 3, value=1, key="max_revisions",
            help="More rounds can improve the documents but take longer and use more of your LLM quota.",
        )
        st.caption("LLM provider: set via LLM_PROVIDER env var")
        if st.button("Check email settings"):
            st.info("In the real app, this verifies SMTP credentials without sending.")
        if st.session_state.get("demo_stage") and st.button("Start over"):
            st.session_state.pop("demo_stage", None)
            st.rerun()


# ============================================================
# MAIN — Page config, navigation, screen routing
# ============================================================

def main() -> None:
    st.set_page_config(page_title="Job Application Agent", layout="wide")
    st.title("Job Application Agent")
    st.caption(
        "Tailors your resume and cover letter to a job using only facts from your resume. "
        "Nothing is sent without your approval."
    )

    if not access_gate():
        return

    sidebar()

    view = st.radio(
        "View", ["New application", "My applications"], key="view",
        horizontal=True, label_visibility="collapsed",
    )

    if view == "My applications":
        applications_screen()
        return

    stage = st.session_state.get("demo_stage")
    if stage == "review":
        review_screen()
    elif stage in ("finished_approved", "finished_rejected"):
        finished_screen()
    else:
        input_screen()


main()
