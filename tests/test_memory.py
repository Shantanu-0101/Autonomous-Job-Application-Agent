from datetime import UTC, datetime, timedelta

import pytest

from app.config import settings
from app.llm.models import CompanyResearch, JobPosting
from app.memory.history import check_history, record_application
from app.memory.job_memory import JobMemory, get_job_memory, job_text
from app.memory.tracker import Tracker, get_tracker, normalize_job_url
from tests.conftest import fake_embed

BACKEND = JobPosting(
    title="Backend Engineer", company="Globex", description="Build Python APIs on AWS",
    requirements=["Python", "FastAPI", "PostgreSQL"], source_url="https://boards.io/globex/1?gh_src=abc",
)
MARKETING = JobPosting(
    title="Marketing Manager", company="Globex", description="Run brand campaigns",
    requirements=["SEO", "Social media"], source_url="https://boards.io/globex/2",
)


@pytest.fixture
def tracker(tmp_path) -> Tracker:
    return Tracker(tmp_path / "t.sqlite")


@pytest.fixture
def memory(tmp_path) -> JobMemory:
    return JobMemory(tmp_path / "index", fake_embed)


# --- URL keys ---


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://boards.io/globex/1?gh_src=abc", "http://www.Boards.io/globex/1/"),
        ("https://jobs.lever.co/acme/x?utm_source=linkedin&lever-source=LinkedIn", "https://jobs.lever.co/acme/x"),
        ("https://acme.io/careers?id=7&ref=feed", "https://acme.io/careers?id=7"),
    ],
)
def test_same_posting_same_key(a, b):
    assert normalize_job_url(a) == normalize_job_url(b)


def test_different_job_ids_differ():
    assert normalize_job_url("https://acme.io/careers?id=7") != normalize_job_url("https://acme.io/careers?id=8")
    assert normalize_job_url("not a url") == ""


# --- Tracker ---


def test_record_then_find_by_url_or_role(tracker):
    record = tracker.record(BACKEND, "ready", fit_score=72)
    assert (record.id, record.status, record.fit_score) == (1, "ready", 72)
    assert tracker.find("http://boards.io/globex/1/") == record
    assert tracker.find(company="GLOBEX", title="backend engineer!") == record
    assert tracker.find("https://boards.io/globex/9", "Globex", "Data Engineer") is None


def test_recording_the_same_job_updates_it(tracker):
    first = tracker.record(BACKEND, "ready", fit_score=72)
    second = tracker.record(BACKEND, "sent", submitted_to="hr@globex.io")
    assert second.id == first.id
    assert (second.status, second.fit_score, second.submitted_to) == ("sent", 72, "hr@globex.io")
    assert len(tracker.list()) == 1


def test_update_status_and_list(tracker):
    backend = tracker.record(BACKEND, "sent")
    tracker.record(MARKETING, "skipped")
    updated = tracker.update_status(backend.id, "interviewing", notes="Round 1 on Monday")
    assert (updated.status, updated.notes) == ("interviewing", "Round 1 on Monday")
    assert [r.title for r in tracker.list(status="interviewing")] == ["Backend Engineer"]
    assert len(tracker.list()) == 2


def test_update_status_rejects_bad_input(tracker):
    record = tracker.record(BACKEND, "ready")
    with pytest.raises(ValueError, match="Unknown status"):
        tracker.update_status(record.id, "hired")
    with pytest.raises(KeyError):
        tracker.update_status(999, "applied")


def test_tracker_persists_across_instances(tmp_path):
    Tracker(tmp_path / "t.sqlite").record(BACKEND, "sent")
    assert Tracker(tmp_path / "t.sqlite").find(company="Globex", title="Backend Engineer") is not None


def test_research_cache_expires(tracker, monkeypatch):
    research = CompanyResearch(name="Globex", summary="Makes APIs")
    tracker.save_research("Globex Inc", research)
    assert tracker.cached_research("globex inc", max_age_days=14) == research
    assert tracker.cached_research("Globex Inc", max_age_days=0) is None

    import app.memory.tracker as tracker_module
    later = datetime.now(UTC) + timedelta(days=15)
    monkeypatch.setattr(tracker_module, "datetime", type("FakeDatetime", (datetime,), {
        "now": classmethod(lambda cls, tz=None: later),
    }))
    assert tracker.cached_research("Globex Inc", max_age_days=14) is None


def test_get_tracker_uses_configured_path():
    assert str(get_tracker().path) == settings.tracker_db_path


# --- Semantic memory ---


def test_memory_finds_similar_postings_and_persists(memory, tmp_path):
    memory.add(1, job_text(BACKEND))
    memory.add(2, job_text(MARKETING))
    reworded = BACKEND.model_copy(update={"title": "Software Engineer, Backend"})

    results = memory.search(job_text(reworded), k=2)
    assert [i for i, _ in results] == [1, 2]
    assert results[0][1] > results[1][1]

    reloaded = JobMemory(tmp_path / "index", fake_embed)
    assert len(reloaded) == 2 and reloaded.search(job_text(reworded), k=1)[0][0] == 1


def test_memory_re_adding_replaces_vector(memory):
    memory.add(1, job_text(BACKEND))
    memory.add(1, job_text(MARKETING))
    assert len(memory) == 1
    assert memory.search(job_text(MARKETING), k=1)[0][0] == 1


def test_empty_memory_and_min_score(memory):
    assert memory.search("anything") == []
    memory.add(1, job_text(BACKEND))
    assert memory.search(job_text(MARKETING), min_score=0.99) == []


def test_get_job_memory_degrades_when_model_unavailable(monkeypatch):
    import app.memory.job_memory as job_memory

    def offline(*_args):
        raise OSError("no network")

    monkeypatch.setattr(job_memory, "_memory_at", offline)
    assert get_job_memory() is None


# --- History checks ---


def test_new_job_has_no_history(tracker, memory):
    check = check_history(BACKEND, tracker, memory)
    assert check.duplicate is None and not check.blocked and check.message == ""


def test_submitted_job_is_blocked(tracker, memory):
    record_application(BACKEND, "sent", tracker, memory)
    repost = BACKEND.model_copy(update={"source_url": "https://boards.io/globex/1?utm_source=x"})
    check = check_history(repost, tracker, memory)
    assert check.blocked and check.duplicate is not None
    assert check.message.startswith("You already applied to Backend Engineer at Globex (status: sent,")


def test_skipped_job_warns_but_does_not_block(tracker, memory):
    record_application(BACKEND, "skipped", tracker, memory)
    check = check_history(BACKEND, tracker, memory)
    assert not check.blocked
    assert check.message.startswith("You looked at Backend Engineer at Globex (status: skipped,")


def test_reposted_job_is_a_possible_duplicate(tracker, memory, monkeypatch):
    monkeypatch.setattr(settings, "duplicate_job_threshold", 0.8)
    record_application(BACKEND, "sent", tracker, memory)
    repost = BACKEND.model_copy(update={"title": "Software Engineer - Backend", "source_url": "https://other.io/9"})

    check = check_history(repost, tracker, memory)

    assert check.duplicate is None and not check.blocked
    assert check.possible_duplicate is not None
    assert check.possible_duplicate.application.title == "Backend Engineer"
    assert 'This looks like "Backend Engineer" at Globex, which you marked sent' in check.message


def test_similar_jobs_at_other_companies_are_listed(tracker, memory, monkeypatch):
    monkeypatch.setattr(settings, "similar_job_threshold", 0.5)
    record_application(BACKEND, "applied", tracker, memory)
    elsewhere = BACKEND.model_copy(update={"company": "Initech", "source_url": "https://initech.io/1"})

    check = check_history(elsewhere, tracker, memory)

    assert check.possible_duplicate is None
    assert [s.application.company for s in check.similar] == ["Globex"]


def test_history_works_without_semantic_memory(tracker):
    record_application(BACKEND, "sent", tracker, None)
    assert check_history(BACKEND, tracker, None).blocked
