"""Tests for storage/database.py — SQLite operations."""

from datetime import datetime

from screenmind.storage.models import ScreenshotEntry, ActivityRecord, DailySummary


def test_insert_and_get_activity(db):
    entry = ScreenshotEntry(
        timestamp=datetime(2026, 5, 16, 10, 30, 0),
        screenshot_path="/tmp/test.jpg",
        window_title="VS Code",
        detected_app_name="Code",
        bookmarked=False,
        analyzed=False,
    )
    activity_id = db.insert_activity(entry)
    assert activity_id is not None
    assert activity_id > 0

    # Retrieve it
    activity = db.get_activity_by_id(activity_id)
    assert activity is not None
    assert activity["window_title"] == "VS Code"
    assert activity["detected_app"] == "Code"


def test_get_activities_by_date(db):
    for hour in range(3):
        entry = ScreenshotEntry(
            timestamp=datetime(2026, 5, 16, 10 + hour, 0, 0),
            screenshot_path=f"/tmp/test_{hour}.jpg",
            analyzed=False,
        )
        db.insert_activity(entry)

    activities = db.get_activities_by_date("2026-05-16")
    assert len(activities) == 3


def test_get_activities_empty_date(db):
    activities = db.get_activities_by_date("2099-01-01")
    assert activities == []


def test_toggle_bookmark(db):
    entry = ScreenshotEntry(
        timestamp=datetime(2026, 5, 16, 12, 0, 0),
        screenshot_path="/tmp/bm.jpg",
        bookmarked=False,
        analyzed=False,
    )
    aid = db.insert_activity(entry)

    # Toggle on
    new_state = db.toggle_bookmark(aid)
    assert new_state is True

    # Toggle off
    new_state = db.toggle_bookmark(aid)
    assert new_state is False


def test_get_bookmarks(db):
    for i in range(3):
        entry = ScreenshotEntry(
            timestamp=datetime(2026, 5, 16, 10 + i, 0, 0),
            screenshot_path=f"/tmp/bm_{i}.jpg",
            bookmarked=(i == 1),  # Only middle one bookmarked
            analyzed=False,
        )
        db.insert_activity(entry)

    bookmarks = db.get_bookmarks()
    assert len(bookmarks) == 1


def test_update_activity_analysis(db):
    entry = ScreenshotEntry(
        timestamp=datetime(2026, 5, 16, 14, 0, 0),
        screenshot_path="/tmp/analysis.jpg",
        analyzed=False,
    )
    aid = db.insert_activity(entry)

    analysis = ActivityRecord(
        app_name="Chrome",
        activity_category="browsing",
        activity_summary="Reading docs",
        mood="learning",
        confidence=0.85,
    )
    db.update_activity_analysis(aid, analysis)

    activity = db.get_activity_by_id(aid)
    assert activity["app_name"] == "Chrome"
    assert activity["category"] == "browsing"
    assert activity["summary"] == "Reading docs"
    assert activity["analyzed"] == 1


def test_upsert_daily_summary(db):
    summary = DailySummary(
        date="2026-05-16",
        summary="Great day",
        total_activities=10,
    )
    db.upsert_daily_summary(summary)

    result = db.get_daily_summary("2026-05-16")
    assert result is not None
    assert result["summary"] == "Great day"

    # Upsert again (update)
    summary.summary = "Updated summary"
    db.upsert_daily_summary(summary)
    result = db.get_daily_summary("2026-05-16")
    assert result["summary"] == "Updated summary"


def test_delete_by_date(db):
    for i in range(5):
        entry = ScreenshotEntry(
            timestamp=datetime(2026, 5, 16, 10 + i, 0, 0),
            screenshot_path=f"/tmp/del_{i}.jpg",
            analyzed=False,
        )
        db.insert_activity(entry)

    deleted = db.delete_by_date("2026-05-16")
    assert deleted == 5

    activities = db.get_activities_by_date("2026-05-16")
    assert len(activities) == 0


def test_get_stats(db):
    for i, cat in enumerate(["coding", "coding", "browsing"]):
        entry = ScreenshotEntry(
            timestamp=datetime(2026, 5, 16, 10 + i, 0, 0),
            screenshot_path=f"/tmp/stat_{i}.jpg",
            analyzed=False,
        )
        aid = db.insert_activity(entry)
        analysis = ActivityRecord(
            app_name="App",
            activity_category=cat,
            activity_summary="test",
        )
        db.update_activity_analysis(aid, analysis)

    stats = db.get_stats("2026-05-16", "2026-05-16")
    assert stats["total_activities"] == 3


def _analyzed_activity(db, hour=10, scene=""):
    """Insert an analyzed activity; return its id."""
    aid = db.insert_activity(ScreenshotEntry(
        timestamp=datetime(2026, 5, 16, hour, 0, 0),
        screenshot_path=f"/tmp/scene_{hour}.jpg",
        analyzed=True,
    ))
    db.update_activity_analysis(
        aid,
        ActivityRecord(
            app_name="Chrome",
            activity_category="browsing",
            activity_summary="Reading docs",
            scene_description=scene,
        ),
        embedding=[0.5] * 384,
        ocr_text="some screen text",
    )
    return aid


def test_update_scene_description_sets_field(db):
    aid = _analyzed_activity(db, scene="")
    db.update_scene_description(aid, "A browser showing documentation")
    activity = db.get_activity_by_id(aid)
    assert activity["scene_description"] == "A browser showing documentation"


def test_update_scene_description_preserves_other_fields(db):
    """Backfill must touch only the scene — summary/category/app stay intact."""
    aid = _analyzed_activity(db, scene="")
    db.update_scene_description(aid, "New scene")
    activity = db.get_activity_by_id(aid)
    assert activity["app_name"] == "Chrome"
    assert activity["category"] == "browsing"
    assert activity["summary"] == "Reading docs"
    assert activity["ocr_text"] == "some screen text"
    assert activity["analyzed"] == 1


def test_update_scene_description_keeps_embedding_when_none(db):
    """embedding=None (embedder unavailable) must not wipe the existing vector."""
    aid = _analyzed_activity(db, scene="")
    db.update_scene_description(aid, "New scene", embedding=None)
    conn = db._get_conn()
    blob = conn.execute(
        "SELECT embedding FROM activities WHERE id = ?", (aid,)
    ).fetchone()[0]
    assert blob is not None
    assert db._decode_embedding(blob) == [0.5] * 384


def test_update_scene_description_replaces_embedding_when_given(db):
    aid = _analyzed_activity(db, scene="")
    # 0.25 is exactly representable in float32 (blob storage packs floats)
    db.update_scene_description(aid, "New scene", embedding=[0.25] * 384)
    conn = db._get_conn()
    blob = conn.execute(
        "SELECT embedding FROM activities WHERE id = ?", (aid,)
    ).fetchone()[0]
    assert db._decode_embedding(blob) == [0.25] * 384


def test_update_scene_description_syncs_fts(db):
    """The AFTER UPDATE trigger must index the new scene for keyword search."""
    aid = _analyzed_activity(db, scene="")
    db.update_scene_description(aid, "uniquekeyword_xyzzy in the scene")
    conn = db._get_conn()
    hit = conn.execute(
        "SELECT rowid FROM activities_fts WHERE activities_fts MATCH 'uniquekeyword_xyzzy'"
    ).fetchall()
    assert [r[0] for r in hit] == [aid]


def _seed_day(db, hours):
    """Insert activities at given hours on 2026-05-16; return ids in order."""
    return [
        db.insert_activity(ScreenshotEntry(
            timestamp=datetime(2026, 5, 16, h, 0, 0),
            screenshot_path=f"/tmp/day_{h}.jpg",
            analyzed=False,
        ))
        for h in hours
    ]


def test_day_number_chronological_within_day(db):
    """day_number counts events 1..N in chronological order, newest row first."""
    _seed_day(db, [9, 10, 11])
    acts = db.get_activities_by_date("2026-05-16")
    assert len(acts) == 3
    # Rows come back DESC by time; numbers must still be 1,2,3 chronologically
    assert [a["day_number"] for a in acts] == [3, 2, 1]


def test_day_number_scoped_to_date(db):
    """Numbers restart at 1 for each day."""
    _seed_day(db, [9, 10])
    db.insert_activity(ScreenshotEntry(
        timestamp=datetime(2026, 5, 17, 9, 0, 0),
        screenshot_path="/tmp/next.jpg",
        analyzed=False,
    ))
    acts = db.get_activities_by_date("2026-05-17")
    assert [a["day_number"] for a in acts] == [1]


def test_day_number_stable_across_pages(db):
    """Pagination must not shift numbers — computed over the whole day."""
    _seed_day(db, [8, 9, 10, 11])
    page2 = db.get_activities_by_date("2026-05-16", limit=2, offset=2)
    assert [a["day_number"] for a in page2] == [2, 1]


def test_count_activities_by_date(db):
    _seed_day(db, [9, 10, 11])
    assert db.count_activities_by_date("2026-05-16") == 3
    assert db.count_activities_by_date("2099-01-01") == 0


def test_get_day_number(db):
    ids = _seed_day(db, [9, 10, 11])
    assert db.get_day_number(ids[0]) == 1
    assert db.get_day_number(ids[2]) == 3


def test_get_day_number_missing_activity(db):
    assert db.get_day_number(99999) is None
