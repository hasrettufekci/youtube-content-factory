"""
storage/analytics_timing_cache.py tests - YouTube Analytics Timing V1's
simple TTL cache. Mirrors tests/test_publish_schedules_storage.py's own
CRUD/migration test pattern.

Covers spec scenarios 5 (cache hit), 6 (cache expired), 7 (cache refresh -
the invalidate-then-refetch behavior itself; the CLI's --refresh flag is
covered in tests/test_analytics_timing_demo.py).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone

from storage import analytics_timing_cache as cache


def test_make_cache_key_is_deterministic_and_distinguishes_inputs():
    a = cache.make_cache_key("english", "Europe/Istanbul", 90)
    b = cache.make_cache_key("english", "Europe/Istanbul", 90)
    c = cache.make_cache_key("turkish", "Europe/Istanbul", 90)
    assert a == b
    assert a != c
    assert cache.make_cache_key(None, "Europe/Istanbul", 90) == "default:Europe/Istanbul:90"


def test_get_fresh_returns_none_when_nothing_cached(isolated_storage):
    assert cache.get_fresh("no-such-key") is None


# ---------------------------------------------------------------------------
# 5 - cache hit
# ---------------------------------------------------------------------------

def test_set_then_get_fresh_returns_the_cached_result(isolated_storage):
    key = cache.make_cache_key("english", "Europe/Istanbul", 90)
    slots = [{"weekday": 5, "hour": 20, "minute": 0, "timezone": "Europe/Istanbul",
              "score": 0.8, "sample_size": 12, "source": "inferred_performance"}]
    cache.set_cache(
        cache_key=key, channel_key="english", timezone_name="Europe/Istanbul", lookback_days=90,
        source="inferred_performance", sample_size=12, confidence="medium", slots=slots, ttl_hours=24,
    )
    result = cache.get_fresh(key)
    assert result is not None
    assert result["source"] == "inferred_performance"
    assert result["sample_size"] == 12
    assert result["confidence"] == "medium"
    assert result["slots"] == slots


# ---------------------------------------------------------------------------
# 6 - cache expired
# ---------------------------------------------------------------------------

def test_get_fresh_returns_none_after_ttl_expires(isolated_storage):
    key = cache.make_cache_key("english", "Europe/Istanbul", 90)
    fetched_at = datetime(2026, 9, 1, 12, 0, tzinfo=dt_timezone.utc)
    cache.set_cache(
        cache_key=key, channel_key="english", timezone_name="Europe/Istanbul", lookback_days=90,
        source="inferred_performance", sample_size=10, confidence="medium", slots=[], ttl_hours=24,
        fetched_at=fetched_at,
    )
    just_before_expiry = fetched_at + timedelta(hours=23, minutes=59)
    just_after_expiry = fetched_at + timedelta(hours=24, minutes=1)
    assert cache.get_fresh(key, now_dt=just_before_expiry) is not None
    assert cache.get_fresh(key, now_dt=just_after_expiry) is None


def test_set_cache_upserts_replacing_the_previous_entry_for_the_same_key(isolated_storage):
    key = cache.make_cache_key("english", "Europe/Istanbul", 90)
    cache.set_cache(cache_key=key, channel_key="english", timezone_name="Europe/Istanbul", lookback_days=90,
                     source="inferred_performance", sample_size=5, confidence="low", slots=[], ttl_hours=24)
    cache.set_cache(cache_key=key, channel_key="english", timezone_name="Europe/Istanbul", lookback_days=90,
                     source="inferred_performance", sample_size=15, confidence="medium", slots=[], ttl_hours=24)
    with cache.connect() as conn:
        rows = conn.execute("SELECT * FROM analytics_timing_cache WHERE cache_key = ?", (key,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["sample_size"] == 15


def test_different_cache_keys_coexist_independently(isolated_storage):
    cache.set_cache(cache_key="a", channel_key="english", timezone_name="Europe/Istanbul", lookback_days=90,
                     source="inferred_performance", sample_size=5, confidence="low", slots=[], ttl_hours=24)
    cache.set_cache(cache_key="b", channel_key="turkish", timezone_name="Europe/Istanbul", lookback_days=90,
                     source="inferred_performance", sample_size=8, confidence="low", slots=[], ttl_hours=24)
    assert cache.get_fresh("a")["sample_size"] == 5
    assert cache.get_fresh("b")["sample_size"] == 8


def test_clear_cache_removes_all_rows(isolated_storage):
    cache.set_cache(cache_key="a", channel_key=None, timezone_name="Europe/Istanbul", lookback_days=90,
                     source="inferred_performance", sample_size=5, confidence="low", slots=[], ttl_hours=24)
    cache.clear_cache()
    assert cache.get_fresh("a") is None


def test_init_db_is_idempotent_on_a_brand_new_db(tmp_path):
    db_path = tmp_path / "fresh.db"
    cache.init_db(db_path)
    cache.init_db(db_path)
    cache.init_db(db_path)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(analytics_timing_cache)").fetchall()]
    finally:
        conn.close()
    assert columns.count("cache_key") == 1


def test_unavailable_result_can_be_cached_with_empty_slots(isolated_storage):
    """An 'unavailable' outcome is just as cacheable as a real one - see
    module docstring on why re-deriving 'unavailable' on every approval
    within the TTL window is exactly what caching should prevent too."""
    key = cache.make_cache_key(None, "Europe/Istanbul", 90)
    cache.set_cache(cache_key=key, channel_key=None, timezone_name="Europe/Istanbul", lookback_days=90,
                     source="unavailable", sample_size=0, confidence="insufficient", slots=[], ttl_hours=24)
    result = cache.get_fresh(key)
    assert result["source"] == "unavailable"
    assert result["slots"] == []
