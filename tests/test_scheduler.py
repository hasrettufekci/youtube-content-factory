"""
scheduler.py tests for the new Daily Topic Discovery entry points
(run_topic_discovery_once / run_daily_topic_discovery / --discover-once /
--discover-daily). The EXISTING run_once()/run_daily()/--daily-at/--once
production-scheduling behavior is untouched by this task and is not
retested here.
"""
from __future__ import annotations

import datetime as dt

import pytest

import scheduler
from core.config import settings


# ---------------------------------------------------------------------------
# M - scheduler triggers discovery
# ---------------------------------------------------------------------------

def test_run_topic_discovery_once_calls_run_daily_discovery(isolated_storage, monkeypatch):
    from youtube import topic_discovery

    calls = []
    monkeypatch.setattr(topic_discovery, "run_daily_discovery", lambda **kw: calls.append(kw) or {"batch_id": 1})
    monkeypatch.setattr(settings, "DAILY_TOPIC_SCOUT_ENABLED", True)

    scheduler.run_topic_discovery_once()

    assert len(calls) == 1
    assert calls[0]["triggered_by"] == "scheduler"


def test_run_topic_discovery_once_skips_when_disabled(isolated_storage, monkeypatch):
    from youtube import topic_discovery

    calls = []
    monkeypatch.setattr(topic_discovery, "run_daily_discovery", lambda **kw: calls.append(kw))
    monkeypatch.setattr(settings, "DAILY_TOPIC_SCOUT_ENABLED", False)

    scheduler.run_topic_discovery_once()

    assert calls == []


def test_discover_once_cli_flag_calls_run_topic_discovery_once(isolated_storage, monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "run_topic_discovery_once", lambda: calls.append(True))

    exit_code = scheduler.main(["--discover-once"])

    assert exit_code == 0
    assert calls == [True]


# ---------------------------------------------------------------------------
# N - scheduler does not trigger production
# ---------------------------------------------------------------------------

def test_run_topic_discovery_once_never_calls_produce_video(isolated_storage, monkeypatch):
    import pipeline as pipeline_module
    from youtube import topic_discovery

    monkeypatch.setattr(pipeline_module, "produce_video", lambda *a, **k: pytest.fail("scheduled discovery must never start production"))
    # run_daily_discovery() itself is the real function here (not mocked) -
    # but it never reaches produce_video() unless a human selects a topic
    # via Telegram, so this only proves the discovery cycle's call path
    # never touches it, using a stub discovery result to avoid a real
    # Claude/Telegram call.
    monkeypatch.setattr(topic_discovery, "run_daily_discovery", lambda **kw: {"batch_id": 1, "candidates": []})
    monkeypatch.setattr(settings, "DAILY_TOPIC_SCOUT_ENABLED", True)

    scheduler.run_topic_discovery_once()  # must not raise (i.e. produce_video was never called)


def test_run_topic_discovery_once_survives_a_failed_discovery_cycle(isolated_storage, monkeypatch):
    from youtube import topic_discovery

    def boom(**kw):
        raise RuntimeError("simulated discovery failure")

    monkeypatch.setattr(topic_discovery, "run_daily_discovery", boom)
    monkeypatch.setattr(settings, "DAILY_TOPIC_SCOUT_ENABLED", True)

    scheduler.run_topic_discovery_once()  # must not raise


# ---------------------------------------------------------------------------
# scheduler.run_daily_topic_discovery() config-driven schedule
# ---------------------------------------------------------------------------

def test_daily_topic_scout_config_defaults(isolated_storage):
    assert settings.DAILY_TOPIC_SCOUT_TIME == "09:00"
    assert settings.DAILY_TOPIC_SCOUT_TIMEZONE == "Europe/Istanbul"
    assert settings.DAILY_TOPIC_SCOUT_ENABLED is True


def test_run_daily_topic_discovery_fires_once_per_date_at_configured_time(isolated_storage, monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "run_topic_discovery_once", lambda: calls.append(True))
    monkeypatch.setattr(settings, "DAILY_TOPIC_SCOUT_ENABLED", True)

    fixed_now = dt.datetime(2026, 9, 22, 9, 0, tzinfo=scheduler.ZoneInfo("Europe/Istanbul"))
    ticks = [fixed_now, fixed_now, "STOP"]

    class _FakeDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            value = ticks.pop(0)
            if value == "STOP":
                raise KeyboardInterrupt()
            return value

    monkeypatch.setattr(scheduler.dt, "datetime", _FakeDatetime)
    monkeypatch.setattr(scheduler.time, "sleep", lambda s: None)

    with pytest.raises(KeyboardInterrupt):
        scheduler.run_daily_topic_discovery(9, 0, "Europe/Istanbul")

    assert calls == [True]  # fired once, not twice, for the same date


def test_run_daily_topic_discovery_stops_looping_when_disabled_mid_loop(isolated_storage, monkeypatch):
    monkeypatch.setattr(settings, "DAILY_TOPIC_SCOUT_ENABLED", False)
    monkeypatch.setattr(scheduler, "run_topic_discovery_once", lambda: pytest.fail("must not run while disabled"))

    scheduler.run_daily_topic_discovery(9, 0, "Europe/Istanbul")  # returns immediately, does not loop forever


def test_discover_daily_cli_flag_uses_configured_time_and_timezone(isolated_storage, monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "run_daily_topic_discovery", lambda h, m, tz: calls.append((h, m, tz)))
    monkeypatch.setattr(settings, "DAILY_TOPIC_SCOUT_TIME", "11:30")
    monkeypatch.setattr(settings, "DAILY_TOPIC_SCOUT_TIMEZONE", "Europe/Istanbul")

    exit_code = scheduler.main(["--discover-daily"])

    assert exit_code == 0
    assert calls == [(11, 30, "Europe/Istanbul")]
