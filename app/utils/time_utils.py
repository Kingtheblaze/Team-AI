"""
Time and UTC formatting utilities for temporal RAG.
"""
from datetime import datetime, timezone, timedelta
import re
from typing import Optional, Tuple


def now_utc() -> datetime:
    """Return current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def to_iso_utc(dt: Optional[datetime] = None) -> str:
    """Convert datetime to standard ISO 8601 UTC string (YYYY-MM-DDTHH:MM:SSZ)."""
    if dt is None:
        dt = now_utc()
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(ts_str: str) -> datetime:
    """Parse ISO 8601 UTC timestamp string into timezone-aware datetime."""
    clean_str = ts_str.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        # Fallback to current UTC if malformed
        return now_utc()


def parse_time_window_to_delta(window_str: str) -> Optional[timedelta]:
    """
    Parse colloquial time window expressions such as:
    - 'last 10 minutes' -> timedelta(minutes=10)
    - 'last 1 hour' -> timedelta(hours=1)
    - 'past 30 seconds' -> timedelta(seconds=30)
    """
    if not window_str:
        return None
    s = window_str.lower().strip()
    match = re.search(r"(\d+)\s*(m|min|minute|minutes|h|hr|hour|hours|s|sec|second|seconds|d|day|days)", s)
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)

    if unit.startswith("s"):
        return timedelta(seconds=value)
    elif unit.startswith("m"):
        return timedelta(minutes=value)
    elif unit.startswith("h"):
        return timedelta(hours=value)
    elif unit.startswith("d"):
        return timedelta(days=value)
    return None


def extract_temporal_bounds(query: str, default_window_minutes: Optional[int] = None) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Extract start and end UTC timestamps from user query expressions.
    Supports:
    - 'last N minutes/hours'
    - 'since HH:MM UTC'
    - 'between T1 and T2'
    """
    now = now_utc()
    # Check 'last N minutes/hours'
    delta = parse_time_window_to_delta(query)
    if delta is not None:
        return (now - delta, now)

    # Check 'since HH:MM' pattern
    since_match = re.search(r"since\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:utc)?", query, re.IGNORECASE)
    if since_match:
        hr = int(since_match.group(1))
        mn = int(since_match.group(2))
        sec = int(since_match.group(3) or 0)
        since_time = now.replace(hour=hr, minute=mn, second=sec, microsecond=0)
        if since_time > now:
            since_time -= timedelta(days=1)
        return (since_time, now)

    if default_window_minutes is not None:
        return (now - timedelta(minutes=default_window_minutes), now)

    return (None, None)
