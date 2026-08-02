"""Application-wide Jakarta time helpers."""

from datetime import datetime
from zoneinfo import ZoneInfo

JAKARTA_TZ = ZoneInfo("Asia/Jakarta")


def jakarta_now():
    """Return Jakarta local time as a SQLite-compatible naive datetime."""
    return datetime.now(JAKARTA_TZ).replace(tzinfo=None)
