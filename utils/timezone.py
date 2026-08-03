"""Application-wide Jakarta time helpers."""

from datetime import datetime
from zoneinfo import ZoneInfo

JAKARTA_TZ = ZoneInfo("Asia/Jakarta")
INDONESIAN_DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
INDONESIAN_MONTHS = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def jakarta_now():
    """Return Jakarta local time as a SQLite-compatible naive datetime."""
    return datetime.now(JAKARTA_TZ).replace(tzinfo=None)


def format_indonesian_date(value, include_time=False):
    """Format a datetime using full Indonesian day and month names."""
    if value is None:
        return "-"
    formatted = (
        f"{INDONESIAN_DAYS[value.weekday()]}, {value.day:02d} "
        f"{INDONESIAN_MONTHS[value.month - 1]} {value.year}"
    )
    return f"{formatted}, {value:%H:%M} WIB" if include_time else formatted
