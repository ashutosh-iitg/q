"""Plain JSON for records and datetimes."""

from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime
from typing import Any


def plain(record: Any) -> dict[str, Any]:
    return {f.name: getattr(record, f.name) for f in fields(record)}


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")
