"""
Generic filter helpers for SDG-patent RAG pipeline.

Each filter:  {"column": "...", "op": "...", "value": ...}

Supported ops:  eq, neq, contains, startswith, in, gte, lte, between
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Sequence
import numbers

__all__ = ["apply_filter"]


# ───────────────────── helpers ───────────────────────────────────────────
def _to_date(val: str | datetime):
    if isinstance(val, datetime):
        return val
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(str(val)[: len(fmt)], fmt)
        except ValueError:
            continue
    return None


# ───────────────────── public API ────────────────────────────────────────
def apply_filter(row_val: Any, op: str, value: Any) -> bool:
    """Return True iff *row_val* satisfies <op> against <value>."""

    # normalised views
    row_text = str(row_val).lower()
    row_num  = float(row_val) if isinstance(row_val, numbers.Number) else None

    # text ops
    if op == "eq":
        return str(row_val) == str(value)
    if op == "neq":
        return str(row_val) != str(value)
    if op == "contains":
        return str(value).lower() in row_text
    if op == "startswith":
        return row_text.startswith(str(value).lower())
    if op == "in":
        if isinstance(value, Sequence):
            vals = {str(v).lower() for v in value}
            return row_text in vals or any(v in row_text for v in vals)
        return str(value).lower() in row_text

    # numeric / date ops
    if op in {"gte", "lte", "between"}:
        if row_num is not None and isinstance(value, numbers.Number):
            return row_num >= value if op == "gte" else row_num <= value
        row_date = _to_date(row_val)
        if row_date is None:
            return False
        if op == "between" and isinstance(value, Sequence) and len(value) == 2:
            start, end = map(_to_date, value)
            return start and end and start <= row_date <= end
        cmp_date = _to_date(value)
        return cmp_date and ((row_date >= cmp_date) if op == "gte" else (row_date <= cmp_date))

    # unknown op → do not filter out
    return True
