"""
Evaluation metrics for the conditional date generation task.

Since this is a *generation* problem with many correct answers per input,
exact-match accuracy is not meaningful.  We instead use:

1. Date Validity Rate
   Fraction of generated strings that parse as a real calendar date
   within [1-1-1800, 31-12-2200].

2. Condition Satisfaction Rate (CSR) – per condition and overall
   For each generated date that is valid, check whether it satisfies
   each of the four input conditions.  Report per-condition rates and
   the "all-four-pass" overall rate.

   - Day CSR   : generated date falls on the required weekday
   - Month CSR : generated date falls in the required month
   - Leap CSR  : year is (or is not) a leap year as required
   - Decade CSR: year falls in the required decade  ([196] → 1960-1969)

3. Diversity Score
   Ratio of unique valid date strings to total valid date strings.
   A perfectly collapsed generator always outputs the same date → score 0.
   A perfectly diverse generator has score ≈ 1.

Usage
-----
    from metrics import evaluate, print_metrics

    preds = ["[MON] [DEC] [False] [196] 3-12-1962", ...]
    m = evaluate(preds)
    print_metrics(m, prefix="VAE epoch 10")
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import Dict, List, Optional, Tuple


# ── Lookup tables ─────────────────────────────────────────────────────────────

_DAY_TO_WEEKDAY: Dict[str, int] = {
    "[MON]": 0, "[TUE]": 1, "[WED]": 2, "[THU]": 3,
    "[FRI]": 4, "[SAT]": 5, "[SUN]": 6,
}

_MONTH_TO_INT: Dict[str, int] = {
    "[JAN]": 1,  "[FEB]": 2,  "[MAR]": 3,  "[APR]": 4,
    "[MAY]": 5,  "[JUN]": 6,  "[JUL]": 7,  "[AUG]": 8,
    "[SEP]": 9,  "[OCT]": 10, "[NOV]": 11, "[DEC]": 12,
}

_DATE_MIN = date(1800, 1, 1)
_DATE_MAX = date(2200, 12, 31)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_line(
    line: str,
) -> Optional[Tuple[str, str, str, str, str]]:
    """Split a prediction line into its five fields."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def _parse_date(date_str: str) -> Optional[date]:
    """
    Parse a date string of the form d-m-yyyy or dd-mm-yyyy.
    Returns None if invalid.
    """
    try:
        p = date_str.split("-")
        if len(p) != 3:
            return None
        d, m, y = int(p[0]), int(p[1]), int(p[2])
        return date(y, m, d)
    except (ValueError, OverflowError, TypeError):
        return None


# ── Per-sample condition checks ───────────────────────────────────────────────

def check_day(dt: date, day_tok: str) -> bool:
    """True if the date's weekday matches the day token."""
    expected = _DAY_TO_WEEKDAY.get(day_tok)
    return expected is not None and dt.weekday() == expected


def check_month(dt: date, month_tok: str) -> bool:
    """True if the date's month matches the month token."""
    expected = _MONTH_TO_INT.get(month_tok)
    return expected is not None and dt.month == expected


def check_leap(dt: date, leap_tok: str) -> bool:
    """True if the year's leap-year status matches the token."""
    expected = (leap_tok == "[True]")
    return calendar.isleap(dt.year) == expected


def check_decade(dt: date, decade_tok: str) -> bool:
    """
    True if the year falls in the decade specified by the token.
    E.g. [196] → 1960–1969.
    """
    inner = decade_tok[1:-1]
    if len(inner) != 3:
        return False
    decade_start = int(inner) * 10
    return decade_start <= dt.year <= decade_start + 9


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate(predictions: List[str]) -> Dict[str, float | int]:
    """
    Evaluate a list of generated lines.

    Each line must be the full format:
        "[MON] [DEC] [False] [196] 3-12-1962"

    Returns
    -------
    dict with keys:
        validity_rate, day_csr, month_csr, leap_csr, decade_csr,
        overall_csr, diversity_score, n_valid, n_total
    """
    n_total = len(predictions)
    if n_total == 0:
        return {}

    n_valid = 0
    day_pass = month_pass = leap_pass = decade_pass = overall_pass = 0
    unique_dates: set[str] = set()

    for line in predictions:
        parsed = _parse_line(line)
        if parsed is None:
            continue
        day_tok, month_tok, leap_tok, decade_tok, date_str = parsed

        dt = _parse_date(date_str)
        if dt is None or not (_DATE_MIN <= dt <= _DATE_MAX):
            continue

        n_valid += 1
        unique_dates.add(date_str)

        d_ok   = check_day(dt,    day_tok)
        m_ok   = check_month(dt,  month_tok)
        l_ok   = check_leap(dt,   leap_tok)
        dec_ok = check_decade(dt, decade_tok)

        if d_ok:            day_pass   += 1
        if m_ok:            month_pass += 1
        if l_ok:            leap_pass  += 1
        if dec_ok:          decade_pass += 1
        if d_ok and m_ok and l_ok and dec_ok:
            overall_pass += 1

    base = max(n_valid, 1)

    return {
        "validity_rate":   round(n_valid      / n_total, 4),
        "day_csr":         round(day_pass     / base,    4),
        "month_csr":       round(month_pass   / base,    4),
        "leap_csr":        round(leap_pass    / base,    4),
        "decade_csr":      round(decade_pass  / base,    4),
        "overall_csr":     round(overall_pass / base,    4),
        "diversity_score": round(len(unique_dates) / base, 4),
        "n_valid":         n_valid,
        "n_total":         n_total,
    }


def print_metrics(metrics: Dict, prefix: str = "") -> None:
    """Pretty-print an evaluate() result dict."""
    bar = "=" * 52
    print(f"\n{bar}")
    if prefix:
        print(f"  {prefix}")
        print(f"  {'-' * (len(prefix))}")
    print(f"  Validity Rate  : {metrics.get('validity_rate', 0):.2%}")
    print(f"  Overall CSR    : {metrics.get('overall_csr',   0):.2%}  (all 4 conditions)")
    print(f"    Day CSR      : {metrics.get('day_csr',       0):.2%}")
    print(f"    Month CSR    : {metrics.get('month_csr',     0):.2%}")
    print(f"    Leap CSR     : {metrics.get('leap_csr',      0):.2%}")
    print(f"    Decade CSR   : {metrics.get('decade_csr',    0):.2%}")
    print(f"  Diversity      : {metrics.get('diversity_score', 0):.4f}  (unique / valid)")
    print(f"  Valid / Total  : {metrics.get('n_valid', 0)}/{metrics.get('n_total', 0)}")
    print(f"{bar}\n")
