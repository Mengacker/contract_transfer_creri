"""Utilities functions for input validation."""

import re

iso_date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_iso_date(date_str: str) -> bool:
    """
    Check wheter a date match iso date format.
    Args:
        date_str (string): date string.
    Returns:
        bool: True if date_str match iso date format, False otherwise.
    """
    return bool(iso_date_pattern.match(date_str))
