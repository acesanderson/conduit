from __future__ import annotations

import re

import click


def parse_duration(value: str) -> str:
    """
    Parse a human-readable duration string into a Postgres interval string.

    Accepted formats: ``^\\d+[dwh]$``
      'd' -> 'N days'
      'w' -> 'N*7 days'
      'h' -> 'N hours'

    Raises click.BadParameter for zero durations or invalid formats.
    """
    pattern = re.compile(r"^(\d+)([dwh])$")
    match = pattern.match(value)
    if not match:
        raise click.BadParameter(
            f"Invalid duration '{value}'. Expected format: <number>[d|w|h] (e.g. 7d, 2w, 48h)."
        )

    amount = int(match.group(1))
    unit = match.group(2)

    if amount == 0:
        raise click.BadParameter("Duration must be greater than zero.")

    if unit == "d":
        return f"{amount} days"
    elif unit == "w":
        return f"{amount * 7} days"
    else:  # unit == "h"
        return f"{amount} hours"
