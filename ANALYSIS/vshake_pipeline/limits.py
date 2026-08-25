"""Parse optional axis limit pairs from pipeline YAML."""

from __future__ import annotations

from typing import Any


def parse_optional_lim_pair(
    value: Any,
    *,
    field_name: str,
) -> tuple[float, float] | None:
    """
    Return a (min, max) pair for fixed axis limits.

    ``None`` (YAML ``null``) means leave limits to matplotlib or the script default.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be null or [min, max]")
    return (float(value[0]), float(value[1]))


def parse_optional_lim_pair_from_section(
    section: dict[str, Any],
    key: str,
    *,
    field_name: str | None = None,
) -> tuple[float, float] | None:
    """Read *key* from a config section; missing key → ``None`` (autoscale)."""
    if key not in section:
        return None
    name = field_name or key
    return parse_optional_lim_pair(section[key], field_name=name)
