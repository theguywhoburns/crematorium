from __future__ import annotations

from typing import Any, Callable, Union

__all__ = [
    "multiple_of",
    "normalize_spatial",
    "positive",
    "spatial_rank",
    "spatial_spec",
]


def positive(value: Union[int, float]) -> None:
    """Validate that ``value`` is a positive int or float (not bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"must be a positive number, got {value!r}")


def multiple_of(n: int) -> Callable[[Any], None]:
    """Validate that ``value`` is a positive multiple of ``n`` (not bool)."""

    def validate(value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % n:
            raise ValueError(f"must be a positive multiple of {n}, got {value!r}")

    return validate


def spatial_spec(value: Any) -> None:
    """Validate an int or tuple of ints, all non-negative (not bool)."""
    if isinstance(value, bool):
        raise ValueError(f"must be an int or tuple of ints, got {value!r}")

    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"must be a non-negative int, got {value!r}")

        return

    if (
        isinstance(value, (tuple, list))
        and len(value) > 0
        and all(type(v) is int and v >= 0 for v in value)
    ):
        return

    raise ValueError(f"must be an int or tuple of ints, got {value!r}")


def spatial_rank(value: Any) -> None:
    """Validate a spatial rank: 1, 2, or 3 (not bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError(f"rank must be the spatial rank 1, 2, or 3, got {value!r}")


def normalize_spatial(
    value: Any, spatial: int, name: str, minimum: int
) -> tuple[int, ...]:
    """Broadcast an int spec to a rank-tuple (tuples pass through checked)."""
    items = (value,) * spatial if isinstance(value, int) else tuple(value)

    if len(items) != spatial:
        raise ValueError(
            f"{name} length {len(items)} != spatial rank {spatial}; "
            f"pass an int or a {spatial}-tuple"
        )

    if any(v < minimum for v in items):
        raise ValueError(f"{name} entries must be >= {minimum}, got {items!r}")

    return items
