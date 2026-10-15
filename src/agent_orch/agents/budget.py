"""Validate budget-only continuation without moving existing phase boundaries."""

from __future__ import annotations

from collections.abc import Sequence


def canonical_schedule(schedule: Sequence | None) -> tuple[tuple[str, int], ...] | None:
    """JSON manifests use lists; torch checkpoints use tuples."""
    if schedule is None:
        return None
    result = tuple(tuple(item) for item in schedule)
    if not result or any(
        len(item) != 2 or item[0] not in ("composition", "deployment", "joint")
        or type(item[1]) is not int or item[1] <= 0
        for item in result
    ):
        raise ValueError("Invalid saved or requested PPO phase schedule")
    return result


def validate_resume_schedule(
    saved: Sequence | None,
    requested: Sequence | None,
    *,
    allow_extension: bool = False,
) -> bool:
    """Allow only an explicit increase of the final existing phase's budget.

    Earlier phase lengths, phase order, and the final phase identity stay fixed.
    Return whether the budget changed. This never modifies a checkpoint.
    """
    old, new = canonical_schedule(saved), canonical_schedule(requested)
    if old == new:
        return False
    if not allow_extension:
        raise ValueError("PPO phase schedule changed; budget extension requires --resume --extend-budget")
    if (old is None or new is None or old[:-1] != new[:-1]
            or old[-1][0] != new[-1][0] or new[-1][1] <= old[-1][1]):
        raise ValueError("Budget extension may only increase the final existing phase; "
                         "cannot shorten, add, reorder phases or move earlier phase boundaries")
    return True
