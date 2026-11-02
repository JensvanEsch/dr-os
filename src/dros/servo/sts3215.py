"""STS3215 position units. 12-bit encoder: 4096 ticks per revolution.

Only unit conversion lives here for now. The serial driver (feetech-servo-sdk,
optional extra "hw") will be added during bring-up; see .claude/commands/hw-bringup.md.
"""
from __future__ import annotations

import numpy as np

TICKS_PER_REV = 4096
CENTER_TICK = 2048
RAD_PER_TICK = 2 * np.pi / TICKS_PER_REV


def rad_to_ticks(theta, zero_tick: int = CENTER_TICK, direction: int = 1) -> np.ndarray:
    """Joint angle (rad) -> raw position ticks, per-servo zero and direction."""
    ticks = zero_tick + direction * np.asarray(theta, dtype=float) * TICKS_PER_REV / (2 * np.pi)
    return np.rint(ticks).astype(int)


def ticks_to_rad(ticks, zero_tick: int = CENTER_TICK, direction: int = 1) -> np.ndarray:
    """Raw position ticks -> joint angle (rad)."""
    return direction * (np.asarray(ticks, dtype=float) - zero_tick) * 2 * np.pi / TICKS_PER_REV
