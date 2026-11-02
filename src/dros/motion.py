"""Commanding the end-effector position, or the shoulder angles, through any joint bus.

This is not trajectory control (ADR 0006): a move is one joint target, carried by
the bus's own velocity and acceleration limits. The path between positions is not
a straight line, and the shoulders may arrive at different times.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from .geometry import DeltaGeometry
from .kinematics import inverse
from .servo.bus import JointBus


def set_end_effector_position(
    bus: JointBus,
    geom: DeltaGeometry,
    position: ArrayLike,
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.01,
    tolerance: float = 0.002,
) -> None:
    """Move the end-effector to `position` (m) and wait until the shoulders reach it.

    Inverse kinematics gives the shoulder angles; `set_shoulder_angles` sends and
    waits for them, with `sleep`, `poll_interval` and `tolerance` passed on.
    """
    set_shoulder_angles(
        bus,
        inverse(geom, position),
        sleep=sleep,
        poll_interval=poll_interval,
        tolerance=tolerance,
    )


def set_shoulder_angles(
    bus: JointBus,
    theta: ArrayLike,
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.01,
    tolerance: float = 0.002,
) -> None:
    """Move the shoulders to `theta` (rad) and wait until they have settled there.

    The joint target goes out in one `write_angles`, which checks the joint limits.
    The wait polls `read_angles` every `poll_interval` seconds through `sleep`, until
    the shoulders have settled: every one within `tolerance` (rad) of its target, and
    the reading unchanged since the poll before. `tolerance` only has to cover how far
    the bus rounds a target; settling is what puts the shoulders on it. Pass a
    simulated clock's advance as `sleep` to run on virtual time.
    """
    target = np.asarray(theta, dtype=float)
    bus.write_angles(target)
    before = bus.read_angles()
    while True:
        sleep(poll_interval)
        now = bus.read_angles()
        if np.array_equal(now, before) and np.all(np.abs(now - target) <= tolerance):
            return
        before = now
