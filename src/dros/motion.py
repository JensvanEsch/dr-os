"""Commanding positions in metres through any joint bus.

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


def move_to(
    bus: JointBus,
    geom: DeltaGeometry,
    position: ArrayLike,
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.01,
    tolerance: float = 0.002,
) -> None:
    """Move the end-effector to `position` (m) and wait until the shoulders reach it.

    The joint target goes out in one `write_angles`. The wait polls `read_angles`
    every `poll_interval` seconds through `sleep`, until every shoulder is within
    `tolerance` (rad) of its target. Pass a simulated clock's advance as `sleep` to
    run on virtual time.
    """
    target = inverse(geom, position)
    bus.write_angles(target)
    while np.any(np.abs(bus.read_angles() - target) > tolerance):
        sleep(poll_interval)
