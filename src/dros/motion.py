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
    every `poll_interval` seconds through `sleep`, until the shoulders have settled:
    every one within `tolerance` (rad) of its target, and the reading unchanged since
    the poll before. `tolerance` only has to cover how far the bus rounds a target;
    settling is what puts the end-effector on it. Pass a simulated clock's advance
    as `sleep` to run on virtual time.
    """
    target = inverse(geom, position)
    bus.write_angles(target)
    before = bus.read_angles()
    while True:
        sleep(poll_interval)
        now = bus.read_angles()
        if np.array_equal(now, before) and np.all(np.abs(now - target) <= tolerance):
            return
        before = now


def first_move(
    bus: JointBus,
    geom: DeltaGeometry,
    *,
    depth: float = 0.020,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """The first move: from the rest position `depth` (m) down in z, back, torque off.

    Torque turns on at the start and off only at the end, with the end-effector at
    the rest position, so it is never dropped from height.
    """
    rest = np.array(geom.rest_position, dtype=float)
    bus.enable_torque()
    for position in (rest, rest - [0.0, 0.0, depth], rest):
        move_to(bus, geom, position, sleep=sleep)
    bus.disable_torque()
