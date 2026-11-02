"""Commanding the end-effector position, or the shoulder angles, through any joint bus.

Two levels:
- `set_*` sends a joint target and returns at once, for callers that run their own
  loop or stream targets.
- `move_*` sends a joint target and waits until the shoulders have settled on it,
  so the next move starts from a known state.

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
from .kinematics import UnreachableError, inverse
from .servo.bus import JointBus


def _shoulder_angles_for(geom: DeltaGeometry, position: ArrayLike) -> np.ndarray:
    """Shoulder angles (rad) for `position` (m), checked against the joint limits.

    Raises ValueError if the position contains NaN or infinity, and UnreachableError
    if it is out of the workspace: out of the arms' reach, or only reachable with a
    shoulder outside its joint limits.
    """
    if not np.all(np.isfinite(np.asarray(position, dtype=float))):
        raise ValueError(f"position {position} is not finite")
    theta = inverse(geom, position)
    try:
        geom.check_joint_limits(theta)
    except ValueError as error:
        raise UnreachableError(f"{position} is outside the workspace: {error}") from error
    return theta


class ShoulderOutsideLimitsError(RuntimeError):
    """A shoulder measures outside its joint limits, so no motion was sent.

    The operator has to move that arm back inside its limits by hand.
    """


def _check_start(bus: JointBus, geom: DeltaGeometry) -> None:
    """Raise ShoulderOutsideLimitsError if any measured shoulder angle is outside the limits."""
    measured = bus.read_angles()
    for arm, theta in enumerate(measured):
        try:
            geom.check_joint_limits(theta)
        except ValueError:
            raise ShoulderOutsideLimitsError(
                f"arm {arm} measures {theta:.3f} rad, outside the joint limits "
                f"[{geom.joint_min}, {geom.joint_max}] rad, so nothing was moved. "
                "Switch the servo power off, move that arm back inside its limits by hand, "
                "and start again."
            ) from None


def set_shoulder_angles(bus: JointBus, geom: DeltaGeometry, theta: ArrayLike) -> None:
    """Send `theta` (rad) as the joint target and return without waiting.

    Raises ShoulderOutsideLimitsError, with no motion sent, if the measured angles
    start outside the joint limits. `write_angles` checks the target against the
    joint limits and refuses while torque is off.
    """
    _check_start(bus, geom)
    bus.write_angles(np.asarray(theta, dtype=float))


def set_end_effector_position(bus: JointBus, geom: DeltaGeometry, position: ArrayLike) -> None:
    """Send the joint target for `position` (m) and return without waiting.

    Raises UnreachableError before any bus command if the position is out of the
    workspace, and ShoulderOutsideLimitsError as `set_shoulder_angles` does.
    """
    set_shoulder_angles(bus, geom, _shoulder_angles_for(geom, position))


def move_shoulder_angles(
    bus: JointBus,
    geom: DeltaGeometry,
    theta: ArrayLike,
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.01,
    tolerance: float = 0.002,
) -> None:
    """Move the shoulders to `theta` (rad) and wait until they have settled there.

    The wait polls `read_angles` every `poll_interval` seconds through `sleep`, until
    every shoulder is within `tolerance` (rad) of its target and the reading is
    unchanged since the poll before. `tolerance` only has to cover how far the bus
    rounds a target; settling is what puts the shoulders on it. Pass a simulated
    clock's advance as `sleep` to run on virtual time.
    """
    target = np.asarray(theta, dtype=float)
    set_shoulder_angles(bus, geom, target)
    before = bus.read_angles()
    while True:
        sleep(poll_interval)
        now = bus.read_angles()
        if np.array_equal(now, before) and np.all(np.abs(now - target) <= tolerance):
            return
        before = now


def move_end_effector_position(
    bus: JointBus,
    geom: DeltaGeometry,
    position: ArrayLike,
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.01,
    tolerance: float = 0.002,
) -> None:
    """Move the end-effector to `position` (m) and wait until the shoulders reach it.

    Inverse kinematics gives the shoulder angles; `move_shoulder_angles` sends and
    waits for them, with `sleep`, `poll_interval` and `tolerance` passed on. Raises
    UnreachableError before any bus command if the position is out of the workspace.
    """
    move_shoulder_angles(
        bus,
        geom,
        _shoulder_angles_for(geom, position),
        sleep=sleep,
        poll_interval=poll_interval,
        tolerance=tolerance,
    )
