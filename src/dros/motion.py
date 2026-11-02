"""Commanding the end-effector position, or the shoulder angles, through any joint bus.

`set_*` checks the request, sends one joint target and returns at once. Waiting for
the shoulders to arrive, and ending a run at the rest position with torque off, is
up to the application.

This is not trajectory control (ADR 0006): a move is one joint target, carried by
the bus's own velocity and acceleration limits. The path between positions is not
a straight line, and the shoulders may arrive at different times.

These functions never touch torque. If anything fails after a target is sent, the
servos keep holding it with torque on: turning torque off would drop the
end-effector (ADR 0005).
"""
from __future__ import annotations

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
        if not np.isfinite(theta):
            raise ShoulderOutsideLimitsError(
                f"arm {arm} reads {theta} rad, which is not finite, so nothing was moved. "
                "Check that servo's connection and reading before starting again."
            )
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

    Raises ValueError before any bus command if the target is outside the joint
    limits, and ShoulderOutsideLimitsError, with no motion sent, if the measured
    angles start outside them or are not finite. `write_angles` checks the target
    again and refuses while torque is off.
    """
    target = np.asarray(theta, dtype=float)
    geom.check_joint_limits(target)
    _check_start(bus, geom)
    bus.write_angles(target)


def set_end_effector_position(bus: JointBus, geom: DeltaGeometry, position: ArrayLike) -> None:
    """Send the joint target for `position` (m) and return without waiting.

    Raises ValueError before any bus command if the position is not finite, and
    UnreachableError if it is out of the workspace; otherwise fails as
    `set_shoulder_angles` does.
    """
    set_shoulder_angles(bus, geom, _shoulder_angles_for(geom, position))
