"""Hardware-independent joint bus interface. Real drivers implement JointBus.

Every joint bus enforces safety itself in `write_angles` (ADR 0005).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..geometry import DeltaGeometry


class JointBus(Protocol):
    def read_angles(self) -> np.ndarray: ...
    def write_angles(self, theta: np.ndarray) -> None: ...
    def enable_torque(self) -> None: ...
    def disable_torque(self) -> None: ...
    def close(self) -> None: ...


def check_motion_limits(max_speed: float, max_acceleration: float) -> None:
    """Raise ValueError unless the speed and acceleration limits are positive and finite."""
    for name, value in {"max_speed": max_speed, "max_acceleration": max_acceleration}.items():
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive and finite, got {value}")


def check_target(geom: DeltaGeometry, theta: ArrayLike) -> np.ndarray:
    """The safety checks every joint bus runs in `write_angles` before a target goes out.

    Raises ValueError for a target that isn't three angles or one outside the joint
    limits. Returns the target as a float array.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError(f"a target is three shoulder angles, got shape {theta.shape}")
    geom.check_joint_limits(theta)
    return theta


class MockBus:
    """In-memory joint bus for development and tests. Moves instantly.

    max_speed (rad/s) and max_acceleration (rad/s^2) are accepted like on any bus,
    but the mock bus has no motion of its own to apply them to.
    """

    def __init__(
        self,
        geom: DeltaGeometry,
        *,
        max_speed: float,
        max_acceleration: float,
        start_angles: ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        check_motion_limits(max_speed, max_acceleration)
        self.geom = geom
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self._theta = np.asarray(start_angles, dtype=float).copy()
        self._torque_on = False

    def read_angles(self) -> np.ndarray:
        return self._theta.copy()

    def write_angles(self, theta: np.ndarray) -> None:
        if not self._torque_on:
            raise RuntimeError("torque is off: turn it on before commanding the shoulders")
        self._theta = check_target(self.geom, theta).copy()

    def enable_torque(self) -> None:
        self._torque_on = True

    def disable_torque(self) -> None:
        self._torque_on = False

    def close(self) -> None:
        pass
