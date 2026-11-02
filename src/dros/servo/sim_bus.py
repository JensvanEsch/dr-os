"""Sim bus: a joint bus that imitates servo motion over simulated time. No physics."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..geometry import DeltaGeometry
from .bus import check_motion_limits, check_target


class SimBus:
    """Joint bus that moves towards its target only when simulated time advances."""

    def __init__(
        self,
        geom: DeltaGeometry,
        *,
        max_velocity: float,
        max_acceleration: float,
        start_angles: ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        check_motion_limits(max_velocity, max_acceleration)
        self.geom = geom
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self._theta = np.asarray(start_angles, dtype=float).copy()
        self._target = self._theta.copy()
        self._torque_on = False

    def read_angles(self) -> np.ndarray:
        return self._theta.copy()

    def write_angles(self, theta: np.ndarray) -> None:
        if not self._torque_on:
            raise RuntimeError("torque is off: turn it on before commanding the shoulders")
        self._target = check_target(self.geom, theta).copy()

    def enable_torque(self) -> None:
        self._torque_on = True

    def disable_torque(self) -> None:
        self._torque_on = False

    def close(self) -> None:
        pass

    def advance(self, dt: float) -> None:
        self._theta = self._target.copy()
