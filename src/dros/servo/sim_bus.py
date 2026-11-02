"""Sim bus: a joint bus that imitates servo motion over simulated time. No physics."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..geometry import DeltaGeometry
from .bus import check_motion_limits, check_target
from .sts3215 import rad_to_ticks, ticks_to_rad


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
        self._velocity = np.zeros(3)
        self._torque_on = False

    def read_angles(self) -> np.ndarray:
        return _on_ticks(self._theta)

    def write_angles(self, theta: np.ndarray) -> None:
        if not self._torque_on:
            raise RuntimeError("torque is off: turn it on before commanding the shoulders")
        # Like a servo's goal position, the target is a whole number of ticks.
        self._target = _on_ticks(check_target(self.geom, theta))

    def enable_torque(self) -> None:
        self._torque_on = True

    def disable_torque(self) -> None:
        self._torque_on = False

    def close(self) -> None:
        pass

    def advance(self, dt: float) -> None:
        """Advance simulated time by dt seconds; each shoulder moves towards its target."""
        for i in range(3):
            self._theta[i], self._velocity[i] = _follow(
                self._theta[i],
                self._velocity[i],
                self._target[i],
                dt,
                self.max_velocity,
                self.max_acceleration,
            )


def _on_ticks(theta: np.ndarray) -> np.ndarray:
    """Round angles (rad) to the nearest tick, as a servo encoder reports them."""
    return ticks_to_rad(rad_to_ticks(theta))


# Below this, a distance (rad) or speed (rad/s) counts as zero.
_EPS = 1e-12


def _follow(
    x: float, v: float, goal: float, dt: float, v_max: float, a_max: float
) -> tuple[float, float]:
    """Move one shoulder from angle x and velocity v towards goal for dt seconds.

    Time-optimal under |v| <= v_max and |a| <= a_max, ending at rest on the goal.
    The motion is a chain of constant-acceleration segments, each stepped exactly.
    Returns the new angle and velocity.
    """
    while dt > 0:
        # Work in a frame where the goal lies ahead (in the + direction).
        s = 1.0 if goal >= x else -1.0
        d, u = s * (goal - x), s * v
        if d <= _EPS and abs(u) <= _EPS:
            return goal, 0.0
        stopping = u * u / (2 * a_max) if u > 0 else 0.0
        if u < 0:  # moving away: stop first
            a, t, lands = a_max, -u / a_max, False
        elif stopping > d + _EPS:  # too fast to stop on the goal: stop past it
            a, t, lands = -a_max, u / a_max, False
        elif stopping >= d - _EPS:  # brake onto the goal
            a, t, lands = -u * u / (2 * d), 2 * d / u, True
        elif u >= v_max - _EPS:  # cruise until it is time to brake
            a, t, lands = 0.0, (d - stopping) / u, False
        else:  # speed up, towards the max velocity or the point where braking must start
            peak = min(v_max, np.sqrt(a_max * d + u * u / 2))
            a, t, lands = a_max, (peak - u) / a_max, False
        if lands and t <= dt:
            return goal, 0.0
        h = min(t, dt)
        x += s * (u * h + a * h * h / 2)
        v = s * (u + a * h)
        dt -= h
    return x, v
