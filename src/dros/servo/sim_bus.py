"""Sim bus: a joint bus that imitates servo motion over simulated time. No physics."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..geometry import DeltaGeometry
from .bus import check_motion_limits, check_target
from .sts3215 import RAD_PER_TICK, rad_to_ticks, ticks_to_rad


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
        # Like a servo's goal position, the target is a whole number of ticks,
        # rounded towards the inside of the joint limits so it never ends past them.
        target = _on_ticks(check_target(self.geom, theta))
        target[target > self.geom.joint_max] -= RAD_PER_TICK
        target[target < self.geom.joint_min] += RAD_PER_TICK
        self._target = target

    def enable_torque(self) -> None:
        # Like a servo, hold the present position rather than chase an old target.
        self._target = _on_ticks(self._theta)
        self._velocity[:] = 0.0
        self._torque_on = True

    def disable_torque(self) -> None:
        # No physics: without torque the shoulders stop and stay where they are.
        self._velocity[:] = 0.0
        self._torque_on = False

    def close(self) -> None:
        pass

    def advance(self, dt: float) -> None:
        """Advance simulated time by dt seconds; each shoulder moves towards its target."""
        if not (np.isfinite(dt) and dt >= 0):
            raise ValueError(f"dt must be zero or more seconds and finite, got {dt}")
        if not self._torque_on:
            return
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
    theta: float, velocity: float, goal: float, dt: float, v_max: float, a_max: float
) -> tuple[float, float]:
    """Move one shoulder from angle theta and velocity towards goal for dt seconds.

    Time-optimal under |velocity| <= v_max and |acceleration| <= a_max, ending at
    rest on the goal. The motion is a chain of constant-acceleration segments, each
    stepped exactly. Returns the new angle and velocity.
    """
    while dt > 0:
        # Work in a frame where the goal lies ahead, in the + direction.
        ahead = 1.0 if goal >= theta else -1.0
        distance, speed = ahead * (goal - theta), ahead * velocity
        if distance <= _EPS and abs(speed) <= _EPS:
            return goal, 0.0
        stopping = speed * speed / (2 * a_max) if speed > 0 else 0.0
        if speed < 0:  # moving away: stop first
            accel, duration, lands = a_max, -speed / a_max, False
        elif stopping > distance + _EPS:  # too fast to stop on the goal: stop past it
            accel, duration, lands = -a_max, speed / a_max, False
        elif stopping >= distance - _EPS:  # brake onto the goal
            if distance <= _EPS:  # already on it, creeping slower than _EPS per second
                return goal, 0.0
            accel = -speed * speed / (2 * distance)
            duration, lands = 2 * distance / speed, True
        elif speed >= v_max - _EPS:  # cruise until it is time to brake
            accel, duration, lands = 0.0, (distance - stopping) / speed, False
        else:  # speed up, to the max velocity or until braking must start
            peak = min(v_max, np.sqrt(a_max * distance + speed * speed / 2))
            accel, duration, lands = a_max, (peak - speed) / a_max, False
        if lands and duration <= dt:
            return goal, 0.0
        step = min(duration, dt)
        theta += ahead * (speed * step + accel * step * step / 2)
        velocity = ahead * (speed + accel * step)
        dt -= step
    return theta, velocity
