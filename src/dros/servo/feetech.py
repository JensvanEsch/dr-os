"""Feetech bus: the joint bus for real Feetech UART-bus servos (ADR 0004).

Ticks live only here. Above this module everything is SI. The Feetech bus talks to
the servos through a register port, so tests can put a fake servo behind it and
no test opens a serial port. The real register port uses the Feetech SDK behind
the optional `hw` extra.

The register addresses and per-model units come from memory of Feetech's STS and
HLS memory tables. Confirm them against the official tables at bring-up.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..geometry import DeltaGeometry
from .bus import check_motion_limits, check_target

TICKS_PER_REV = 4096
CENTER_TICK = 2048
RAD_PER_TICK = 2 * np.pi / TICKS_PER_REV


def rad_to_ticks(
    theta,
    zero_tick: int = CENTER_TICK,
    direction: int = 1,
    ticks_per_rev: int = TICKS_PER_REV,
) -> np.ndarray:
    """Joint angle (rad) -> raw position ticks, per-servo zero and direction."""
    ticks = zero_tick + direction * np.asarray(theta, dtype=float) * ticks_per_rev / (2 * np.pi)
    return np.rint(ticks).astype(int)


def ticks_to_rad(
    ticks,
    zero_tick: int = CENTER_TICK,
    direction: int = 1,
    ticks_per_rev: int = TICKS_PER_REV,
) -> np.ndarray:
    """Raw position ticks -> joint angle (rad)."""
    return direction * (np.asarray(ticks, dtype=float) - zero_tick) * 2 * np.pi / ticks_per_rev


class RegisterPort(Protocol):
    """Reads and writes servo registers by servo ID. The only part that touches a serial port."""

    def read(self, servo_id: int, address: int, size: int) -> int: ...
    def write(self, servo_id: int, address: int, size: int, value: int) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class FeetechModel:
    """What differs between Feetech servo families, for the registers the bus uses.

    ticks_per_rev:        encoder ticks per revolution
    speed_unit:           ticks/s per unit of the speed register
    acceleration_unit:    ticks/s^2 per unit of the acceleration register
    speed_register_max:   largest speed register value
    acceleration_register_max: largest acceleration register value
    full_torque_limit:    torque limit register value for the servo's full torque
    """

    name: str
    ticks_per_rev: int
    speed_unit: float
    acceleration_unit: float
    speed_register_max: int
    acceleration_register_max: int
    full_torque_limit: int


# Unconfirmed: from vendor wiki memory tables, to be checked on hardware (ADR 0004).
STS = FeetechModel(
    name="STS",
    ticks_per_rev=4096,
    speed_unit=1.0,
    acceleration_unit=100.0,
    speed_register_max=32767,
    acceleration_register_max=254,
    full_torque_limit=1000,
)
HLS = FeetechModel(
    name="HLS",
    ticks_per_rev=4096,
    speed_unit=1.0,
    acceleration_unit=100.0,
    speed_register_max=32767,
    acceleration_register_max=254,
    full_torque_limit=1000,
)

# Register addresses, believed shared by the STS and HLS families: (address, size).
_TORQUE_ENABLE = (40, 1)
_ACCELERATION = (41, 1)
_GOAL_POSITION = (42, 2)
_GOAL_SPEED = (46, 2)
_TORQUE_LIMIT = (48, 2)
_PRESENT_POSITION = (56, 2)


class FeetechBus:
    """Joint bus for three Feetech servos, one per shoulder, behind a register port.

    `servo_ids[i]` drives arm i. Angles are converted with the model's centre tick as zero
    and direction +1 until calibration exists.

    On connect the bus writes max velocity (rad/s), max acceleration (rad/s^2) and
    the torque limit to every servo, so the servos hold to them even if this
    process stalls (ADR 0005). `torque_limit` is a fraction of the servo's full
    torque, in (0, 1].

    The bus leaves the servos' torque as it finds it: a servo still holding from an
    earlier run keeps holding rather than dropping the end-effector. Targets are
    refused until `enable_torque`, which holds the present position first.
    """

    def __init__(
        self,
        port: RegisterPort,
        geom: DeltaGeometry,
        *,
        model: FeetechModel,
        max_velocity: float,
        max_acceleration: float,
        servo_ids: Sequence[int] = (1, 2, 3),
        torque_limit: float = 1.0,
    ) -> None:
        check_motion_limits(max_velocity, max_acceleration)
        if not 0 < torque_limit <= 1:  # False for NaN
            raise ValueError(f"torque_limit must be in (0, 1] of full torque, got {torque_limit}")
        if len(servo_ids) != 3:
            raise ValueError(f"a Feetech bus drives three shoulders, got IDs {servo_ids}")
        self.geom = geom
        self.model = model
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self._port = port
        self._ids = tuple(servo_ids)
        self._torque_on = False
        self._write_limits(torque_limit)

    def read_angles(self) -> np.ndarray:
        return self._to_rad([self._read(i, _PRESENT_POSITION) for i in self._ids])

    def write_angles(self, theta: np.ndarray) -> None:
        if not self._torque_on:
            raise RuntimeError("torque is off: turn it on before commanding the shoulders")
        ticks = self._to_ticks(check_target(self.geom, theta))
        # Rounding to ticks must not put a goal past the joint limits: step back inside.
        angles = self._to_rad(ticks)
        ticks[angles > self.geom.joint_max] -= 1
        ticks[angles < self.geom.joint_min] += 1
        for servo_id, goal in zip(self._ids, ticks):
            self._write(servo_id, _GOAL_POSITION, int(goal))

    def enable_torque(self) -> None:
        # A servo drives to its goal register when torque comes on, so first make
        # the goal its present position: it holds still rather than chase an old goal.
        for servo_id in self._ids:
            self._write(servo_id, _GOAL_POSITION, self._read(servo_id, _PRESENT_POSITION))
            self._write(servo_id, _TORQUE_ENABLE, 1)
        self._torque_on = True

    def disable_torque(self) -> None:
        self._torque_on = False
        for servo_id in self._ids:
            self._write(servo_id, _TORQUE_ENABLE, 0)

    def close(self) -> None:
        self._port.close()

    def _write_limits(self, torque_limit: float) -> None:
        """Write the motion and torque limits to every servo, in the model's units."""
        ticks_per_rad = self.model.ticks_per_rev / (2 * np.pi)
        speed = _register_value(
            self.max_velocity * ticks_per_rad / self.model.speed_unit, self.model.speed_register_max
        )
        acceleration = _register_value(
            self.max_acceleration * ticks_per_rad / self.model.acceleration_unit,
            self.model.acceleration_register_max,
        )
        torque = max(round(torque_limit * self.model.full_torque_limit), 1)  # 0 is no torque
        for servo_id in self._ids:
            self._write(servo_id, _GOAL_SPEED, speed)
            self._write(servo_id, _ACCELERATION, acceleration)
            self._write(servo_id, _TORQUE_LIMIT, torque)

    def _to_ticks(self, theta: ArrayLike) -> np.ndarray:
        per_rev = self.model.ticks_per_rev
        return rad_to_ticks(theta, zero_tick=per_rev // 2, ticks_per_rev=per_rev)

    def _to_rad(self, ticks: ArrayLike) -> np.ndarray:
        per_rev = self.model.ticks_per_rev
        return ticks_to_rad(ticks, zero_tick=per_rev // 2, ticks_per_rev=per_rev)

    def _read(self, servo_id: int, register: tuple[int, int]) -> int:
        return self._port.read(servo_id, *register)

    def _write(self, servo_id: int, register: tuple[int, int], value: int) -> None:
        self._port.write(servo_id, register[0], register[1], value)


def _register_value(units: float, largest: int) -> int:
    """A speed or acceleration in register units, rounded down and kept in [1, largest].

    Never 0: on the HLS a speed of 0 stops the servo, while on the STS it is believed
    to mean maximum speed (ADR 0004), and an acceleration of 0 no ramp at all. Above `largest` the register can't hold it,
    so the servo runs at its own maximum.
    """
    return min(max(int(units), 1), largest)
