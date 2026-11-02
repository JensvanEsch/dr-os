"""Fake servo: Feetech servos held in memory behind the register port. Opens no serial port.

The register addresses are written out here on purpose, rather than imported from
`dros.servo.feetech`, so that the fake stands for the servo's own memory table and
a wrong address in the Feetech bus shows up as a failing test. They come from
memory of Feetech's STS table and still need confirming on hardware.
"""
from __future__ import annotations

TORQUE_ENABLE = 40  # 1 byte
ACCELERATION = 41  # 1 byte
GOAL_POSITION = 42  # 2 bytes
GOAL_SPEED = 46  # 2 bytes
TORQUE_LIMIT = 48  # 2 bytes
PRESENT_POSITION = 56  # 2 bytes


class FakeServos:
    """Servos by ID that hold registers and move to their goal instantly while torque is on.

    Like a real servo, turning torque on drives it to whatever the goal position
    register holds, even if that is an old goal.
    """

    def __init__(self, present_ticks: dict[int, int]) -> None:
        self.registers = {
            servo_id: {TORQUE_ENABLE: 0, GOAL_POSITION: ticks, PRESENT_POSITION: ticks}
            for servo_id, ticks in present_ticks.items()
        }
        self.writes: list[tuple[int, int, int]] = []  # (servo_id, address, value)
        self.closed = False

    def read(self, servo_id: int, address: int, size: int) -> int:
        return self._servo(servo_id).get(address, 0)

    def write(self, servo_id: int, address: int, size: int, value: int) -> None:
        servo = self._servo(servo_id)
        if not 0 <= value < 256**size:
            raise ValueError(f"{value} does not fit in {size} bytes at address {address}")
        self.writes.append((servo_id, address, value))
        servo[address] = value
        if servo[TORQUE_ENABLE]:
            servo[PRESENT_POSITION] = servo[GOAL_POSITION]

    def close(self) -> None:
        self.closed = True

    def move_by_hand(self, servo_id: int, ticks: int) -> None:
        """Turn a servo's output to `ticks`, as an operator does with torque off."""
        self._servo(servo_id)[PRESENT_POSITION] = ticks

    def _servo(self, servo_id: int) -> dict[int, int]:
        if self.closed:
            raise RuntimeError("the port is closed")
        if servo_id not in self.registers:
            raise TimeoutError(f"no servo answers on ID {servo_id}")
        return self.registers[servo_id]
