import numpy as np
import pytest
from fake_servo import ACCELERATION, GOAL_SPEED, TORQUE_LIMIT, FakeServos
from joint_bus_contract import JointBusContract

from dros import DeltaGeometry
from dros.servo import STS, FeetechBus, rad_to_ticks

SERVO_IDS = (1, 2, 3)


def fake_servos_at(start_angles):
    return FakeServos(dict(zip(SERVO_IDS, rad_to_ticks(start_angles).tolist())))


class TestFeetechBusFollowsTheContract(JointBusContract):
    def make_bus(self, geom, start_angles, **limits):
        return FeetechBus(fake_servos_at(start_angles), geom, model=STS, **limits)


GEOM = DeltaGeometry()


def connect(servos, model=STS, max_velocity=3.0, max_acceleration=10.0, **options):
    return FeetechBus(
        servos,
        GEOM,
        model=model,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        **options,
    )


def register(servos, address):
    return [servos.registers[servo_id].get(address) for servo_id in SERVO_IDS]


# Worked by hand for the STS: 4096 ticks per revolution, speed in ticks/s,
# acceleration in 100 ticks/s^2, torque limit in 0.1 % of full torque.
# 3 rad/s = 1955.8 ticks/s and 10 rad/s^2 = 65.2 units, both rounded down.
def test_the_limits_are_converted_and_written_to_every_servo_on_connect():
    servos = fake_servos_at([0.1, 0.2, 0.3])
    connect(servos, max_velocity=3.0, max_acceleration=10.0, torque_limit=0.5)
    assert register(servos, GOAL_SPEED) == [1955, 1955, 1955]
    assert register(servos, ACCELERATION) == [65, 65, 65]
    assert register(servos, TORQUE_LIMIT) == [500, 500, 500]


def test_a_new_bus_writes_full_torque_unless_told_otherwise():
    servos = fake_servos_at([0.1, 0.2, 0.3])
    connect(servos)
    assert register(servos, TORQUE_LIMIT) == [1000, 1000, 1000]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, np.nan])
def test_a_bus_cannot_be_made_with_an_unusable_torque_limit(bad):
    with pytest.raises(ValueError, match="torque_limit"):
        connect(fake_servos_at([0.1, 0.2, 0.3]), torque_limit=bad)
