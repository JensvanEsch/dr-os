import numpy as np
import pytest
from fake_servo import ACCELERATION, GOAL_POSITION, GOAL_SPEED, TORQUE_LIMIT, FakeServos
from joint_bus_contract import JointBusContract

from dros import DeltaGeometry
from dros.servo import HLS, RAD_PER_TICK, STS, FeetechBus, FeetechModel, rad_to_ticks

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


# 1e-4 rad/s is 0.07 ticks/s and 1e-3 rad/s^2 is 0.007 units: both round down to 0.
def test_a_limit_that_rounds_down_to_zero_is_sent_as_the_smallest_nonzero_value():
    servos = fake_servos_at([0.1, 0.2, 0.3])
    connect(servos, max_velocity=1e-4, max_acceleration=1e-3)
    assert register(servos, GOAL_SPEED) == [1, 1, 1]
    assert register(servos, ACCELERATION) == [1, 1, 1]


def test_speed_and_acceleration_zero_are_never_sent():
    servos = fake_servos_at([0.1, 0.2, 0.3])
    bus = connect(servos, max_velocity=1e-6, max_acceleration=1e-6)
    bus.enable_torque()
    bus.write_angles([0.2, 0.3, 0.4])
    bus.disable_torque()
    sent_zero = [w for w in servos.writes if w[1] in (GOAL_SPEED, ACCELERATION) and w[2] == 0]
    assert sent_zero == []


# 100 rad/s is 65190 ticks/s and 1000 rad/s^2 is 6519 units: past what the STS registers hold.
def test_a_limit_beyond_the_register_is_capped_at_the_largest_value_it_holds():
    servos = fake_servos_at([0.1, 0.2, 0.3])
    connect(servos, max_velocity=100.0, max_acceleration=1000.0)
    assert register(servos, GOAL_SPEED) == [32767, 32767, 32767]
    assert register(servos, ACCELERATION) == [254, 254, 254]


# A made-up model, so the numbers below don't depend on the STS or HLS table:
# 1024 ticks per revolution, speed in 50 ticks/s, acceleration in 10 ticks/s^2.
COARSE = FeetechModel(
    name="coarse",
    ticks_per_rev=1024,
    speed_unit=50.0,
    acceleration_unit=10.0,
    max_speed=1000,
    max_acceleration=100,
    full_torque_limit=200,
)


def test_the_limits_are_written_in_the_units_of_the_servo_model():
    # 3 rad/s = 488.9 ticks/s = 9.8 units; 10 rad/s^2 = 1629.7 ticks/s^2 = 163 units, capped.
    servos = fake_servos_at([0.1, 0.2, 0.3])
    connect(servos, model=COARSE, max_velocity=3.0, max_acceleration=10.0, torque_limit=0.5)
    assert register(servos, GOAL_SPEED) == [9, 9, 9]
    assert register(servos, ACCELERATION) == [100, 100, 100]
    assert register(servos, TORQUE_LIMIT) == [100, 100, 100]


def test_angles_are_converted_with_the_tick_resolution_of_the_servo_model():
    # At 1024 ticks per revolution the centre is tick 512, and a quarter turn is 256 ticks.
    servos = FakeServos({1: 512 + 256, 2: 512, 3: 512 - 128})
    bus = connect(servos, model=COARSE)
    assert np.allclose(bus.read_angles(), [np.pi / 2, 0.0, -np.pi / 4])
    bus.enable_torque()
    bus.write_angles([np.pi / 4, 0.0, 0.0])
    assert register(servos, GOAL_POSITION) == [512 + 128, 512, 512]


@pytest.mark.parametrize("model", [STS, HLS], ids=["STS", "HLS"])
def test_the_sts_and_hls_families_both_drive_a_feetech_bus(model):
    servos = fake_servos_at([0.1, 0.2, 0.3])
    bus = connect(servos, model=model)
    bus.enable_torque()
    bus.write_angles([0.2, 0.3, 0.4])
    assert np.allclose(bus.read_angles(), [0.2, 0.3, 0.4], atol=RAD_PER_TICK)
    assert all(value > 0 for value in register(servos, GOAL_SPEED) + register(servos, ACCELERATION))
