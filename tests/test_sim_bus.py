import time

import numpy as np
import pytest
from joint_bus_contract import JointBusContract

from dros import DeltaGeometry
from dros.servo import RAD_PER_TICK, SimBus

GEOM = DeltaGeometry()
MAX_VELOCITY = 3.0  # rad/s
MAX_ACCELERATION = 10.0  # rad/s^2


def sim_bus(start_angles=(0.0, 0.0, 0.0)):
    bus = SimBus(
        GEOM,
        max_velocity=MAX_VELOCITY,
        max_acceleration=MAX_ACCELERATION,
        start_angles=start_angles,
    )
    bus.enable_torque()
    return bus


def reached(bus, target):
    return np.allclose(bus.read_angles(), target, atol=RAD_PER_TICK)


class TestSimBusFollowsTheContract(JointBusContract):
    def make_bus(self, geom, start_angles, **limits):
        return SimBus(geom, start_angles=start_angles, **limits)

    def settle(self, bus):
        bus.advance(10.0)


# Worked by hand from 3 rad/s and 10 rad/s^2, starting and ending at rest.
# 1.0 rad: speeds up for 0.3 s, cruises, slows down for 0.3 s: 1/3 + 0.3 s.
# 0.4 rad: peaks at 2 rad/s, below the max velocity: 2 * sqrt(0.4 / 10) s.
@pytest.mark.parametrize(
    "distance, duration",
    [(1.0, 1.0 / 3.0 + 0.3), (0.4, 0.4)],
    ids=["reaches max velocity", "never reaches max velocity"],
)
def test_a_move_takes_the_time_its_velocity_and_acceleration_limits_imply(distance, duration):
    target = [distance, 0.0, 0.0]
    bus = sim_bus()
    bus.write_angles(target)
    # 0.05 s before the end, braking still has 0.5 * 10 * 0.05^2 = 12.5 mrad to go.
    bus.advance(duration - 0.05)
    assert not reached(bus, target)
    bus.advance(0.05)
    assert reached(bus, target)


def on_ticks(angles):
    ticks = np.asarray(angles) / RAD_PER_TICK
    return np.allclose(ticks, np.rint(ticks), rtol=0, atol=1e-9)


def test_read_back_angles_are_rounded_to_ticks_at_rest_and_while_moving():
    bus = sim_bus(start_angles=(0.1234567, 0.2345678, 0.3456789))
    assert on_ticks(bus.read_angles())
    bus.write_angles([0.7654321, 0.2345678, 0.1])
    bus.advance(0.123)
    assert on_ticks(bus.read_angles())
    bus.advance(10.0)
    assert on_ticks(bus.read_angles())


def test_with_torque_off_the_sim_bus_does_not_move():
    bus = sim_bus()
    bus.write_angles([1.0, 0.5, 0.0])
    bus.advance(0.2)
    stopped_at = bus.read_angles()
    bus.disable_torque()
    bus.advance(10.0)
    assert np.array_equal(bus.read_angles(), stopped_at)


def test_turning_torque_back_on_holds_the_shoulders_where_they_are():
    bus = sim_bus()
    bus.write_angles([1.0, 0.5, 0.0])
    bus.advance(0.2)
    bus.disable_torque()
    stopped_at = bus.read_angles()
    bus.enable_torque()
    bus.advance(10.0)
    assert np.allclose(bus.read_angles(), stopped_at, atol=RAD_PER_TICK)


def test_simulated_time_advances_only_through_an_explicit_call_without_real_waiting():
    bus = sim_bus()
    bus.write_angles([1.0, 0.5, 0.0])
    assert reached(bus, [0.0, 0.0, 0.0])
    started = time.monotonic()
    bus.advance(60.0)
    assert time.monotonic() - started < 0.5
    assert reached(bus, [1.0, 0.5, 0.0])


@pytest.mark.parametrize("dt", [-0.1, np.nan, np.inf])
def test_simulated_time_cannot_run_backwards_or_jump_by_an_unusable_step(dt):
    bus = sim_bus()
    with pytest.raises(ValueError, match="dt"):
        bus.advance(dt)
