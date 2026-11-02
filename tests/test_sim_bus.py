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
