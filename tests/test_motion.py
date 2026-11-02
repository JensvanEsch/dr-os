import numpy as np

from dros import DeltaGeometry, forward
from dros.motion import move_to
from dros.servo import SimBus

GEOM = DeltaGeometry()  # placeholder dimensions: nothing here waits on the real ones
POSITION_TOLERANCE = 0.5e-3  # m, about 2.5 times the worst tick rounding error


def sim_bus():
    bus = SimBus(GEOM, max_velocity=3.0, max_acceleration=10.0)
    bus.enable_torque()
    return bus


def position(bus):
    return forward(GEOM, bus.read_angles())


def test_move_to_reaches_a_position_on_simulated_time():
    bus = sim_bus()
    target = np.array([0.02, -0.01, -0.24])
    move_to(bus, GEOM, target, sleep=bus.advance)
    assert np.linalg.norm(position(bus) - target) <= POSITION_TOLERANCE
