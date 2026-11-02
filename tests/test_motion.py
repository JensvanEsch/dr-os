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


class RecordingBus(SimBus):
    """A sim bus that records each target and where the end-effector was when it came."""

    def __init__(self):
        super().__init__(GEOM, max_velocity=3.0, max_acceleration=10.0)
        self.enable_torque()
        self.targets = []
        self.positions_at_write = []

    def write_angles(self, theta):
        self.targets.append(np.array(theta))
        self.positions_at_write.append(position(self))
        super().write_angles(theta)


def test_a_long_move_goes_out_as_one_joint_target():
    bus = RecordingBus()
    # From the shoulders at 0 rad to all three at about 1.17 rad.
    target = np.array([0.0, 0.0, -0.27])
    move_to(bus, GEOM, target, sleep=bus.advance)
    assert len(bus.targets) == 1
    assert np.linalg.norm(position(bus) - target) <= POSITION_TOLERANCE


def test_each_target_is_reached_before_the_next_move_starts():
    bus = RecordingBus()
    targets = [
        np.array([0.0, 0.0, -0.25]),
        np.array([0.05, 0.03, -0.20]),
        np.array([0.03, 0.03, -0.26]),
    ]
    for target in targets:
        move_to(bus, GEOM, target, sleep=bus.advance)
    for previous, at_write in zip(targets, bus.positions_at_write[1:]):
        assert np.linalg.norm(at_write - previous) <= POSITION_TOLERANCE


def test_move_to_reaches_a_position_on_simulated_time():
    bus = sim_bus()
    target = np.array([0.02, -0.01, -0.24])
    move_to(bus, GEOM, target, sleep=bus.advance)
    assert np.linalg.norm(position(bus) - target) <= POSITION_TOLERANCE
