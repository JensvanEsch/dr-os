import time

import numpy as np
import pytest

from dros import DeltaGeometry, forward
from dros.motion import first_move, move_to
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


def test_the_first_move_goes_20_mm_down_and_back_and_ends_at_rest_with_torque_off():
    bus = RecordingBus()
    rest = np.array(GEOM.rest_position)
    down = rest - [0.0, 0.0, 0.020]
    first_move(bus, GEOM, sleep=bus.advance)

    commanded = np.array([forward(GEOM, t) for t in bus.targets])
    assert np.allclose(commanded, [rest, down, rest], rtol=0, atol=1e-9)
    # Each move started only once the one before it had arrived.
    for previous, at_write in zip([rest, down], bus.positions_at_write[1:]):
        assert np.linalg.norm(at_write - previous) <= POSITION_TOLERANCE
    assert np.linalg.norm(position(bus) - rest) <= POSITION_TOLERANCE
    with pytest.raises(RuntimeError, match="torque is off"):
        bus.write_angles(bus.read_angles())


def test_the_first_move_turns_torque_on_itself():
    bus = SimBus(GEOM, max_velocity=3.0, max_acceleration=10.0)
    first_move(bus, GEOM, sleep=bus.advance)
    assert np.linalg.norm(position(bus) - GEOM.rest_position) <= POSITION_TOLERANCE


def test_the_first_move_runs_on_virtual_time_with_no_real_waiting():
    bus = SimBus(GEOM, max_velocity=3.0, max_acceleration=10.0)
    simulated = []

    def sleep(dt):
        simulated.append(dt)
        bus.advance(dt)

    started = time.monotonic()
    first_move(bus, GEOM, sleep=sleep)
    # Coming from the shoulders at 0 rad takes about half a second of simulated time.
    assert sum(simulated) > 0.3
    assert time.monotonic() - started < sum(simulated) / 10


def test_move_to_reaches_a_position_on_simulated_time():
    bus = sim_bus()
    target = np.array([0.02, -0.01, -0.24])
    move_to(bus, GEOM, target, sleep=bus.advance)
    assert np.linalg.norm(position(bus) - target) <= POSITION_TOLERANCE
