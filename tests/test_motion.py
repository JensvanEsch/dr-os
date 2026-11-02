import time

import numpy as np
import pytest

from dros import DeltaGeometry, forward, inverse
from dros.motion import set_end_effector_position, set_shoulder_angles
from dros.servo import RAD_PER_TICK, SimBus

GEOM = DeltaGeometry()  # placeholder dimensions: nothing here waits on the real ones
POSITION_TOLERANCE = 0.5e-3  # m, about 2.5 times the worst tick rounding error
MAX_VELOCITY = 3.0  # rad/s
MAX_ACCELERATION = 10.0  # rad/s^2


def sim_bus():
    bus = SimBus(GEOM, max_velocity=MAX_VELOCITY, max_acceleration=MAX_ACCELERATION)
    bus.enable_torque()
    return bus


def end_effector_position(bus):
    return forward(GEOM, bus.read_angles())


class RecordingBus(SimBus):
    """A sim bus that records each target and where the end-effector was when it came."""

    def __init__(self):
        super().__init__(GEOM, max_velocity=MAX_VELOCITY, max_acceleration=MAX_ACCELERATION)
        self.enable_torque()
        self.targets = []
        self.positions_at_write = []

    def write_angles(self, theta):
        self.targets.append(np.array(theta))
        self.positions_at_write.append(end_effector_position(self))
        super().write_angles(theta)


def test_a_long_move_goes_out_as_one_joint_target():
    bus = RecordingBus()
    # From the shoulders at 0 rad to all three at about 1.17 rad.
    target = np.array([0.0, 0.0, -0.27])
    set_end_effector_position(bus, GEOM, target, sleep=bus.advance)
    assert len(bus.targets) == 1
    assert np.linalg.norm(end_effector_position(bus) - target) <= POSITION_TOLERANCE


def test_each_target_is_reached_before_the_next_move_starts():
    bus = RecordingBus()
    targets = [
        np.array([0.0, 0.0, -0.25]),
        np.array([0.05, 0.03, -0.20]),
        np.array([0.03, 0.03, -0.26]),
    ]
    for target in targets:
        set_end_effector_position(bus, GEOM, target, sleep=bus.advance)
    for previous, at_write in zip(targets, bus.positions_at_write[1:]):
        assert np.linalg.norm(at_write - previous) <= POSITION_TOLERANCE


def test_a_run_of_20_mm_down_from_rest_and_back_ends_at_rest_with_torque_off():
    # The milestone run of issue #8, composed from set_end_effector_position as an application would.
    bus = RecordingBus()
    rest = np.array(GEOM.rest_position)
    down = rest - [0.0, 0.0, 0.020]
    for target in (rest, down, rest):
        set_end_effector_position(bus, GEOM, target, sleep=bus.advance)
    bus.disable_torque()

    commanded = np.array([forward(GEOM, t) for t in bus.targets])
    assert np.allclose(commanded, [rest, down, rest], rtol=0, atol=1e-9)
    # Each move started only once the one before it had arrived.
    for previous, at_write in zip([rest, down], bus.positions_at_write[1:]):
        assert np.linalg.norm(at_write - previous) <= POSITION_TOLERANCE
    assert np.linalg.norm(end_effector_position(bus) - rest) <= POSITION_TOLERANCE
    with pytest.raises(RuntimeError, match="torque is off"):
        bus.write_angles(bus.read_angles())


def test_set_end_effector_position_runs_on_virtual_time_with_no_real_waiting():
    bus = sim_bus()
    simulated = []

    def sleep(dt):
        simulated.append(dt)
        bus.advance(dt)

    started = time.monotonic()
    set_end_effector_position(bus, GEOM, GEOM.rest_position, sleep=sleep)
    # Coming from the shoulders at 0 rad takes about half a second of simulated time.
    assert sum(simulated) > 0.3
    assert time.monotonic() - started < 0.1


def reachable_positions(rng, count):
    """`count` random positions near the rest position that lie inside the joint limits."""
    positions = []
    while len(positions) < count:
        p = np.array(GEOM.rest_position) + rng.uniform(-0.04, 0.04, 3)
        try:
            GEOM.check_joint_limits(inverse(GEOM, p))
        except ValueError:  # UnreachableError is a ValueError too
            continue
        positions.append(p)
    return positions


@pytest.mark.parametrize("seed", range(20))
def test_set_end_effector_position_returns_with_the_shoulders_at_rest_on_the_target(seed):
    rng = np.random.default_rng(seed)
    bus = sim_bus()
    for target in reachable_positions(rng, 5):
        set_end_effector_position(bus, GEOM, target, sleep=bus.advance)
        arrived = bus.read_angles()
        bus.advance(1.0)
        assert np.array_equal(bus.read_angles(), arrived)
        assert np.linalg.norm(end_effector_position(bus) - target) <= POSITION_TOLERANCE


def test_set_end_effector_position_reaches_a_position_on_simulated_time():
    bus = sim_bus()
    target = np.array([0.02, -0.01, -0.24])
    set_end_effector_position(bus, GEOM, target, sleep=bus.advance)
    assert np.linalg.norm(end_effector_position(bus) - target) <= POSITION_TOLERANCE


def test_set_shoulder_angles_returns_with_the_shoulders_at_rest_on_the_target():
    bus = sim_bus()
    target = np.array([0.9, 1.1, 0.3])
    set_shoulder_angles(bus, target, sleep=bus.advance)
    arrived = bus.read_angles()
    bus.advance(1.0)
    assert np.array_equal(bus.read_angles(), arrived)
    # The sim bus reads back whole ticks: within half a tick of the target.
    assert np.allclose(arrived, target, rtol=0, atol=RAD_PER_TICK / 2)
