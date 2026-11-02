import numpy as np
import pytest

from dros import DeltaGeometry, UnreachableError, forward, inverse
from dros.motion import (
    ShoulderOutsideLimitsError,
    set_end_effector_position,
    set_shoulder_angles,
)
from dros.servo import RAD_PER_TICK, SimBus

GEOM = DeltaGeometry()  # placeholder dimensions: nothing here waits on the real ones
POSITION_TOLERANCE = 0.5e-3  # m, about 2.5 times the worst tick rounding error
MAX_VELOCITY = 3.0  # rad/s
MAX_ACCELERATION = 10.0  # rad/s^2
# Simulated time for any move within the joint limits to arrive, at the limits above.
SETTLE_TIME = 2.0  # s


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
        self.positions_at_torque_off = []

    def write_angles(self, theta):
        self.targets.append(np.array(theta))
        self.positions_at_write.append(end_effector_position(self))
        super().write_angles(theta)

    def disable_torque(self):
        self.positions_at_torque_off.append(end_effector_position(self))
        super().disable_torque()


class CallLogBus(SimBus):
    """A sim bus that logs the name of every bus method called on it."""

    def __init__(self, **kwargs):
        super().__init__(GEOM, max_velocity=MAX_VELOCITY, max_acceleration=MAX_ACCELERATION, **kwargs)
        self.enable_torque()
        self.calls = []

    def read_angles(self):
        self.calls.append("read_angles")
        return super().read_angles()

    def write_angles(self, theta):
        self.calls.append("write_angles")
        super().write_angles(theta)

    def enable_torque(self):
        if hasattr(self, "calls"):
            self.calls.append("enable_torque")
        super().enable_torque()

    def disable_torque(self):
        self.calls.append("disable_torque")
        super().disable_torque()


UNREACHABLE = {
    "beyond the arms": np.array(GEOM.rest_position) - [0.0, 0.0, 1.0],
    "past joint_max": forward(GEOM, [GEOM.joint_max + 0.1] * 3),
    "past joint_min": forward(GEOM, [GEOM.joint_min - 0.1] * 3),
}


@pytest.mark.parametrize("position", UNREACHABLE.values(), ids=UNREACHABLE.keys())
def test_an_unreachable_position_raises_before_any_bus_command(position):
    bus = CallLogBus()
    with pytest.raises(UnreachableError):
        set_end_effector_position(bus, GEOM, position)
    assert bus.calls == []


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("axis", range(3))
def test_a_position_that_is_not_finite_raises_before_any_bus_command(bad, axis):
    bus = CallLogBus()
    position = np.array(GEOM.rest_position)
    position[axis] = bad
    with pytest.raises(ValueError, match="not finite"):
        set_end_effector_position(bus, GEOM, position)
    assert bus.calls == []


@pytest.mark.parametrize("arm", range(3))
@pytest.mark.parametrize("past", ["joint_min", "joint_max"])
def test_starting_outside_the_joint_limits_stops_and_asks_to_reposition_by_hand(arm, past):
    start = np.full(3, 0.5)
    start[arm] = GEOM.joint_min - 0.05 if past == "joint_min" else GEOM.joint_max + 0.05
    bus = CallLogBus(start_angles=start)
    with pytest.raises(ShoulderOutsideLimitsError, match=rf"arm {arm}.*by hand"):
        set_end_effector_position(bus, GEOM, GEOM.rest_position)
    assert "write_angles" not in bus.calls
    bus.advance(1.0)
    assert np.allclose(bus.read_angles(), start, rtol=0, atol=RAD_PER_TICK / 2)


def test_a_reading_that_is_not_finite_stops_without_asking_to_move_an_arm():
    class FailedReadBus(CallLogBus):
        def read_angles(self):
            return super().read_angles() * [1.0, np.nan, 1.0]

    bus = FailedReadBus(start_angles=[0.5, 0.5, 0.5])
    with pytest.raises(ShoulderOutsideLimitsError, match="arm 1.*not finite") as error:
        set_end_effector_position(bus, GEOM, GEOM.rest_position)
    assert "by hand" not in str(error.value)
    assert "write_angles" not in bus.calls


def test_a_shoulder_target_outside_the_joint_limits_raises_before_any_bus_command():
    bus = CallLogBus()
    with pytest.raises(ValueError, match="joint limits"):
        set_shoulder_angles(bus, GEOM, [0.5, GEOM.joint_max + 0.1, 0.5])
    assert bus.calls == []


def test_a_failure_after_sending_holds_the_last_target_with_torque_on():
    bus = CallLogBus()
    target = np.array(GEOM.rest_position)
    set_end_effector_position(bus, GEOM, target)
    bus.advance(0.05)  # part of the way there, the application fails
    assert np.linalg.norm(end_effector_position(bus) - target) > POSITION_TOLERANCE
    assert "disable_torque" not in bus.calls
    # The servos carry on to the last target on their own and hold it there.
    bus.advance(SETTLE_TIME)
    assert np.linalg.norm(end_effector_position(bus) - target) <= POSITION_TOLERANCE


def test_a_long_move_goes_out_as_one_joint_target():
    bus = RecordingBus()
    # From the shoulders at 0 rad to all three at about 1.17 rad.
    target = np.array([0.0, 0.0, -0.27])
    set_end_effector_position(bus, GEOM, target)
    bus.advance(SETTLE_TIME)
    assert len(bus.targets) == 1
    assert np.linalg.norm(end_effector_position(bus) - target) <= POSITION_TOLERANCE


def test_a_run_of_20_mm_down_from_rest_and_back_ends_at_rest_with_torque_off():
    # The milestone run of issue #8, composed from set_end_effector_position as an
    # application would: it waits for each target and ends the run itself.
    bus = RecordingBus()
    rest = np.array(GEOM.rest_position)
    down = rest - [0.0, 0.0, 0.020]
    for target in (rest, down, rest):
        set_end_effector_position(bus, GEOM, target)
        bus.advance(SETTLE_TIME)
    bus.disable_torque()

    commanded = np.array([forward(GEOM, t) for t in bus.targets])
    assert np.allclose(commanded, [rest, down, rest], rtol=0, atol=1e-9)
    for previous, at_write in zip([rest, down], bus.positions_at_write[1:]):
        assert np.linalg.norm(at_write - previous) <= POSITION_TOLERANCE
    # Torque went off once, with the end-effector at the rest position.
    [at_torque_off] = bus.positions_at_torque_off
    assert np.linalg.norm(at_torque_off - rest) <= POSITION_TOLERANCE
    with pytest.raises(RuntimeError, match="torque is off"):
        bus.write_angles(bus.read_angles())


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
def test_the_end_effector_lands_within_half_a_millimetre_of_each_target(seed):
    rng = np.random.default_rng(seed)
    bus = sim_bus()
    for target in reachable_positions(rng, 5):
        set_end_effector_position(bus, GEOM, target)
        bus.advance(SETTLE_TIME)
        assert np.linalg.norm(end_effector_position(bus) - target) <= POSITION_TOLERANCE


def test_set_shoulder_angles_sends_the_target_and_returns_without_waiting():
    bus = sim_bus()
    target = np.array([0.9, 1.1, 0.3])
    set_shoulder_angles(bus, GEOM, target)
    assert np.array_equal(bus.read_angles(), [0.0, 0.0, 0.0])  # no time has passed
    bus.advance(2.0)
    assert np.allclose(bus.read_angles(), target, rtol=0, atol=RAD_PER_TICK / 2)


def test_set_end_effector_position_sends_the_target_and_returns_without_waiting():
    bus = sim_bus()
    target = np.array([0.02, -0.01, -0.24])
    set_end_effector_position(bus, GEOM, target)
    assert np.array_equal(bus.read_angles(), [0.0, 0.0, 0.0])  # no time has passed
    bus.advance(2.0)
    assert np.linalg.norm(end_effector_position(bus) - target) <= POSITION_TOLERANCE
