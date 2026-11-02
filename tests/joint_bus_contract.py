"""The rules every joint bus must follow (ADR 0005), written once and reused per bus.

A bus's test module subclasses `JointBusContract` as `Test...` and implements
`make_bus`. A bus whose motion takes time overrides `settle` to let it reach its
target, for example by advancing simulated time.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

from dros import DeltaGeometry
from dros.servo import RAD_PER_TICK, JointBus

GEOM = DeltaGeometry()
LIMITS = {"max_speed": 3.0, "max_acceleration": 10.0, "far_target_threshold": 0.2}
FAR_TARGET_THRESHOLD = LIMITS["far_target_threshold"]  # rad
# A bus may round what it reads back to one tick.
READ_BACK_TOLERANCE = RAD_PER_TICK


class JointBusContract:
    def make_bus(
        self,
        geom: DeltaGeometry,
        start_angles: ArrayLike,
        *,
        max_speed: float,
        max_acceleration: float,
        far_target_threshold: float,
    ) -> JointBus:
        """A bus with its shoulders at `start_angles` (rad) and torque off.

        The limits are in rad/s, rad/s^2 and rad, passed on to the bus's constructor.
        """
        raise NotImplementedError

    def settle(self, bus: JointBus) -> None:
        """Let the bus reach its last target."""

    def bus(self, start_angles: ArrayLike = (0.1, 0.2, 0.3)) -> JointBus:
        return self.make_bus(GEOM, start_angles, **LIMITS)

    def test_a_safe_target_is_reached(self):
        bus = self.bus()
        bus.enable_torque()
        bus.write_angles([0.2, 0.3, 0.4])
        self.settle(bus)
        assert np.allclose(bus.read_angles(), [0.2, 0.3, 0.4], atol=READ_BACK_TOLERANCE)

    def test_a_new_bus_refuses_targets_until_torque_is_on(self):
        bus = self.bus()
        with pytest.raises(RuntimeError):
            bus.write_angles([0.2, 0.3, 0.4])
        self.settle(bus)
        assert np.allclose(bus.read_angles(), [0.1, 0.2, 0.3], atol=READ_BACK_TOLERANCE)

    # Each target is within the far-target threshold of its start, so only the limits refuse it.
    @pytest.mark.parametrize(
        "start, target",
        [
            ([0.3, 0.3, GEOM.joint_max - 0.05], [0.3, 0.3, GEOM.joint_max + 0.05]),
            ([GEOM.joint_min + 0.05, 0.3, 0.3], [GEOM.joint_min - 0.05, 0.3, 0.3]),
            ([0.3, 0.3, 0.3], [0.3, np.nan, 0.3]),
        ],
        ids=["above max", "below min", "nan"],
    )
    def test_a_target_outside_the_joint_limits_is_refused(self, start, target):
        bus = self.bus(start)
        bus.enable_torque()
        with pytest.raises(ValueError, match="joint limits"):
            bus.write_angles(target)
        self.settle(bus)
        assert np.allclose(bus.read_angles(), start, atol=READ_BACK_TOLERANCE)

    def test_a_target_further_than_the_threshold_from_the_measured_angles_is_refused(self):
        bus = self.bus([0.1, 0.2, 0.3])
        bus.enable_torque()
        with pytest.raises(ValueError, match="far"):
            bus.write_angles([0.1, 0.2 - (FAR_TARGET_THRESHOLD + 0.01), 0.3])
        self.settle(bus)
        assert np.allclose(bus.read_angles(), [0.1, 0.2, 0.3], atol=READ_BACK_TOLERANCE)

    def test_a_target_just_inside_the_threshold_is_accepted(self):
        bus = self.bus([0.1, 0.2, 0.3])
        bus.enable_torque()
        target = np.array([0.1, 0.2, 0.3]) + 0.99 * FAR_TARGET_THRESHOLD
        bus.write_angles(target)
        self.settle(bus)
        assert np.allclose(bus.read_angles(), target, atol=READ_BACK_TOLERANCE)

    def test_the_threshold_is_measured_from_where_the_shoulders_are(self):
        # Stepping within the threshold is how long moves are made (ADR 0005).
        bus = self.bus([0.1, 0.1, 0.1])
        bus.enable_torque()
        for step in range(1, 5):
            target = np.full(3, 0.1 + step * 0.9 * FAR_TARGET_THRESHOLD)
            bus.write_angles(target)
            self.settle(bus)
        assert np.allclose(bus.read_angles(), target, atol=READ_BACK_TOLERANCE)

    @pytest.mark.parametrize("target", [[0.2], [0.2, 0.3], [0.2, 0.3, 0.4, 0.5]])
    def test_a_target_without_three_angles_is_refused(self, target):
        bus = self.bus()
        bus.enable_torque()
        with pytest.raises(ValueError, match="three"):
            bus.write_angles(target)

    @pytest.mark.parametrize("name", LIMITS)
    @pytest.mark.parametrize("bad", [0.0, -0.1, np.nan, np.inf])
    def test_a_bus_cannot_be_made_with_an_unusable_limit(self, name, bad):
        with pytest.raises(ValueError, match=name):
            self.make_bus(GEOM, (0.1, 0.2, 0.3), **{**LIMITS, name: bad})

    def test_a_bus_closes_after_use(self):
        bus = self.bus()
        bus.enable_torque()
        bus.write_angles([0.2, 0.3, 0.4])
        self.settle(bus)
        bus.disable_torque()
        bus.close()

    def test_targets_are_refused_after_torque_is_turned_off(self):
        bus = self.bus()
        bus.enable_torque()
        bus.disable_torque()
        with pytest.raises(RuntimeError):
            bus.write_angles([0.2, 0.3, 0.4])

    def test_targets_are_accepted_again_once_torque_is_back_on(self):
        bus = self.bus()
        bus.enable_torque()
        bus.disable_torque()
        bus.enable_torque()
        bus.write_angles([0.2, 0.3, 0.4])
        self.settle(bus)
        assert np.allclose(bus.read_angles(), [0.2, 0.3, 0.4], atol=READ_BACK_TOLERANCE)
