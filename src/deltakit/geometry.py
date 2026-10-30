"""Robot dimensions and joint limits. The only place where geometry numbers live."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeltaGeometry:
    """Rotary delta robot geometry, SI units.

    base_radius:     centre of base -> shoulder joint axis (m)
    effector_radius: centre of effector -> wrist (lower-arm attach) point (m)
    upper_arm:       shoulder axis -> elbow (m)
    lower_arm:       elbow -> wrist, parallelogram length (m)
    joint_min/max:   joint limits (rad), theta = 0 is horizontal, + is down

    The defaults are PLACEHOLDERS until the printed robot is measured.
    """

    base_radius: float = 0.060
    effector_radius: float = 0.025
    upper_arm: float = 0.090
    lower_arm: float = 0.200
    joint_min: float = -0.6
    joint_max: float = 1.4

    @property
    def arm_angles(self) -> np.ndarray:
        """Angle of each arm around z (rad): 0, 120, 240 deg."""
        return np.deg2rad([0.0, 120.0, 240.0])

    def check_joint_limits(self, theta: np.ndarray) -> None:
        """Raise ValueError if any joint angle is outside the limits."""
        theta = np.asarray(theta, dtype=float)
        bad = (theta < self.joint_min) | (theta > self.joint_max)
        if np.any(bad):
            raise ValueError(
                f"joint limits [{self.joint_min}, {self.joint_max}] rad violated: {theta}"
            )
