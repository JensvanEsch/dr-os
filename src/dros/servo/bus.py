"""Hardware-independent joint bus interface. Real drivers implement JointBus."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..geometry import DeltaGeometry


class JointBus(Protocol):
    def read_angles(self) -> np.ndarray: ...
    def write_angles(self, theta: np.ndarray) -> None: ...


class MockBus:
    """In-memory bus for development and tests. Moves instantly."""

    def __init__(self) -> None:
        self._theta = np.zeros(3)

    def read_angles(self) -> np.ndarray:
        return self._theta.copy()

    def write_angles(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, dtype=float).copy()


def safe_write(bus: JointBus, geom: DeltaGeometry, theta) -> None:
    """The only sanctioned path to command joints: limits are checked first."""
    theta = np.asarray(theta, dtype=float)
    geom.check_joint_limits(theta)
    bus.write_angles(theta)
