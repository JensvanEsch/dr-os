"""Workspace analysis: which positions a geometry can reach, and if they cover a target."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil

import numpy as np

from .geometry import DeltaGeometry
from .kinematics import UnreachableError, forward, inverse

# How densely a target cylinder is sampled. Rings are a fixed distance apart because
# joint_min leaves small round holes in the workspace that wider rings can step over.
# The angles are multiples of 15 deg so they land on the arm directions and halfway
# between them, where the workspace is extreme.
_LAYER_SPACING = 0.002  # m
_RING_SPACING = 0.010  # m
_RING_ANGLES = np.deg2rad(np.arange(0.0, 360.0, 15.0))


class Cause(Enum):
    """Why the end-effector cannot reach a position."""

    ARM_LENGTH = "arm length"
    JOINT_LIMITS = "joint limits"
    UPPER_SOLUTION = "upper forward solution"


def unreachable_cause(geom: DeltaGeometry, position) -> Cause | None:
    """Why the end-effector cannot reach position (m), or None if it can.

    forward() returns the lower of its two solutions, so a position that only the
    upper one gives could never be read back. Those count as out of reach.
    """
    position = np.asarray(position, dtype=float)
    try:
        theta = inverse(geom, position)
    except UnreachableError:
        return Cause.ARM_LENGTH
    try:
        geom.check_joint_limits(theta)
    except ValueError:
        return Cause.JOINT_LIMITS
    try:
        back = forward(geom, theta)
    except UnreachableError:
        return Cause.UPPER_SOLUTION
    return None if np.allclose(back, position, rtol=0.0, atol=1e-8) else Cause.UPPER_SOLUTION


def reachable(geom: DeltaGeometry, positions) -> np.ndarray:
    """Which positions (m, shape (..., 3)) the end-effector can reach (bool, shape (...))."""
    positions = np.asarray(positions, dtype=float)
    flat = [unreachable_cause(geom, p) is None for p in positions.reshape(-1, 3)]
    return np.array(flat, dtype=bool).reshape(positions.shape[:-1])


@dataclass(frozen=True)
class TargetCylinder:
    """Work area a geometry must cover: upright and centred on the z axis, SI units.

    The z placement is not part of the target; check_cylinder finds the best one.
    """

    diameter: float = 0.160
    height: float = 0.100

    def __post_init__(self) -> None:
        if self.diameter <= 0.0 or self.height <= 0.0:
            raise ValueError(f"target dimensions must be positive: {self}")


@dataclass(frozen=True)
class Shortfall:
    """A position inside the target that the end-effector cannot reach."""

    position: tuple[float, float, float]
    cause: Cause


@dataclass(frozen=True)
class CylinderCheck:
    """Result of check_cylinder, at the best z placement found.

    centre_z:   centre of the target at that placement (m)
    samples:    number of positions sampled there
    shortfalls: the sampled positions that are out of reach there
    """

    target: TargetCylinder
    centre_z: float
    samples: int
    shortfalls: tuple[Shortfall, ...]

    @property
    def passed(self) -> bool:
        return not self.shortfalls

    def __str__(self) -> str:
        size = f"{self.target.diameter:.3f} m wide, {self.target.height:.3f} m tall cylinder"
        placement = f"centred at z = {self.centre_z:.3f} m"
        if self.passed:
            return f"PASS: {size} fits, {placement}"
        count = f"{len(self.shortfalls)} of {self.samples} sampled positions"
        lines = [f"FAIL: {size} does not fit. Best is {placement}, {count} out of reach:"]
        for cause in Cause:
            hits = [s.position for s in self.shortfalls if s.cause is cause]
            if hits:
                x, y, z = np.array(hits).T
                radius = np.hypot(x, y)
                lines.append(
                    f"  {cause.value}: {len(hits)} positions, "
                    f"z {z.min():.3f} to {z.max():.3f} m, "
                    f"radius {radius.min():.3f} to {radius.max():.3f} m"
                )
        return "\n".join(lines)


DEFAULT_TARGET = TargetCylinder()


def _disc(radius: float) -> np.ndarray:
    """Sample (x, y) offsets across a disc (m, shape (N, 2)), centre included."""
    offsets = [(0.0, 0.0)]
    for r in np.linspace(0.0, radius, ceil(radius / _RING_SPACING) + 1)[1:]:
        offsets.extend(zip(r * np.cos(_RING_ANGLES), r * np.sin(_RING_ANGLES)))
    return np.array(offsets)


def check_cylinder(geom: DeltaGeometry, target: TargetCylinder = DEFAULT_TARGET) -> CylinderCheck:
    """Check whether geom can reach all of target, sliding it along z to find the best fit.

    The target passes if some placement has no sampled position out of reach. Placements
    are tried one layer spacing apart, so a target that fits with less than that to
    spare can fail.
    """
    intervals = ceil(target.height / _LAYER_SPACING)
    spacing = target.height / intervals
    depth = max(geom.upper_arm + geom.lower_arm, target.height)
    layers = -spacing * np.arange(ceil(depth / spacing) + 1)
    disc = _disc(target.diameter / 2.0)

    causes = [[unreachable_cause(geom, [x, y, z]) for x, y in disc] for z in layers]
    misses = np.array([sum(c is not None for c in layer) for layer in causes])

    # A placement spans intervals + 1 neighbouring layers. Of the equally good ones
    # take the middle, which leaves the most room when several placements pass.
    totals = np.convolve(misses, np.ones(intervals + 1, dtype=int), mode="valid")
    best = np.flatnonzero(totals == totals.min())
    first = int(best[len(best) // 2])
    last = first + intervals

    shortfalls = tuple(
        Shortfall((float(x), float(y), float(layers[i])), cause)
        for i in range(first, last + 1)
        for (x, y), cause in zip(disc, causes[i])
        if cause is not None
    )
    return CylinderCheck(
        target=target,
        centre_z=float((layers[first] + layers[last]) / 2.0),
        samples=(intervals + 1) * len(disc),
        shortfalls=shortfalls,
    )
