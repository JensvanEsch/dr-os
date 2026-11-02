"""Workspace analysis: what a geometry can reach, and the shoulder torque it takes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import ceil

import numpy as np
from numpy.typing import ArrayLike

from .geometry import DeltaGeometry
from .kinematics import UnreachableError, forward, inverse, jacobian, singularity_margin
from .servo import RAD_PER_TICK

# How densely a target cylinder is sampled. Rings are a fixed distance apart because
# joint_min leaves small round holes in the workspace that wider rings can step over.
# The angles are multiples of 15 deg so they land on the arm directions and halfway
# between them, where the workspace is extreme.
_LAYER_SPACING = 0.002  # m
_RING_SPACING = 0.010  # m
_RING_ANGLES = np.deg2rad(np.arange(0.0, 360.0, 15.0))

GRAVITY = 9.81  # m/s^2

# The sweep for worst cases. Arms sit 120 deg apart and each mirrors the workspace, so the
# 60 deg wedge with all three shoulders stands in for the rest.
_SWEEP_SPACING = 0.005  # m
_SWEEP_ANGLES = np.deg2rad(np.arange(0.0, 61.0, 15.0))

# Below this singularity_margin the sweep leaves a position out. Torque grows without bound
# there, so a sampled maximum would only say how close a sample happened to land.
SINGULARITY_LIMIT = 0.1

# Each shoulder can be half a tick off. Position error is largest at a corner of that cube.
_ROUNDING_CORNERS = 0.5 * np.array(list(product([-1.0, 1.0], repeat=3)))


class Cause(Enum):
    """Why the end-effector cannot reach a position."""

    ARM_LENGTH = "arm length"
    JOINT_LIMITS = "joint limits"
    UPPER_SOLUTION = "upper forward solution"


def unreachable_cause(geom: DeltaGeometry, position: ArrayLike) -> Cause | None:
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


def reachable(geom: DeltaGeometry, positions: ArrayLike) -> np.ndarray:
    """Which positions (m, shape (..., 3)) the end-effector can reach (bool, shape (...))."""
    positions = np.asarray(positions, dtype=float)
    flat = [unreachable_cause(geom, p) is None for p in positions.reshape(-1, 3)]
    return np.array(flat, dtype=bool).reshape(positions.shape[:-1])


@dataclass(frozen=True)
class TargetCylinder:
    """Volume a geometry must cover: upright and centred on the z axis, SI units.

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
            for band in _z_bands(np.array(hits).reshape(-1, 3)):
                x, y, z = band.T
                radius = np.hypot(x, y)
                lines.append(
                    f"  {cause.value}: {len(band)} positions, "
                    f"z {z.min():.3f} to {z.max():.3f} m, "
                    f"radius {radius.min():.3f} to {radius.max():.3f} m"
                )
        return "\n".join(lines)


DEFAULT_TARGET = TargetCylinder()


def _z_bands(positions: np.ndarray) -> list[np.ndarray]:
    """Split positions (shape (N, 3)) into runs of neighbouring layers, lowest first."""
    if len(positions) == 0:
        return []
    ordered = positions[np.argsort(positions[:, 2])]
    gaps = np.flatnonzero(np.diff(ordered[:, 2]) > 1.5 * _LAYER_SPACING)
    return np.split(ordered, gaps + 1)


def _disc(radius: float) -> np.ndarray:
    """Sample (x, y) offsets across a disc (m, shape (N, 2)), centre included."""
    offsets = [(0.0, 0.0)]
    for r in np.linspace(0.0, radius, ceil(radius / _RING_SPACING) + 1)[1:]:
        offsets.extend(zip(r * np.cos(_RING_ANGLES), r * np.sin(_RING_ANGLES)))
    return np.array(offsets)


def _disc_representatives(samples: int) -> np.ndarray:
    """For each of _disc's samples, the index of the sample that stands in for it.

    Arms sit 120 deg apart with shared limits and each mirrors the workspace, so an
    angle on a ring reaches what its image between 0 and 60 deg reaches, up to
    rounding: within a few 1e-16 rad of a joint limit the two can differ.
    """
    period = len(_RING_ANGLES) // 3
    steps = np.arange(len(_RING_ANGLES)) % period
    wedge = np.minimum(steps, period - steps)
    rings = (samples - 1) // len(_RING_ANGLES)
    return np.concatenate(
        [[0], *(1 + ring * len(_RING_ANGLES) + wedge for ring in range(rings))]
    )


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
    representatives = _disc_representatives(len(disc))

    causes = []
    for z in layers:
        found = {i: unreachable_cause(geom, [*disc[i], z]) for i in np.unique(representatives)}
        causes.append([found[i] for i in representatives])
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


@dataclass(frozen=True)
class Load:
    """What the shoulders have to carry, SI units.

    payload:           mass carried by the end-effector (kg)
    end_effector_mass: mass of the end-effector itself (kg)
    upper_arm_mass:    mass of one upper arm, taken as uniform along its length (kg)
    acceleration:      size of the end-effector acceleration, in any direction (m/s^2)

    The lower arms are not modelled. Add their share to end_effector_mass.
    """

    payload: float
    end_effector_mass: float
    upper_arm_mass: float
    acceleration: float

    def __post_init__(self) -> None:
        for value in (
            self.payload,
            self.end_effector_mass,
            self.upper_arm_mass,
            self.acceleration,
        ):
            if not 0.0 <= value < float("inf"):  # False for NaN
                raise ValueError(f"load values must be finite and not negative: {self}")


@dataclass(frozen=True)
class ShoulderTorque:
    """Torque size at each shoulder (N m, shape (3,)), indexed like the arms.

    static:       holding still against gravity
    accelerating: with load.acceleration in the worst direction for that shoulder,
                  starting from rest. Each shoulder has its own worst direction.
    """

    static: np.ndarray
    accelerating: np.ndarray


def _jacobians(
    geom: DeltaGeometry, position: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Theta, the jacobian and its inverse at position (m). Raises UnreachableError."""
    position = np.asarray(position, dtype=float)
    cause = unreachable_cause(geom, position)
    if cause is not None:
        raise UnreachableError(f"{position} is out of reach: {cause.value}")
    theta = inverse(geom, position)
    try:
        jac = jacobian(geom, theta)
        return theta, jac, np.linalg.inv(jac)
    except np.linalg.LinAlgError:
        raise UnreachableError(f"{position} is a singular position") from None


def shoulder_torque(geom: DeltaGeometry, position: ArrayLike, load: Load) -> ShoulderTorque:
    """Torque the shoulders need with the end-effector at position (m).

    Raises UnreachableError if the end-effector cannot reach position.
    """
    theta, jac, jac_inverse = _jacobians(geom, position)

    # Virtual work: the shoulders lift and accelerate the end-effector and upper arms.
    # Torques that depend on speed are left out, so this holds from rest.
    mass = load.payload + load.end_effector_mass
    upper_arm_weight = load.upper_arm_mass * GRAVITY * geom.upper_arm / 2.0 * np.cos(theta)
    static = np.abs(mass * GRAVITY * jac[2] - upper_arm_weight)
    inertia = load.upper_arm_mass * geom.upper_arm**2 / 3.0  # uniform, about the shoulder
    per_unit_acceleration = np.linalg.norm(mass * jac.T + inertia * jac_inverse, axis=1)
    return ShoulderTorque(static, static + load.acceleration * per_unit_acceleration)


def tick_error(
    geom: DeltaGeometry, position: ArrayLike, joint_resolution: float = RAD_PER_TICK
) -> float:
    """Largest position error (m) at position from rounding each shoulder to a tick.

    joint_resolution is the size of one tick (rad), and each shoulder can be half of one
    off. Raises UnreachableError if the end-effector cannot reach position.
    """
    _, jac, _ = _jacobians(geom, position)
    return float(np.linalg.norm(jac @ (joint_resolution * _ROUNDING_CORNERS).T, axis=0).max())


def _format_position(position: tuple[float, float, float]) -> str:
    x, y, z = position
    return f"({x:.3f}, {y:.3f}, {z:.3f}) m"


@dataclass(frozen=True)
class TorqueMargin:
    """A required torque against a servo's continuous torque (N m)."""

    required: float
    continuous: float

    @property
    def spare(self) -> float:
        """Torque left over (N m), negative when the servo falls short."""
        return self.continuous - self.required

    @property
    def ratio(self) -> float:
        """Continuous torque as a multiple of the required torque."""
        return self.continuous / self.required if self.required > 0.0 else float("inf")

    @property
    def passed(self) -> bool:
        return self.spare >= 0.0

    def __str__(self) -> str:
        verdict = "OK" if self.passed else "SHORT"
        return (
            f"{verdict}: {self.required:.3f} N m required, {self.continuous:.3f} N m "
            f"continuous, {abs(self.spare):.3f} N m {'spare' if self.passed else 'short'} "
            f"(x{self.ratio:.2f})"
        )


@dataclass(frozen=True)
class WorstTorque:
    """The largest torque found, on which shoulder, with the end-effector where.

    The workspace is symmetric, so the same torque also occurs on the other shoulders at
    the rotated and mirrored positions.
    """

    torque: float
    position: tuple[float, float, float]
    shoulder: int

    def margin(self, continuous_torque: float) -> TorqueMargin:
        """Compare with a servo's continuous torque (N m)."""
        if not continuous_torque > 0.0:  # False for NaN
            raise ValueError(f"continuous torque must be positive: {continuous_torque}")
        return TorqueMargin(self.torque, continuous_torque)

    def __str__(self) -> str:
        where = _format_position(self.position)
        return f"{self.torque:.3f} N m on shoulder {self.shoulder}, end-effector at {where}"


@dataclass(frozen=True)
class TorqueReport:
    """Worst cases over the sampled workspace.

    static, accelerating: worst shoulder torques, each with its own position
    tick_error:           worst position error from rounding to ticks (m)
    tick_error_position:  where that occurs (m)
    joint_resolution:     the tick size used (rad)
    samples:              number of reachable positions the worst cases come from
    near_singular:        number of reachable positions left out for being within
                          SINGULARITY_LIMIT of a singularity. If not 0, the workspace
                          holds positions the robot should not be sent to.
    """

    load: Load
    static: WorstTorque
    accelerating: WorstTorque
    tick_error: float
    tick_error_position: tuple[float, float, float]
    joint_resolution: float
    samples: int
    near_singular: int

    def __str__(self) -> str:
        error = f"{self.tick_error:.6f} m at {self.joint_resolution:.5f} rad per tick"
        where = _format_position(self.tick_error_position)
        lines = [
            f"Over {self.samples} sampled positions:",
            f"  static: {self.static}",
            f"  at {self.load.acceleration:.1f} m/s^2: {self.accelerating}",
            f"  tick error: {error}, end-effector at {where}",
        ]
        if self.near_singular:
            lines.append(f"  left out {self.near_singular} positions near a singularity")
        return "\n".join(lines)


def _sweep_positions(geom: DeltaGeometry) -> np.ndarray:
    """Reachable positions (m, shape (N, 3)) sampled across the wedge of the workspace.

    Rings and layers are _SWEEP_SPACING apart. Around a ring only _SWEEP_ANGLES are taken.
    """
    depth = geom.upper_arm + geom.lower_arm
    reach = geom.base_radius - geom.end_effector_radius + depth
    radii = _SWEEP_SPACING * np.arange(1, ceil(reach / _SWEEP_SPACING) + 1)
    offsets = [(0.0, 0.0)]
    for r in radii:
        offsets.extend(zip(r * np.cos(_SWEEP_ANGLES), r * np.sin(_SWEEP_ANGLES)))
    top = ceil(depth / _SWEEP_SPACING)
    layers = _SWEEP_SPACING * np.arange(-top, top + 1)
    positions = np.array([(x, y, z) for z in layers for x, y in offsets])
    return positions[reachable(geom, positions)]


def _as_tuple(position: np.ndarray) -> tuple[float, float, float]:
    x, y, z = position
    return float(x), float(y), float(z)


def _worst(torques: np.ndarray, positions: np.ndarray) -> WorstTorque:
    """Largest entry of torques (shape (N, 3)), with positions (shape (N, 3)) alongside."""
    sample, shoulder = np.unravel_index(np.argmax(torques), torques.shape)
    return WorstTorque(float(torques[sample, shoulder]), _as_tuple(positions[sample]), int(shoulder))


def torque_report(
    geom: DeltaGeometry, load: Load, joint_resolution: float = RAD_PER_TICK
) -> TorqueReport:
    """Find the worst shoulder torque and tick error across the workspace of geom.

    Positions near a singularity are left out and counted. Raises ValueError if the sweep
    finds nothing else. A worst case that falls between samples is missed, so treat the
    numbers as good to a few percent.
    """
    reachable_positions = _sweep_positions(geom)
    margins = [singularity_margin(geom, inverse(geom, p)) for p in reachable_positions]
    positions = reachable_positions[np.array(margins) >= SINGULARITY_LIMIT]
    if len(positions) == 0:
        raise ValueError(f"the sweep found no usable workspace for {geom}")
    torques = [shoulder_torque(geom, p, load) for p in positions]
    errors = np.array([tick_error(geom, p, joint_resolution) for p in positions])
    return TorqueReport(
        load=load,
        static=_worst(np.array([t.static for t in torques]), positions),
        accelerating=_worst(np.array([t.accelerating for t in torques]), positions),
        tick_error=float(errors.max()),
        tick_error_position=_as_tuple(positions[np.argmax(errors)]),
        joint_resolution=joint_resolution,
        samples=len(positions),
        near_singular=len(reachable_positions) - len(positions),
    )
