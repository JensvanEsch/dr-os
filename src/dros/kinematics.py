"""Inverse and forward kinematics for a rotary (Clavel-type) delta robot.

Frame and sign conventions are documented in CLAUDE.md.
"""
from __future__ import annotations

import numpy as np

from .geometry import DeltaGeometry


class UnreachableError(ValueError):
    """The requested position or joint configuration has no valid solution."""


def _elbows_shifted(geom: DeltaGeometry, theta: np.ndarray) -> np.ndarray:
    """Elbow positions shifted inward by end_effector_radius (3x3, one row per arm).

    The end-effector centre lies at distance lower_arm from every one of these points.
    """
    rho = geom.base_radius - geom.end_effector_radius + geom.upper_arm * np.cos(theta)
    phi = geom.arm_angles
    return np.column_stack(
        [rho * np.cos(phi), rho * np.sin(phi), -geom.upper_arm * np.sin(theta)]
    )


def inverse(geom: DeltaGeometry, p) -> np.ndarray:
    """Joint angles (rad, shape (3,)) that place the end-effector centre at p (m).

    Picks the elbow-out branch. Raises UnreachableError if p is out of reach.
    Joint limits are NOT checked here; use geom.check_joint_limits.
    """
    x, y, z = np.asarray(p, dtype=float)
    L, l = geom.upper_arm, geom.lower_arm
    offset = geom.base_radius - geom.end_effector_radius
    theta = np.empty(3)

    for i, phi in enumerate(geom.arm_angles):
        # rotate the target into the plane of arm i
        c, s = np.cos(phi), np.sin(phi)
        xr = c * x + s * y
        yr = -s * x + c * y

        # |elbow - wrist|^2 = l^2  ->  A cos(t) + B sin(t) = C
        u = offset - xr
        A = 2.0 * L * u
        B = 2.0 * L * z
        C = l**2 - L**2 - u**2 - yr**2 - z**2
        R = np.hypot(A, B)
        if R < 1e-12 or abs(C) > R:
            raise UnreachableError(f"arm {i} cannot reach {p}")

        base = np.arctan2(B, A)
        d = np.arccos(np.clip(C / R, -1.0, 1.0))
        candidates = (base + d, base - d)

        # elbow-out: elbow lies on the outer side of the shoulder->wrist line
        wx, wz = xr + geom.end_effector_radius - geom.base_radius, z
        side = [wx * (-L * np.sin(c)) - wz * (L * np.cos(c)) for c in candidates]
        t = candidates[0] if side[0] >= side[1] else candidates[1]
        theta[i] = (t + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-pi, pi]

    return theta


def forward(geom: DeltaGeometry, theta) -> np.ndarray:
    """End-effector centre position (m, shape (3,)) for joint angles theta (rad).

    Intersection of three spheres of radius lower_arm; the lower solution
    (end-effector below the base) is returned.
    """
    theta = np.asarray(theta, dtype=float)
    p1, p2, p3 = _elbows_shifted(geom, theta)
    r = geom.lower_arm

    d = np.linalg.norm(p2 - p1)
    ex = (p2 - p1) / d
    i = ex @ (p3 - p1)
    ey_raw = p3 - p1 - i * ex
    j = np.linalg.norm(ey_raw)
    if d < 1e-12 or j < 1e-12:
        raise UnreachableError(f"degenerate configuration {theta}")
    ey = ey_raw / j
    ez = np.cross(ex, ey)

    x = d / 2.0
    y = (i**2 + j**2) / (2.0 * j) - (i / j) * x
    z2 = r**2 - x**2 - y**2
    if z2 < 0.0:
        raise UnreachableError(f"lower arms cannot close for {theta}")
    zc = np.sqrt(z2)

    a = p1 + x * ex + y * ey + zc * ez
    b = p1 + x * ex + y * ey - zc * ez
    return a if a[2] < b[2] else b
