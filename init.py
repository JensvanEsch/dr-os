#!/usr/bin/env python3
"""
init_delta_project.py - scaffold the open-source delta robot repo for Claude Code.

Creates:
  CLAUDE.md                      project memory Claude Code reads every session
  .claude/settings.json          shared permissions (safe defaults, hardware always asks)
  .claude/commands/*.md          slash commands: /kin-check /feature /devlog /hw-bringup
  .claude/agents/*.md            kinematics-reviewer subagent
  src/<pkg>/                     geometry, IK/FK, servo bus abstraction (mock + STS3215 tick maths)
  tests/                         round-trip kinematics tests
  docs/                          decisions (ADRs), hardware notes, devlog
  hw/                            home for scripts that move real servos

Usage:
  python init_delta_project.py [target_dir] [--package deltakit] [--no-git] [--force]

Existing files are never overwritten unless --force is given.
Only the Python standard library is needed to run this script.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

PKG = "deltakit"  # replaced by --package

FILES: dict[str, str] = {}


def f(path: str, body: str) -> None:
    FILES[path] = textwrap.dedent(body).lstrip("\n")


# --------------------------------------------------------------------------- #
# CLAUDE.md - the most important file: Claude Code loads it every session
# --------------------------------------------------------------------------- #
f("CLAUDE.md", r'''
# deltakit - open-source delta robot

Cheap 3D-printed rotary (Clavel-type) delta robot driven by three Feetech STS3215
serial-bus servos (the same servos as the SO-101 arm). The goal is a reusable
software ecosystem for parallel manipulators, developed in the open.
Small payload, no hard requirements: cheap and reproducible beats fast.

## Conventions (do not change without an ADR)
- **Units:** SI everywhere in code (m, rad, s). Convert to servo ticks only inside
  `src/deltakit/servo/`.
- **Base frame:** origin at the centre of the shoulder plane, z up. The effector
  works at negative z. Arm i sits at angle 120 deg * i around z; arm 0 points along +x.
- **Joint angle:** theta = 0 means the upper arm is horizontal; positive theta
  rotates the arm downward. The IK picks the elbow-out branch.
- **Geometry lives only in `DeltaGeometry`** (`src/deltakit/geometry.py`).
  Never hard-code dimensions anywhere else.
- Python >= 3.10, numpy, type hints, dataclasses. Tests with pytest. Lint with ruff.

## Layout
- `src/deltakit/geometry.py`   - DeltaGeometry: dimensions + joint limits
- `src/deltakit/kinematics.py` - inverse / forward kinematics
- `src/deltakit/servo/`        - JointBus protocol, MockBus, STS3215 tick conversion
- `src/deltakit/sim/`          - digital twin (not started)
- `tests/`                     - pytest; kinematics round-trip tests live here
- `hw/`                        - scripts that touch real hardware (see safety rules)
- `docs/decisions/`            - short ADRs (context, decision, consequences)
- `docs/devlog/`               - dated development notes, source for posts/videos
- `docs/hardware.md`           - BOM, measured dimensions, servo IDs, wiring

## Commands
- Install: `pip install -e ".[dev]"`
- Test: `pytest`
- Lint: `ruff check .`

## Working rules
- Any change to kinematics or geometry needs a round-trip test
  (FK(IK(p)) == p and IK(FK(theta)) == theta within tolerance).
  After such a change, use the `kinematics-reviewer` subagent.
- **Hardware safety:** never run anything in `hw/`, or anything that opens a
  serial port, without asking me first and stating what will move and how far.
  Develop against `MockBus` by default.
- Joint limits are enforced in one place (`DeltaGeometry.check_joint_limits`)
  and checked before any command reaches a bus.
- Architectural decisions go in `docs/decisions/NNNN-title.md`.
- Keep dependencies minimal; heavy things (sim, viz, hardware SDK) go behind
  optional extras in pyproject.toml.
- Small commits with human-readable messages - I turn them into devlog posts.
- When unsure about a physical dimension, ask; do not guess numbers.

## Status
- [ ] Measure the printed robot and replace the placeholder dimensions in DeltaGeometry
- [ ] Servo bring-up: assign IDs 1-3, find zero offsets and directions
- [ ] Pick a licence for the open-source release
- [ ] Workspace plot / reachability map
- [ ] Digital twin in `sim/`
''')

# --------------------------------------------------------------------------- #
# Claude Code configuration
# --------------------------------------------------------------------------- #
f(".claude/settings.json", r'''
{
  "permissions": {
    "allow": [
      "Bash(pytest:*)",
      "Bash(python -m pytest:*)",
      "Bash(ruff check:*)",
      "Bash(git status)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(pip install -e:*)"
    ],
    "deny": [
      "Read(./.env)",
      "Read(./.env.*)"
    ]
  }
}
''')

f(".claude/commands/kin-check.md", r'''
---
description: Run the kinematics tests and a dense IK/FK round-trip sweep, report the worst error
---
1. Run `pytest tests/test_kinematics.py -q`.
2. Write and run a short throwaway script (do not commit it) that samples a dense
   grid of joint angles inside the joint limits, runs FK then IK, and reports:
   max position error (m), max joint error (rad), and how many samples were
   unreachable or hit a branch flip.
3. Summarise in at most 5 lines. If anything exceeds 1e-9 m or 1e-9 rad,
   investigate before proposing a fix.
''')

f(".claude/commands/feature.md", r'''
---
description: Plan and implement a feature the deltakit way (plan, ADR if needed, tests, code)
argument-hint: <what to build>
---
Feature request: $ARGUMENTS

1. Read CLAUDE.md and the relevant modules. Propose a short plan (max 8 bullets)
   and wait for my OK.
2. If the plan changes a convention, frame, unit, or public interface, draft an
   ADR in `docs/decisions/` first.
3. Write tests first, then the implementation. Use MockBus, never real hardware.
4. Run `pytest` and `ruff check .` and fix what fails.
5. Finish with a one-paragraph summary I could paste into the devlog.
''')

f(".claude/commands/devlog.md", r'''
---
description: Turn recent commits into a dated devlog entry for posts and videos
argument-hint: [since, e.g. "1 week ago"]
---
1. Run `git log --since="${ARGUMENTS:-1 week ago}" --stat`.
2. Write `docs/devlog/YYYY-MM-DD.md` (today's date) with:
   - **What changed** - 3-6 bullets in plain language, no jargon dumps
   - **What I learned** - 1-3 bullets
   - **Next** - 1-3 bullets
   - **Post idea** - one 2-3 sentence caption a non-roboticist would enjoy
3. Do not commit; show me the file path.
''')

f(".claude/commands/hw-bringup.md", r'''
---
description: Guided servo bring-up checklist (asks before anything moves)
---
Walk me through bringing up the three STS3215 servos, one step at a time,
waiting for my confirmation between steps:

1. Check `docs/hardware.md` for the adapter, port, baud rate and servo IDs.
   Ask me for anything missing and update the file.
2. Propose a script for `hw/` that only *reads* (ping IDs 1-3, read position,
   voltage, temperature). Show it, then ask before running it.
3. For zero offsets and directions: propose a procedure, ask before any
   motion, keep torque low and moves small, and record results in
   `docs/hardware.md` and in the servo config, never as magic numbers in code.
Never write positions to a servo without an explicit "go" from me in this session.
''')

f(".claude/agents/kinematics-reviewer.md", r'''
---
name: kinematics-reviewer
description: Reviews changes to kinematics, geometry, frames or units in deltakit. Use proactively after edits to kinematics.py, geometry.py or anything that converts angles or positions.
tools: Read, Grep, Glob, Bash
---
You are a careful reviewer of parallel-manipulator kinematics.

For the change under review:
1. Check it against the conventions in CLAUDE.md (SI units, frame, theta sign,
   arm placement at 120 deg * i, elbow-out branch).
2. Look for: degree/radian mix-ups, mm/m mix-ups, wrong rotation direction,
   branch selection changes, singularities near the workspace boundary,
   silent NaNs from acos/sqrt of out-of-range values.
3. Run `pytest -q`. If useful, write a throwaway numerical check (do not commit).
4. Report: verdict (OK / needs changes), concrete issues with file:line,
   and a suggested test for each issue. Do not edit source files yourself.
''')

# --------------------------------------------------------------------------- #
# Python package
# --------------------------------------------------------------------------- #
f("pyproject.toml", r'''
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "deltakit"
version = "0.0.1"
description = "Software ecosystem for a cheap open-source delta robot (STS3215 servos)"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest>=7", "ruff>=0.4"]
hw = ["feetech-servo-sdk"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
''')

f("src/deltakit/__init__.py", r'''
"""deltakit - kinematics and control for a cheap open-source delta robot."""
from .geometry import DeltaGeometry
from .kinematics import UnreachableError, forward, inverse

__all__ = ["DeltaGeometry", "UnreachableError", "forward", "inverse"]
''')

f("src/deltakit/geometry.py", r'''
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
''')

f("src/deltakit/kinematics.py", r'''
"""Inverse and forward kinematics for a rotary (Clavel-type) delta robot.

Frame and sign conventions are documented in CLAUDE.md.
"""
from __future__ import annotations

import numpy as np

from .geometry import DeltaGeometry


class UnreachableError(ValueError):
    """The requested pose or joint configuration has no valid solution."""


def _elbows_shifted(geom: DeltaGeometry, theta: np.ndarray) -> np.ndarray:
    """Elbow positions shifted inward by effector_radius (3x3, one row per arm).

    The effector centre lies at distance lower_arm from every one of these points.
    """
    rho = geom.base_radius - geom.effector_radius + geom.upper_arm * np.cos(theta)
    phi = geom.arm_angles
    return np.column_stack(
        [rho * np.cos(phi), rho * np.sin(phi), -geom.upper_arm * np.sin(theta)]
    )


def inverse(geom: DeltaGeometry, p) -> np.ndarray:
    """Joint angles (rad, shape (3,)) that place the effector centre at p (m).

    Picks the elbow-out branch. Raises UnreachableError if p is out of reach.
    Joint limits are NOT checked here; use geom.check_joint_limits.
    """
    x, y, z = np.asarray(p, dtype=float)
    L, l = geom.upper_arm, geom.lower_arm
    offset = geom.base_radius - geom.effector_radius
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
        wx, wz = xr + geom.effector_radius - geom.base_radius, z
        side = [wx * (-L * np.sin(c)) - wz * (L * np.cos(c)) for c in candidates]
        t = candidates[0] if side[0] >= side[1] else candidates[1]
        theta[i] = (t + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-pi, pi]

    return theta


def forward(geom: DeltaGeometry, theta) -> np.ndarray:
    """Effector centre position (m, shape (3,)) for joint angles theta (rad).

    Intersection of three spheres of radius lower_arm; the lower solution
    (effector below the base) is returned.
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
''')

f("src/deltakit/servo/__init__.py", r'''
from .bus import JointBus, MockBus, safe_write
from .sts3215 import CENTER_TICK, TICKS_PER_REV, rad_to_ticks, ticks_to_rad

__all__ = [
    "CENTER_TICK",
    "TICKS_PER_REV",
    "JointBus",
    "MockBus",
    "rad_to_ticks",
    "safe_write",
    "ticks_to_rad",
]
''')

f("src/deltakit/servo/bus.py", r'''
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
''')

f("src/deltakit/servo/sts3215.py", r'''
"""STS3215 position units. 12-bit encoder: 4096 ticks per revolution.

Only unit conversion lives here for now. The serial driver (feetech-servo-sdk,
optional extra "hw") will be added during bring-up; see .claude/commands/hw-bringup.md.
"""
from __future__ import annotations

import numpy as np

TICKS_PER_REV = 4096
CENTER_TICK = 2048


def rad_to_ticks(theta, zero_tick: int = CENTER_TICK, direction: int = 1) -> np.ndarray:
    """Joint angle (rad) -> raw position ticks, per-servo zero and direction."""
    ticks = zero_tick + direction * np.asarray(theta, dtype=float) * TICKS_PER_REV / (2 * np.pi)
    return np.rint(ticks).astype(int)


def ticks_to_rad(ticks, zero_tick: int = CENTER_TICK, direction: int = 1) -> np.ndarray:
    """Raw position ticks -> joint angle (rad)."""
    return direction * (np.asarray(ticks, dtype=float) - zero_tick) * 2 * np.pi / TICKS_PER_REV
''')

f("src/deltakit/sim/__init__.py", r'''
"""Digital twin - not started. Ideas go in docs/decisions before code goes here."""
''')

f("tests/test_kinematics.py", r'''
import numpy as np
import pytest

from deltakit import DeltaGeometry, UnreachableError, forward, inverse
from deltakit.servo import MockBus, rad_to_ticks, safe_write, ticks_to_rad

GEOM = DeltaGeometry()
RNG = np.random.default_rng(0)


def test_home_is_centred_below_base():
    p = forward(GEOM, np.zeros(3))
    assert abs(p[0]) < 1e-12 and abs(p[1]) < 1e-12
    assert p[2] < 0


def test_ik_fk_round_trip_joint_space():
    for _ in range(2000):
        theta = RNG.uniform(-0.3, 1.0, size=3)
        p = forward(GEOM, theta)
        assert np.allclose(inverse(GEOM, p), theta, atol=1e-9)


def test_fk_ik_round_trip_cartesian():
    z0 = forward(GEOM, np.full(3, 0.4))[2]
    for _ in range(2000):
        p = np.array([*RNG.uniform(-0.04, 0.04, size=2), z0 + RNG.uniform(-0.03, 0.03)])
        try:
            theta = inverse(GEOM, p)
        except UnreachableError:
            continue
        assert np.allclose(forward(GEOM, theta), p, atol=1e-9)


def test_symmetry_on_z_axis():
    theta = inverse(GEOM, [0.0, 0.0, -0.2])
    assert np.allclose(theta, theta[0])


def test_unreachable_raises():
    with pytest.raises(UnreachableError):
        inverse(GEOM, [0.0, 0.0, -1.0])


def test_safe_write_enforces_limits():
    bus = MockBus()
    safe_write(bus, GEOM, [0.1, 0.2, 0.3])
    assert np.allclose(bus.read_angles(), [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        safe_write(bus, GEOM, [0.0, 0.0, 3.0])


def test_tick_conversion_round_trip():
    theta = np.array([-0.5, 0.0, 1.2])
    back = ticks_to_rad(rad_to_ticks(theta, zero_tick=2000, direction=-1),
                        zero_tick=2000, direction=-1)
    assert np.allclose(back, theta, atol=2 * np.pi / 4096)
''')

# --------------------------------------------------------------------------- #
# Docs and misc
# --------------------------------------------------------------------------- #
f("README.md", r'''
# deltakit

An open-source, as-cheap-as-possible delta robot: 3D-printed parts and three
Feetech STS3215 serial-bus servos - plus a growing software ecosystem for
parallel manipulators.

```bash
pip install -e ".[dev]"
pytest
```

Status: early development. See `docs/devlog/` for progress.
''')

f("docs/hardware.md", r'''
# Hardware

## Bill of materials
| Part | Qty | Source | Price | Notes |
|------|-----|--------|-------|-------|
| Feetech STS3215 servo | 3 | | | same as SO-101 |
| Bus servo adapter board | 1 | | | |
| Power supply | 1 | | | voltage per servo variant |
| Lower-arm rods | 6 | | | cheap + precise: still deciding |
| Ball joints / magnetic joints | 12 | | | |

## Measured geometry (fill in, then update DeltaGeometry)
| Parameter | Value (m) | How measured |
|-----------|-----------|--------------|
| base_radius | | |
| effector_radius | | |
| upper_arm | | |
| lower_arm | | |

## Servo bus
| Arm | Servo ID | zero_tick | direction | Notes |
|-----|----------|-----------|-----------|-------|
| 0 (+x) | 1 | | | |
| 1 (120 deg) | 2 | | | |
| 2 (240 deg) | 3 | | | |

Port: ?   Baud rate: ?
''')

f("docs/decisions/0001-sts3215-servos.md", r'''
# 0001 - STS3215 serial-bus servos instead of steppers

## Context
The earlier delta robot used NEMA23 steppers with ~40 cm arms: too large and too
expensive. The new goal is the cheapest possible open-source delta robot.

## Decision
Drive the three arms with Feetech STS3215 serial-bus servos (as on the SO-101),
everything else 3D printed. Small payload is acceptable.

## Consequences
- Cheap, compact, daisy-chained bus, built-in encoder feedback.
- Limited torque and resolution (4096 ticks/rev) -> small, light robot.
- Software is the main product: it should stay hardware-agnostic (JointBus).
''')

f("docs/devlog/.gitkeep", "")

f("hw/README.md", r'''
# hw/

Scripts in this folder talk to real servos. Claude Code must ask before running
any of them and say what will move. Prefer read-only scripts first.
''')

f(".gitignore", r'''
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.pytest_cache/
.ruff_cache/
.env
.env.*
.claude/settings.local.json
''')


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Scaffold the delta robot repo for Claude Code.")
    ap.add_argument("target", nargs="?", default="delta-robot", help="target directory")
    ap.add_argument("--package", default=PKG, help="python package name (default: deltakit)")
    ap.add_argument("--no-git", action="store_true", help="skip git init")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    if not args.package.isidentifier():
        print(f"--package must be a valid python identifier, got {args.package!r}")
        return 1

    root = Path(args.target).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []

    for rel, body in FILES.items():
        rel = rel.replace(f"src/{PKG}/", f"src/{args.package}/")
        body = body.replace(PKG, args.package)
        path = root / rel
        if path.exists() and not args.force:
            skipped.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        written.append(rel)

    if not args.no_git and shutil.which("git") and not (root / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=root, check=False)
        print("git: initialised repository")

    print(f"\nScaffolded {root}")
    print(f"  wrote {len(written)} files" + (f", skipped {len(skipped)} existing:" if skipped else ""))
    for s in skipped:
        print(f"    - {s}")
    print(textwrap.dedent(f"""
        Next steps:
          cd {root}
          python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
          pip install -e ".[dev]"
          pytest
          claude            # start Claude Code; it reads CLAUDE.md automatically
        Then try: /kin-check   /feature <idea>   /devlog   /hw-bringup
    """))
    return 0


if __name__ == "__main__":
    sys.exit(main())
