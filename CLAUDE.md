# dr-os - Delta Robot Open Source

Cheap 3D-printed rotary (Clavel-type) delta robot driven by three Feetech STS3215
serial-bus servos (the same servos as the SO-101 arm). The goal is a reusable
software ecosystem for parallel manipulators, developed in the open.
Small payload, no hard requirements: cheap and reproducible beats fast.

## Conventions (do not change without an ADR)
- **Vocabulary:** the project's canonical terms are fixed in `GLOSSARY.md`
  (binding, including its _Avoid_ lists); the lock is recorded in
  `docs/decisions/0002-canonical-vocabulary.md`. Use those words in code,
  comments, and commits (e.g. "arm" is a whole chain, never a single link).
- **Units:** SI everywhere in code (m, rad, s). Convert to servo ticks only inside
  `src/dros/servo/`.
- **Base frame:** origin at the centre of the shoulder plane, z up. The end-effector
  works at negative z. Arm i sits at angle 120 deg * i around z; arm 0 points along +x.
- **Joint angle:** theta = 0 means the upper arm is horizontal; positive theta
  rotates the upper arm downward. The IK picks the elbow-out branch.
- **Geometry lives only in `DeltaGeometry`** (`src/dros/geometry.py`).
  Never hard-code dimensions anywhere else.
- Python >= 3.10, numpy, type hints, dataclasses. Tests with pytest. Lint with ruff.

## Layout
- `src/dros/geometry.py`   - DeltaGeometry: dimensions + joint limits
- `src/dros/kinematics.py` - inverse / forward kinematics
- `src/dros/servo/`        - JointBus protocol, MockBus, STS3215 tick conversion
- `src/dros/sim/`          - digital twin (not started)
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
