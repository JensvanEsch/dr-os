# dr-os - Delta Robot Open Source

Cheap 3D-printed rotary (Clavel-type) delta robot driven by three Feetech STS3215
serial-bus servos (the same servos as the SO-101 arm). The goal is a reusable
software ecosystem for parallel manipulators, developed in the open.
Small payload, no hard requirements: cheap and reproducible beats fast.

## Conventions (do not change without an ADR)
- **Vocabulary:** the project's canonical terms are fixed in `GLOSSARY.md`
  (binding, including its _Avoid_ lists); the lock is recorded in
  `docs/adr/0002-canonical-vocabulary.md`. Use those words in code,
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
- `src/dros/kinematics.py` - inverse / forward kinematics, jacobian, singularity margin
- `src/dros/analysis.py`   - reachability, the target cylinder check, worst-case torque, tick error,
  workspace map data
- `src/dros/motion.py`     - `set_end_effector_position` (m) on top of `set_shoulder_angles` (rad):
  one joint target each, waits until the shoulders settle
- `src/dros/viz.py`        - workspace plot (optional `viz` extra, matplotlib)
- `src/dros/servo/`        - JointBus protocol and its safety checks, MockBus, SimBus, STS3215 tick conversion
- `src/dros/sim/`          - digital twin (not started)
- `tests/`                     - pytest; kinematics round-trip tests live here
- `hw/`                        - scripts that touch real hardware (see safety rules)
- `demo/`                      - scripts that touch no hardware, e.g. the workspace plot
- `docs/adr/`                  - short ADRs (context, decision, consequences)
- `docs/devlog/`               - dated development notes, source for posts/videos
- `docs/hardware.md`           - BOM, measured dimensions, servo IDs, wiring

## Commands
- Install: `pip install -e ".[dev,viz]"` (without `viz` the plot tests skip)
- Test: `pytest`
- Lint: `ruff check .`

## Working rules
- Any change to kinematics or geometry needs a round-trip test
  (FK(IK(p)) == p and IK(FK(theta)) == theta within tolerance).
  After such a change, use the `kinematics-reviewer` subagent.
- **Hardware safety:** never run anything in `hw/`, or anything that opens a
  serial port, without asking me first and stating what will move and how far.
  Develop against `MockBus` by default, or `SimBus` when motion over time matters.
- Joint limits are defined in one place (`DeltaGeometry.check_joint_limits`).
  Every joint bus enforces them itself in `write_angles`, together with the
  torque-off refusal (ADR 0005, 0006). New buses must pass
  the shared contract suite in `tests/joint_bus_contract.py`.
- Architectural decisions go in `docs/adr/NNNN-title.md`.
- Keep dependencies minimal; heavy things (sim, viz, hardware SDK) go behind
  optional extras in pyproject.toml.
- Small commits with human-readable messages - I turn them into devlog posts.
- When unsure about a physical dimension, ask; do not guess numbers.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (JensvanEsch/dr-os) via the gh CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default role names are used unchanged. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` at the root, ADRs in `docs/adr/`. See `docs/agents/domain.md`.

## Status
- [ ] Measure the printed robot and replace the placeholder dimensions in DeltaGeometry
- [ ] Servo bring-up: assign IDs 1-3, find zero offsets and directions
- [x] Pick a licence for the open-source release
- [x] Workspace plot / reachability map
- [ ] Digital twin in `sim/`
