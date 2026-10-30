# 0002 - Canonical vocabulary for dr-os

## Context
Before the project grows, the domain vocabulary is being set in stone so code,
docs, and commit messages cannot drift. Two things needed a deliberate call:
whether "arm" (which the scaffold used for both a whole chain and its links) is
ambiguous, and what to call the moving platform. The scaffold also shipped under
the working name `deltakit`.

## Decision
- A kinematic chain is an **arm** (three of them, indexed 0-2). Each arm is made
  of an **upper arm** (shoulder->elbow) and a **lower arm** (elbow->wrist). This
  is a part-whole hierarchy, not an overload: "arm" is the whole chain and is
  never used for a single link, so "limb" is unnecessary.
- The lower arm is a **parallelogram** of two rods; that is what keeps the
  end-effector level and gives the robot its 3-DOF, translation-only motion.
- The moving platform is the **end-effector**; it has a **position** (3-DOF
  translation, never a "pose"). A tool mounted on it later is a separate term,
  **tool / TCP**, to be introduced by its own ADR.
- Joints keep the biological names **shoulder** (actuated), **elbow**, **wrist**.
  The shoulder angle is **theta (θ)**; servo encoder counts are **ticks**.
- The project is **dr-os** ("Delta Robot Open Source"); the Python import package
  is **dros**. The scaffold name `deltakit` is retired.
- `GLOSSARY.md` is the single source of truth; its `_Avoid_` lists are binding.

## Consequences
- Code renamed to match: `src/deltakit/` -> `src/dros/`, `deltakit` imports ->
  `dros`, `effector_radius` -> `end_effector_radius`, docs updated. "arm" and
  `arm_angles` are kept, so the chains need no rename. Kinematics maths is
  unchanged.
- The `dr-os` / `dros` split (hyphen in the repo and distribution name, none in
  the import) is deliberate: `dr-os` is not a legal Python identifier.
- New terms (e.g. the tool / TCP frame) must be added to `GLOSSARY.md`, and if
  they change a convention, recorded as a further ADR.
