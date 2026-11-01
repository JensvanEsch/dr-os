# 0005 - Safety enforced inside the joint bus

## Context
`safe_write` was the only sanctioned way to command joints, but only convention
enforced that: any script could call `write_angles` directly. A real servo also
chases its target at speed, unlike `MockBus`, so a single far target could make an
arm snap.

## Decision
Every `JointBus` enforces safety in `write_angles`. It checks joint limits through
`DeltaGeometry.check_joint_limits`, refuses targets further than a threshold from
the measured angles, and refuses to run with torque off. Max speed, max
acceleration and the threshold are constructor arguments in SI units. `safe_write`
is removed.

Long moves belong to `dros.motion.move_to`, which steps through joint targets
within the threshold. This is safety stepping, not trajectory control.

## Considered options
- Keeping `safe_write` relied on discipline, which a script can bypass by accident.
- A wrapper object holding the bus privately would also work, but adds a new
  concept for the same guarantee.
- A software ramp alone stops protecting the robot if our process stalls, so speed
  and acceleration limits are set on the servo itself.

## Consequences
- `DeltaGeometry.check_joint_limits` stays the single definition of the limits,
  even though each bus calls it.
- `MockBus` enforces the limits too, so tests catch what hardware would reject.
- On a software error the robot holds its last target with torque on, because
  turning torque off would drop the end-effector. The servo power switch is the
  emergency stop, and a normal run ends at the rest position before torque off.
