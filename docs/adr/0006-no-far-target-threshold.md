# 0006 - No far-target threshold on the joint bus

Supersedes the far-target part of ADR 0005.

## Context
ADR 0005 made every joint bus refuse a target further than a threshold from the
measured angles, so that one far command could not make an arm snap. That adds a
third constructor argument to every bus and forces `move_to` to step long moves.

Vendor SDKs (Feetech, Dynamixel, PiPER) do not refuse far targets. The usual
protection is joint limits plus speed and acceleration limits on the servo
itself, which ramp any target. The Feetech bus already writes those limits to the
servos on connect (ADR 0005), so a far target moves at bounded speed, and that
still holds if our process stalls.

## Decision
A joint bus takes only max velocity (rad/s) and max acceleration (rad/s^2) as
motion limits, named velocity rather than the "max speed" of ADR 0005.
`write_angles` checks joint limits through `DeltaGeometry.check_joint_limits` and
refuses commands while torque is off. It does not refuse far targets. The rest of
ADR 0005 stands.

## Considered options
- Keeping the threshold: a second guard against a bad command, but redundant with
  the servo's own limits, and one more number to choose and justify per bus.

## Consequences
- `move_to` does not need to step through targets to get past the bus.
- On a long move each servo runs at its own capped speed, so the shoulders can
  arrive at different times and the closed chains pass through configurations no
  single target describes. If that proves to matter on the robot, it is a motion
  concern for `move_to` (moving the shoulders in step), not a bus safety check.
- A wrong target within the joint limits is no longer caught by the bus. It moves
  there at the servo's limited speed and acceleration.
