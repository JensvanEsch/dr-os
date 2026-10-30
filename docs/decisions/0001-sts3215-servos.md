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
