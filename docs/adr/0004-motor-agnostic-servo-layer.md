# 0004 - Motor-agnostic servo layer

Partly supersedes ADR 0001: serial-bus servos stay, but the STS3215 is no longer
the fixed choice.

## Context
No hardware exists yet, and the choice of servo is open. The Feetech HL-3915 is
attractive for its constant-current mode, which the STS3215 is not known to
offer. Vendor wiki memory tables show the STS and HLS families sharing the
registers needed for position control, so one implementation can plausibly cover
both. This needs confirming on real servos.

## Decision
`JointBus` is the motor-agnostic boundary: it speaks SI units and knows nothing
about motors. One `FeetechBus` covers the Feetech families in use, with a small
per-model table for differences such as tick resolution and speed units. Another
vendor gets its own `JointBus` implementation when it is actually needed.

The servo is the HL-3915 if its datasheet shows at least 6 kg·cm of continuous
torque, otherwise the 12 V STS3215.

## Considered options
A generic UART servo layer above the vendors was rejected. Designed from a single
vendor it would be guesswork, and `JointBus` already provides the boundary.

## Consequences
- `src/dros/servo/sts3215.py` becomes `feetech.py`, and ticks stay inside it.
- `JointBus` stays position-only. Current control comes later with its own ADR,
  because using it well needs a dynamics model.
- The HLS table says a speed of 0 stops the servo, while on the STS it is believed
  to mean maximum speed (unconfirmed), so `FeetechBus` never sends 0.
