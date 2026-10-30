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
