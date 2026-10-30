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
