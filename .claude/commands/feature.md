---
description: Plan and implement a feature the dr-os way (plan, ADR if needed, tests, code)
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
