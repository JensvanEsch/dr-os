---
description: Turn recent commits into a dated devlog entry for posts and videos
argument-hint: [since, e.g. "1 week ago"]
---
1. Run `git log --since="${ARGUMENTS:-1 week ago}" --stat`.
2. Write `docs/devlog/YYYY-MM-DD.md` (today's date) with:
   - **What changed** - 3-6 bullets in plain language, no jargon dumps
   - **What I learned** - 1-3 bullets
   - **Next** - 1-3 bullets
   - **Post idea** - one 2-3 sentence caption a non-roboticist would enjoy
3. Do not commit; show me the file path.
