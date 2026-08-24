# AI_HANDOFF — engineering task logs (historical evidence, not a status source)

**If you're trying to find out what the project currently is or where it
stands, stop here and go to `CLAUDE.md` (repo root) or
`Documentation/PROJECT_FLOW_AND_STATUS.md` instead.** Added 2026-08-24
after the user reported genuinely losing track of which of many `.md`
files to trust — this folder was one of the sources of that confusion,
since it sits right alongside the real status documents in the file tree
without anything marking it otherwise.

## What this folder actually is

52 dated, point-in-time task/audit reports (`CLAUDE_TASK_NN_*`,
`CLAUDE_B1/B2_*`, `CLAUDE_PROJECT_AUDIT.md`, etc.), each written at the end
of one specific piece of work, at the moment it was done. They are an
**engineering log — a paper trail of what was built, checked, or found
broken, and when** — not a maintained description of the project. None of
them are updated after the fact; a later file in this same folder may
directly contradict an earlier one, and the later one is not more
"correct" so much as later — always prefer the current root `CLAUDE.md`
over anything here.

## When it's actually useful

- You need to know the exact reasoning or evidence behind one specific
  historical change (why a threshold was picked, what a specific bug
  looked like before the fix, what a specific merge/audit found).
- You're auditing the project's history for the thesis's own
  methodology/process narrative.
- A current document explicitly cites a specific file in here as its
  source.

Otherwise, treat this folder as an archive you search into, not a folder
you read start-to-finish or use to answer "what's true right now."
