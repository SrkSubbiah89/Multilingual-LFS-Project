# Archive

Superseded documents, kept for history rather than deleted.

- `Implementation_Gap_Analysis_v1.docx`, `Implementation_Gap_Analysis_v2.docx`,
  `Implementation_Gap_Analysis_v3.docx` (moved here 2026-09-11, see below),
  `gap_text.txt` — drafts of the gap-analysis document, all describing a much
  earlier project state (113 ISCO entries, 2 languages, no HITL, 832 tests).
  None of the four is current; there is no non-archived version of this
  document. For real, current project state, see `CLAUDE.md` (repo root) and
  `Documentation/PROJECT_FLOW_AND_STATUS.md`.
  **Correction, 2026-09-11**: this note previously said "the current version
  (v3's content) lives at `Documentation/Implementation/Implementation_Gap_
  Analysis.docx`" — checked directly (full text extracted and diffed against
  CLAUDE.md) during a pre-review documentation audit, and that claim was
  false: v3's actual content is essentially the same stale draft as v1/v2
  (113 ISCO entries, English+Arabic only, 832 tests — all wrong relative to
  the real 436/5-languages/2,508-test current state). The `_v3` filename and
  its non-archived location had suggested it was genuinely current without
  that ever having been verified against its real content. Moved here and
  renamed `_v3` for consistency with its siblings; the empty
  `Documentation/Implementation/` folder was left to disappear naturally
  (git does not track empty directories) rather than removed as a separate
  action.
- `ARCHIVED_ORIGINAL_KICKOFF_PROMPT.md` — the very first project-planning
  prompt (March 2026), moved here 2026-08-24. It used to live at
  `Documentation/CLAUDE.md` — the same filename as the real, current
  `CLAUDE.md` at the repo root, which in practice caused genuine confusion
  (a reader could open either file from an IDE's file tree and not be able
  to tell which was authoritative just from the name). Renamed and moved
  here specifically to remove that ambiguity: there is now exactly one file
  named `CLAUDE.md` in the whole repository, at the root, and it is always
  the current one.
- `AI_HANDOFF_PROJECT_STATE_20260812.md` — the former repo-root
  `AI_HANDOFF_PROJECT_STATE.md`, moved here 2026-08-24. It was a from-
  scratch "AI picking up this project cold" snapshot dated 2026-08-12,
  sitting at the same top-level visibility as `CLAUDE.md` and `README.md`
  — a second "what's the project's state" document was part of the same
  file-sprawl confusion that prompted this cleanup pass. Superseded by
  `CLAUDE.md` (root) and `Documentation/PROJECT_FLOW_AND_STATUS.md`, both
  of which are kept current going forward; this one is not.
- `Phase_1_Summary/` — the former `Documentation/Phase_1_Summary/`, moved
  here 2026-08-24. A 2026-08-02 point-in-time status snapshot that had
  already declared itself historical in its own text (a 2026-08-10
  addendum); moving the whole folder makes that explicit structurally,
  not just in prose. Its `Test_Suite_Report.md` was an exact duplicate of
  a file that also existed directly under `Documentation/` — that
  top-level duplicate was removed rather than archived a second time,
  since this copy already preserves it.
