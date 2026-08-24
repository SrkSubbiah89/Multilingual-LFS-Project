# Archive

Superseded documents, kept for history rather than deleted.

- `Implementation_Gap_Analysis_v1.docx`, `Implementation_Gap_Analysis_v2.docx`,
  `gap_text.txt` — earlier drafts of the gap-analysis document, describing a
  much earlier project state (113 ISCO entries, 2 languages, no HITL). The
  current version (v3's content, no longer version-suffixed) lives at
  `Documentation/Implementation/Implementation_Gap_Analysis.docx`. Git commit
  history doesn't distinguish which draft is genuinely latest (all four files
  were added in a single commit); filesystem modification times and the `_v3`
  naming convention agree on v3 being the most recent, and that determination
  is what this archive move is based on.
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
