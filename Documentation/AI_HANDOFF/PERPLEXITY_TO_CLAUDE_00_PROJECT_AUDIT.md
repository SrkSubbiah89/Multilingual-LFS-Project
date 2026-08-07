# Perplexity to Claude Code: Project Audit Task

**Task ID:** `00-project-audit`  
**Branch:** `reviewer2-enhancement`  
**Purpose:** Establish one verified source of truth before any further Conference I Reviewer #2 work.

## Scope

Perform an audit only. Do not modify application code, model prompts, retrieval parameters, benchmark data, configuration values, dependencies, tests, or GitHub settings in this task.

You may create exactly one completion report:

```text
Documentation/AI_HANDOFF/CLAUDE_PROJECT_AUDIT.md
```

## Required review procedure

1. Start with the current `reviewer2-enhancement` branch and compare it with `master`.
2. Run and record:

   ```bash
   git status --short
   git log --oneline --decorate -10
   git diff --stat master...HEAD
   git diff --name-status master...HEAD
   pytest backend/tests eval/ -q
   ```

3. Inspect the architecture, classifier implementations, evaluation code, documentation, and test suite.
4. Treat committed code and reproducible command outputs as facts. Do not present an uncommitted local claim as verified evidence.
5. Do not fabricate metrics, counts, validation outcomes, citations, benchmark findings, or reviewer-closure status.

## Required audit report

Create `Documentation/AI_HANDOFF/CLAUDE_PROJECT_AUDIT.md` with these sections:

1. **Repository identity**
   - Branch name, HEAD commit, comparison commit on `master`, clean/dirty status, and audit date.

2. **Project objective**
   - Concise statement of the M.Tech project objective.
   - Whether the current code aligns with the objective.

3. **Implemented architecture**
   - Frontend, backend, conversation flow, agent orchestration, persistence, authentication, HITL, audit logging, and deployment components.

4. **Classifier-method truth table**
   - ISCO-08, ISIC Rev. 4, ISCED 2011, ISCED-F 2013, and Semantic Relation Engine.
   - State exact currently implemented method, inputs, outputs, model usage, fallback behavior, and evaluation status.
   - Explicitly distinguish hierarchical RAG from hierarchy-aware keyword/rule logic.

5. **Evaluation and evidence status**
   - Automated-test status.
   - Synthetic-integration evidence.
   - Controlled-benchmark evidence.
   - WISCO preparation and leakage-audit status.
   - Real LFS-validation status.
   - Computational-analysis status.
   - Clearly separate measured results from ready-but-not-run infrastructure.

6. **Conference I Reviewer #2 matrix**
   - One row for each of the eight reviewer comments.
   - Committed evidence, current status, missing evidence, and paper-safe wording.

7. **Consistency and risk audit**
   - Mismatches among code, README, paper-support documents, project claims, standards counts, model descriptions, and test claims.
   - Security, privacy, reproducibility, and deployment risks.
   - Severity: critical, high, medium, or low.

8. **Priority actions**
   - Next five actions in exact order.
   - Identify what must be completed before WISCO benchmark execution, paper revision, real LFS claims, or any merge to `master`.

9. **Evidence boundaries**
   - Statements safe for the paper now.
   - Statements unsafe until data or experiments are completed.

10. **Audit conclusion**
    - Concise M.Tech readiness assessment.
    - Concise Conference I resubmission readiness assessment.

## Completion rules

- Make no source-code changes.
- The only allowed new file is `Documentation/AI_HANDOFF/CLAUDE_PROJECT_AUDIT.md`.
- After writing the report, run `git diff --check`.
- Commit only the audit report with:

  ```bash
  git add Documentation/AI_HANDOFF/CLAUDE_PROJECT_AUDIT.md
  git commit -m "Add Claude project audit handoff report"
  git push origin reviewer2-enhancement
  ```

- Return the exact commit SHA, test command/result, and a concise summary in Claude Code.
