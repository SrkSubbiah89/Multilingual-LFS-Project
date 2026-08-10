"""
eval/legacy824/

Task 39: Literal Legacy ISCO Conditional-LLM WISCO Development Preflight.

This package is the smallest possible glue around the earliest verified
ISCO classifier implementation, commit 824fcf235ae2f8787706cf479a07620519c914de
("Add two-stage ISCO-08 classifier agent"). It never modifies the
historical source (backend/agents/isco_classifier.py,
backend/rag/vector_store.py, backend/llm/llm_client.py at that commit) --
it only reads those files verbatim from a detached git worktree, runs
them against an isolated local Qdrant instance, and adapts WISCO
development-split rows into calls the historical classifier already
accepted historically (job title, optional context, top_k=5).

See Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_03_LEGACY_LLM_WISCO_DEV_PREFLIGHT.md
and Documentation/AI_HANDOFF/CLAUDE_TASK_39_LITERAL_LEGACY_LLM_WISCO_DEV_PREFLIGHT_FINAL_REPORT.md
for the full audited rationale.
"""

LEGACY_SHA = "824fcf235ae2f8787706cf479a07620519c914de"
