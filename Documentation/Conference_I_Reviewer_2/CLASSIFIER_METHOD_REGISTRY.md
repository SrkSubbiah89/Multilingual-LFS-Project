# Classifier Method Registry — Guide

Answers reviewer comment 6: "multiple LLM roles are unclear — the specific
task handled by each model must be documented and measurable."

## What it is

`backend/agents/method_registry.py` defines `REGISTRY`, a list of
`ClassifierMethodEntry` rows — one row per (component, method) — covering
every classifier/agent named in the reviewer response: `LanguageProcessor`,
`ConversationManager`, `ISCOClassifier`, `ISICClassifier`, `ISCEDClassifier`,
`SemanticRelationEngine`, `ValidationAgent`, `HITLQualityManager`,
`SurveyOrchestrator`.

Every field was hand-derived by reading the component's actual source code
(dataclass/Pydantic fields, `get_llm(TaskType.*)` calls, CrewAI
`Agent(role=...)` strings, threshold constants) — not assumed or inferred
from the module's docstring alone.

## How to regenerate it

```bash
python -m backend.agents.method_registry --out Documentation/Conference_I_Reviewer_2/generated/
```

Writes `classifier_method_registry.json` and `classifier_method_registry.md`
to `generated/`. **That generated file, not this one, is the canonical,
always-fresh registry** — this document is a stable guide explaining what
it is and how to regenerate it; it doesn't duplicate the row data (which
would go stale the moment `REGISTRY` changes).

## Reading the registry

- **`category`** is one of `deterministic` / `retrieval` / `llm` / `hybrid`
  — never guessed; it reflects what the code actually does for that method.
- **`affects_hitl_escalation`** is the most reviewer-relevant, least
  obvious field. It is **grep-verified against `survey_orchestrator.py`**,
  not assumed: only `ISCOClassifier`'s per-response confidence and
  `HITLQualityManager`'s own scoring feed the production escalation
  decision. `ValidationAgent.rule_violations` and
  `SemanticRelationEngine`'s `SemanticCoherence` are attached to
  `TurnResult` but are **not** read anywhere near the escalation path today
  — a real, currently-unaddressed architectural gap, documented honestly
  rather than smoothed over. `backend/tests/test_method_registry.py`
  regression-guards this claim against the actual wiring.
- **`evaluated` / `evaluated_ref`** is honest: only `ISCOClassifier` has any
  evaluation history at all (the `eval/` B2 harness), and even that is
  ISCO-digit accuracy only. Every ISIC/ISCED/SRE/HITL row is
  `evaluated=False` until Section D/E of this work produces a real,
  citable run manifest.
- **Stub rows** (`isic_hierarchical_retrieval`,
  `iscedf_hierarchical_retrieval`) have `input_fields`/`output_schema`
  populated with the *intended future* schema, clearly marked
  `(planned)` in every value and `evaluated=False` — useful for the
  agent-role diagram export, but never implying current capability. See
  `backend/agents/classifier_methods.py`'s `NOT_IMPLEMENTED_METHODS`.

## Safe access surface

CLI export only, in this pass — no FastAPI route was added (avoids
introducing a new auth-surface decision without the authors' input). If a
programmatic/internal API route is wanted later, it should reuse
`export_json()`/`export_markdown()` from `method_registry.py` behind
whatever auth the rest of `backend/api/` uses for internal/admin routes.

## Tests

`backend/tests/test_method_registry.py` — covers row coverage per named
component, the honesty checks above (not-implemented rows never claim
evaluation), the `affects_hitl_escalation` regression guard, and JSON/
Markdown export round-trips.
