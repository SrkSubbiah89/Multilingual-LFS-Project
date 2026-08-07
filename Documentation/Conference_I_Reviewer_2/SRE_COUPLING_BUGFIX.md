# SRE-to-ISIC/ISCED Coupling Bugfix

Root-cause note, written before any code change, per Step 5.1's required
sequencing. See `eval/local_runs/step5_synthetic_integration_20260807/FINDING_sre_isic_isced_coupling_bug.md`
for the original discovery record (Step 5, 2026-08-07); this note formalizes
that finding and records the fix.

## Affected configuration

`eval/run_eval.py`, any invocation with `--sre off` (i.e. `sre_enabled=False`
passed into `run_one_case()`). Reached via `eval/ablation_runner.py`'s
`no_sre` named config (`ExperimentConfig(sre="off")`), and via any direct
`--sre off` CLI invocation of `run_eval.py`.

## Intended behaviour

- `--sre off`: ISCO, ISIC, and ISCED/ISCED-F classification all remain
  active; only the Semantic Relation Engine's crosswalk/coherence check is
  skipped. SRE-specific output fields must reflect "disabled by
  configuration", never a blank/zero value indistinguishable from "not
  measured for an unrelated reason" or "evaluated, no violation found".
- `--sre on`: all classifiers remain active; SRE runs after the ISCO/ISIC/
  ISCED base outputs it needs already exist.

## Actual pre-fix behaviour

`eval/run_eval.py:611` (pre-fix):
```python
if sre_enabled and industry_text.strip() and education_text.strip():
    isic_result = isic_clf.classify(industry_text)
    isced_result = isced_clf.classify(education_text)
    ...
    coherence = sre.analyse(...)
```
`sre_enabled` gated **both** the base ISIC/ISCED classification calls **and**
the SRE coherence check in a single `if`. With `--sre off`, `isic_clf.classify()`
and `isced_clf.classify()` were never called at all — every
`pred_isic_*`/`pred_isced_*` field stayed at its dataclass default (`""`),
producing rows indistinguishable from "these classifiers were never run",
which is exactly what happened, but for the wrong reason (SRE being off,
not ISIC/ISCED being unavailable or inapplicable).

## Root cause

A single shared `if` conditional was used to gate two logically independent
operations: (1) whether to classify ISIC/ISCED at all, and (2) whether to
additionally run the SRE coherence check on top of those classifications.
The SRE's own analysis depends on ISIC/ISCED output, which likely motivated
writing the classification calls textually adjacent to (and, by mistake,
inside) the SRE-enabled conditional — but ISIC/ISCED classification has no
actual dependency on SRE being enabled; only `sre.analyse()` does.

## Affected outputs

- **Evaluation harness only** (`eval/run_eval.py`'s `CaseResult` rows and
  every downstream artifact built from them: `eval/manifest.py` manifests,
  `eval/analyze.py` accuracy reports, `eval/ablation_runner.py`'s `no_sre`
  config output, and Step 5's synthetic-fixture evidence summary).
- **Confirmed NOT to affect production backend or API paths**:
  `backend/agents/survey_orchestrator.py::_classify_isic()` and
  `_classify_isced()` are standalone methods with no SRE-enabled gate of any
  kind (grep-verified: no `sre_enabled`-style conditional wraps either
  call in that file). `SemanticRelationEngine` is imported there but is not
  wired to gate ISIC/ISCED classification in production.
- **Confirmed NOT to affect any existing test** prior to this fix: no test
  in `backend/tests/` or `eval/` asserted on `--sre off` producing non-blank
  ISIC/ISCED predictions (the gap existed silently until Step 5's measured
  ablation run surfaced it empirically).

## Evidence impact

Step 5's `no_sre` config manifest and CaseResult CSV
(`eval/local_runs/step5_synthetic_integration_20260807/heldout/*_no_sre.csv`)
show `isic_section`/`isced_level` accuracy of exactly `0.0` in
`phase_d_per_standard_report.json`. This is **not** a measured 0% accuracy —
every `pred_isic_section`/`pred_isced_level` value in that run is blank
because the classifiers never ran, not because they ran and were wrong.

## Why the old SRE ablation comparison is invalid

The `no_sre` vs. `with_sre` ablation comparison is specifically designed to
isolate the effect of the Semantic Relation Engine while holding ISCO/ISIC/
ISCED classification constant. With this bug, `no_sre` silently also removed
ISIC/ISCED classification, so the comparison actually measured "SRE +
ISIC/ISCED classification, all enabled" vs. "everything but ISCO disabled" —
not the intended "SRE on" vs. "SRE off, same base classifiers" comparison.
Any conclusion drawn from that comparison about the SRE's effect is invalid.
ISCO-only figures for `no_sre` (which do not depend on ISIC/ISCED at all)
were unaffected and remain valid pipeline-integration evidence.

## Fix implemented

`eval/run_eval.py`'s `run_one_case()`: the ISIC/ISCED classification calls
were moved out from under the `sre_enabled` condition. ISIC/ISCED
classification now runs whenever `industry_text`/`education_text` are
present, **independent of `sre_enabled`**; the `sre.analyse()` call (and
only that call) remains gated on `sre_enabled`, nested inside the ISIC/ISCED
block so it always has fresh base outputs to consume.

Two new `CaseResult` fields were added (purely additive, default-populated,
no existing field renamed or removed):
- `sre_status: str = "not_applicable"` — one of `"evaluated"` (SRE ran),
  `"disabled_by_configuration"` (`--sre off`; base classifiers still ran),
  `"not_applicable"` (no `industry_text`/`education_text` on this row —
  independent of `--sre`), `"error"` (classification/SRE call raised).
- `sre_status_reason: str = ""` — populated with an explicit reason, e.g.
  `"semantic_relation_engine_disabled_by_configuration"` when disabled.

`sre_severity` keeps its pre-existing values (`""` not evaluated / `"NONE"`
evaluated-no-violation / `"LOW"`/`"MODERATE"`/`"HIGH"`) unchanged for
backward compatibility — `sre_status` is now the authoritative field for
distinguishing *why* `sre_severity` is blank, so no existing consumer that
reads only `sre_severity` changes behaviour, while a Step-5.1-aware consumer
can check `sre_status` for the unambiguous reason.

`--sre`'s CLI help text was corrected to describe the fixed behaviour (was:
"'off' skips the SRE block entirely for every case" — technically true but
easy to misread as including ISIC/ISCED; now explicit that only the
coherence check itself is skipped).

**Verification**: `no_sre` config re-run against the same 5-row synthetic
fixture, same dataset card, same split manifest, same reranker
(`ollama/llama3.2:1b`), post-fix, at
`eval/local_runs/step5_1_synthetic_post_sre_fix_20260807T173123Z/`. Every
row now has non-blank `pred_isic_section`/`pred_isced_level` and
`sre_status=disabled_by_configuration` (never blank, never `"NONE"`). See
that directory's `pre_fix_vs_post_fix_sre_comparison.json` for the exact
before/after figures and `SUPERSEDED_FOR_SRE_COMPARISON.md` for which
evidence remains valid.

**Tests**: `eval/test_sre_isic_isced_coupling_fix.py` (17 new tests) plus
`eval/test_run_eval_b2.py::test_sre_disabled_still_runs_isic_isced_but_skips_sre`
(rewritten — this test previously asserted the bug itself as correct
behaviour). Full suite: 1606 passed, 1 known pre-existing failure (unrelated,
`test_isco_classifier_extended.py::test_llm_used_for_low_similarity`), 1
deselected, 1 warning — zero regressions from the pre-fix baseline of 1589.
