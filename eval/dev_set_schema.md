# `eval/dev_set_v1.csv` schema

This is the schema for the **B2 development set** used only to select the
candidate-capacity parameter `K` (`--reranker-candidates`) before the single,
pre-specified confirmation run on `eval/test_set_full130.csv`.

## Why this file exists (leakage-safety rationale)

`eval/test_set_smoke20.csv` has already been used for model, beam, and
routing experiments earlier in this project, so it is not a clean
development set for a new selection decision — any K chosen against it could
be overfit to prior exploration, not genuinely validated. `eval/
test_set_full130.csv` is the frozen held-out set for B0 vs. B1 vs. B2
confirmation and must never be used to *choose* a hyperparameter, only to
*confirm* one already chosen elsewhere. `dev_set_v1.csv` is a third,
independent set whose only purpose is K selection.

## Required columns

| Column | Type | Description |
|---|---|---|
| `case_id` | string | Unique identifier. **Must not collide with any `case_id` in `test_set_smoke20.csv` or `test_set_full130.csv`** — `validate_dev_set.py` checks this by string equality, not by numeric value, so prefix your IDs (e.g. `dev001`) to make collisions structurally impossible rather than relying on numbering luck. |
| `language` | `en` \| `ar` \| `mixed` | Respondent's input language. |
| `respondent_text` | string | The free-text job-title/description a respondent would give, phrased as a survey answer — not an ISCO dictionary title copy-pasted from a reference table. |
| `gold_isco_code` | string, 4 digits | The correct ISCO-08 unit-group code for `respondent_text`. |
| `gold_label_source` | string | How the gold code was determined, e.g. `single_coder`, `two_coder_agreement`, `adjudicated`. Must not be blank — an unexplained gold label is not usable for a decision this consequential. |
| `coder_or_adjudicator` | string | Identifier (name, initials, or role) of who assigned/adjudicated the gold code. Enables tracing a disputed label back to its source. |
| `major_group` | string, 1 digit | ISCO-08 major group (`gold_isco_code[0]`) — kept as its own column (not derived on the fly) so stratification can be checked and reported without re-deriving it differently in every script that reads this file. |
| `difficulty_level` | `easy` \| `medium` \| `hard` | Coder's/adjudicator's own assessment of how ambiguous the case is. Used for reporting only, not for the K-selection rule itself. |
| `notes` | string | Free text. May be empty. Anything worth flagging (e.g. "borderline between 2512/2519", "colloquial Arabic phrasing"). |

## Coverage targets (see `validate_dev_set.py --dev-set eval/dev_set_v1.csv`)

- **Preferred**: 50+ cases total, at least 15 Arabic (`language == "ar"`).
- **Absolute minimum**: 30 cases, if data collection is genuinely constrained.
  `validate_dev_set.py` fails (non-zero exit) below 30 and warns (exit 0)
  between 30 and the preferred target.
- **Stratification**: cases should span multiple ISCO major groups, not
  cluster in one or two. `validate_dev_set.py` reports the major-group
  distribution; it does not hard-fail on skew (occupational distributions
  are not naturally uniform) but flags any major group with zero cases if
  the full 130-case set has non-trivial representation there.
- **Both languages present**: at least one `en` and one `ar` case is a hard
  requirement — a dev set that can't measure both languages can't be used
  to select a single K for a multilingual system.

## Independence requirements (hard leakage-safety gate)

`validate_dev_set.py` fails the file if:

1. Any `case_id` also appears in `test_set_smoke20.csv` or
   `test_set_full130.csv`.
2. Any `respondent_text` is an exact or near-duplicate (case-insensitive,
   whitespace-normalised) of a `input_text` value in either existing set.
   Near-duplicate here means the normalised strings match after collapsing
   internal whitespace and lowercasing — it is intentionally a strict,
   mechanical check, not a fuzzy semantic one; a human reviewer should still
   eyeball the file for paraphrases that survive this check.
3. Gold codes are not independently derived — cases must be labelled or
   adjudicated by a person, not copied from `test_set_full130.csv`'s
   `gold_isco_4digit` for a similar-sounding title.

## Non-requirements (explicitly out of scope for this file)

- `industry_text` / `education_text` (used for SRE coherence scoring
  elsewhere in `run_eval.py`) are not required here — B2 dev-set selection
  only needs Candidate Recall@K and Top-1, neither of which touches SRE.
- No `gold_isic` / `gold_isced` columns — same reason.
