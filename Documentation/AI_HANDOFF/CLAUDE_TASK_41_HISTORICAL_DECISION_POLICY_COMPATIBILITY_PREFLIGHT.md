# Task 41: Historical Decision-Policy Compatibility Preflight

## Scope and honest label

The strict literal-reproduction path is closed by Tasks 39, 40, and 40.1:

```text
Legacy source: 824fcf235ae2f8787706cf479a07620519c914de
Recovered contemporaneous qdrant-client: 1.17.0
Result: QdrantClient.search() is unavailable in that recovered runtime
```

This task is therefore not a literal historical runtime reproduction. It is a new:

```text
Historical Decision-Policy Compatibility Study
```

Its purpose is to implement and verify the old conditional LLM decision policy in the maintained WISCO-compatible evaluation stack, before any WISCO row or live LLM call occurs.

The study must never be described as:

```text
exact reproduction
literal old implementation execution
old paper result reproduced
```

## What is preserved and what is necessarily different

### Historical policy components preserved exactly

Recover the policy from `backend/agents/isco_classifier.py @ 824fcf235ae2f8787706cf479a07620519c914de`.

| Policy component | Required value/behavior |
|---|---|
| Candidate count | 5 ordered candidates |
| Confidence threshold | `0.92` |
| High-confidence path | confidence `>= 0.92` returns the semantic top candidate with no LLM invocation |
| Low-confidence path | confidence `< 0.92` invokes the reranker exactly once |
| Candidate constraint | LLM may select only a supplied candidate code |
| Historical model route | `anthropic/claude-3-5-sonnet-20241022` |
| Temperature | `0.0` |
| Prompt fields | job title, language/context, candidate code, English/Arabic title, level, score, and description |
| Response contract | JSON containing `selected_code` and `reasoning` |
| Invalid/malformed/out-of-candidate result | select semantic top candidate using the historical fallback behavior |
| Method labels | `semantic` and `llm_ranked` |

### Context components that differ and must be disclosed

| Component | Historical source | Task 41 compatibility study |
|---|---|---|
| Qdrant API | obsolete `QdrantClient.search()` | supported current query API, called through maintained code |
| Flat collection | legacy mixed-granularity `isco_occupations` | official ILO 2021 four-digit-only flat collection |
| Catalogue | legacy curated set | 436 official unit groups |
| Embedding/runtime | historical environment cannot be executed | maintained official WISCO-compatible runtime |
| Dataset | original study data | no WISCO data in this task; a future task may use WISCO development only |

These are deliberate study-context differences required to make a fair current WISCO comparison. The report must state them prominently. Do not claim that only the transport call changed.

## Branch and source

Repository:

```text
https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git
```

Create exactly one branch:

```text
reviewer2-legacy-decision-policy-compatibility-preflight-20260810
```

Verified base:

```text
reviewer2-legacy-runtime-provenance-recovery-20260810
@ 28ccd8d12eb28fe20e86c7d56231b6e358743c77
```

Historical policy source:

```text
824fcf235ae2f8787706cf479a07620519c914de
```

Use:

```bash
REPO_URL="https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git"
BASE_BRANCH="reviewer2-legacy-runtime-provenance-recovery-20260810"
BASE_SHA="28ccd8d12eb28fe20e86c7d56231b6e358743c77"
LEGACY_SHA="824fcf235ae2f8787706cf479a07620519c914de"
TASK_BRANCH="reviewer2-legacy-decision-policy-compatibility-preflight-20260810"

git remote get-url origin
git status --porcelain
git fetch origin "$BASE_BRANCH"
git fetch origin "$LEGACY_SHA"
git rev-parse "origin/$BASE_BRANCH"
git cat-file -e "$LEGACY_SHA^{commit}"
git show -s --format='%H%n%P%n%s' "$LEGACY_SHA"
git ls-remote --heads origin "$TASK_BRANCH"
git switch --create "$TASK_BRANCH" "origin/$BASE_BRANCH"
git status --porcelain
```

Stop if the remote, base SHA, historical source, clean-tree condition, or new-branch condition differs. Do not merge, rebase, reset, clean, stash, pull, force-push, alter remote settings, touch protected branches, or create a PR.

## Implementation constraints

Add the smallest additive policy component and test-only support needed to express the historical decision policy against a caller-provided ordered list of five current candidates.

The policy component must:

1. accept exactly five ordered candidate objects;
2. inspect only the first candidate confidence for the threshold decision;
3. return the first candidate with method `semantic` for confidence `>= 0.92`;
4. invoke an injected reranker callable exactly once for confidence `< 0.92`;
5. construct the historical prompt content without silently omitting required fields;
6. validate that the selected code occurs among the five supplied candidates;
7. use the historical semantic-top fallback for malformed, invalid, or out-of-candidate responses;
8. return only `semantic` or `llm_ranked`;
9. have no Qdrant, embedding-model, CrewAI, current classifier, WISCO, ISIC, ISCED, SRE, or evaluator import; and
10. make no network call itself.

The policy component must not:

- edit the detached historical worktree;
- monkeypatch or emulate `QdrantClient.search()`;
- alter current global Qdrant behavior;
- hardcode WISCO content;
- select a different candidate count;
- tune, relax, or make configurable the 0.92 threshold;
- change the historical model identifier or temperature;
- add LLM retries, fallback providers, stochastic sampling, model downloads, or provider calls;
- translate, normalize, augment, or reorder candidate data; or
- add hierarchy logic.

The current maintained flat retrieval caller may later supply the ordered official four-digit candidates. That integration is explicitly out of scope for this preflight task. This task does not run a Qdrant query or WISCO case.

## Historical-policy identity checks

Create static identity checks against `git show LEGACY_SHA:backend/agents/isco_classifier.py` that fail if any of these change:

```text
top_k = 5
threshold = 0.92
TaskType.CRITICAL
anthropic/claude-3-5-sonnet-20241022
temperature = 0.0
all required prompt fields
candidate-only selected-code validation
semantic / llm_ranked method labels
semantic-top invalid-response fallback
```

The compatibility implementation may reorganize code, but its explicit policy manifest and tests must make every historical preservation point traceable. Quote direct legacy source evidence in the final report.

## Required hermetic tests

Use only fake candidates and injected fake rerankers. No test may read WISCO, load/download an embedding model, connect to Qdrant, call an LLM/provider, construct CrewAI agents, or import current evaluation/classifier modules.

At minimum test:

1. exactly five candidates are required;
2. fewer or more candidates fail closed;
3. confidence exactly `0.92` takes semantic path with zero reranker calls;
4. confidence above `0.92` takes semantic path with zero reranker calls;
5. confidence below `0.92` invokes reranker exactly once;
6. valid candidate-code JSON produces `llm_ranked`;
7. invalid JSON falls back to semantic top candidate;
8. missing `selected_code` falls back to semantic top candidate;
9. out-of-candidate code falls back to semantic top candidate;
10. reranker exception falls back to semantic top candidate;
11. prompt includes every required historical field for all five candidates;
12. candidate order is unchanged in the prompt and returned evidence;
13. only `semantic` and `llm_ranked` labels can be emitted;
14. no prohibited import appears in the compatibility-policy module;
15. static legacy-policy identity checks reject a changed threshold, candidate count, model route, temperature, or fallback condition; and
16. test fixtures prove no network-capable reranker is used.

Run focused tests, then:

```text
python -m pytest backend/tests eval/ -q
```

Any test failure is a hard stop.

## Preservation checks

Before and after the task, prove byte identity for:

1. Task 36 raw official WISCO flat and hierarchical CSVs;
2. Task 37/37.1 analysis evidence and Task 38 documentation evidence;
3. Task 39, 40, and 40.1 reports and task-specific source/test files;
4. official ILO catalogue and collection-builder code;
5. B1 frozen configuration and B2 safety files; and
6. every protected artefact enumerated by Task 40.1.

There must be zero live Qdrant connection, zero WISCO file read, zero LLM/Anthropic call, zero model download, zero Docker command, and zero source change in the detached historical worktree.

## Stop status

Use exactly one:

```text
HISTORICAL_DECISION_POLICY_COMPATIBILITY_PREFLIGHT_READY: yes
HISTORICAL_DECISION_POLICY_COMPATIBILITY_PREFLIGHT_READY: no
```

Set `yes` only if:

1. historical policy source identity is verified;
2. the compatibility policy preserves every required behavior;
3. all hermetic tests pass;
4. the full suite passes;
5. preservation checks pass; and
6. no forbidden live operation occurred.

On `yes`, stop. Do not run WISCO, integrate live retrieval, create an API credential request, call an LLM, make a paper update, or calculate an accuracy value.

## Final report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_41_HISTORICAL_DECISION_POLICY_COMPATIBILITY_PREFLIGHT_FINAL_REPORT.md
```

Include:

1. branch, final SHA, verified base SHA, and historical source SHA;
2. explicit non-literal compatibility-study label;
3. a preserved-versus-different table matching this task document;
4. direct legacy source evidence for each preserved policy component;
5. changed files and why none alter the detached historical source;
6. all required focused and full-suite commands with literal output;
7. test coverage mapped to the required policy rules;
8. before/after preservation evidence;
9. confirmation of zero WISCO/Qdrant/LLM/model/Docker activity;
10. exact status and whether the next dev-only evaluation task is authorised; and
11. confirmation of clean working tree, untouched protected branches, no PR, and no forbidden Git operation.

Commit only task-specific compatibility-policy/test files and the final report. Push only:

```text
reviewer2-legacy-decision-policy-compatibility-preflight-20260810
```

Do not open a pull request.

## Claude Code execution prompt

```text
Execute Task 41 exactly as specified in:
Documentation/AI_HANDOFF/CLAUDE_TASK_41_HISTORICAL_DECISION_POLICY_COMPATIBILITY_PREFLIGHT.md

Create only reviewer2-legacy-decision-policy-compatibility-preflight-20260810 from reviewer2-legacy-runtime-provenance-recovery-20260810 @ 28ccd8d12eb28fe20e86c7d56231b6e358743c77.

This is a Historical Decision-Policy Compatibility Study, not a literal old-runtime reproduction. Preserve the old policy exactly: five ordered candidates, 0.92 threshold, semantic path at >=0.92, exactly one injected reranker call below threshold, candidate-only selection, historical prompt fields, Claude 3.5 Sonnet identifier, temperature 0.0, JSON validation, semantic-top fallback, and semantic/llm_ranked labels.

Disclose that the current WISCO-compatible official four-digit index and supported Qdrant API differ from the historical legacy environment. Do not claim that only the transport changed.

Implement and hermetically test only the additive policy component. Do not run WISCO, query Qdrant, call Anthropic or any LLM, download/load a model, invoke Docker, or integrate a live retrieval caller. Preserve all historical/official evidence and run the full suite. Stop after reporting preflight status, commit only task files/report, push only the new branch, and do not open a PR.
```
