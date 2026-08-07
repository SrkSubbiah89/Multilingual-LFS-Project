"""
eval/dev_sweep.py

B2 dev-set K-sweep: runs eval/dev_set_v1.csv through the SAME B1 pooling
pipeline (backend.agents.isco_classifier.ISCOClassifier, branch_collapse
hardcoded False -- see module docstring below) once per candidate value of
K (--reranker-candidates), applies the pre-specified B2 OPERATIONAL
ELIGIBILITY RULE (not a statistical stability test -- a deterministic,
temperature=0 pipeline has no run-to-run variance for a stability test to
measure), and writes a Markdown report recording every eligibility check
and the resulting decision.

This script NEVER touches eval/test_set_full130.csv -- only --dev-set. It
exists specifically so K can be chosen without looking at the held-out
confirmation set. See eval/dev_set_schema.md for why eval/
test_set_smoke20.csv is also excluded (reused for earlier experiments).

Experiment-separation constraints (do not relax without re-reading the B2
spec this script was built against):
  - branch_collapse is hardcoded False. B2 varies ONLY K; it does not
    combine K with the pre-fix branch-collapse behaviour (that's B0, a
    separate, already-frozen baseline).
  - The reranker model, prompt, parser, and fallback-on-failure behaviour
    are exactly ISCOClassifier's existing _llm_select_from_candidates() /
    _parse_llm_response() -- the same ones B0/B1 use. This script does NOT
    wire in backend.agents.isco_reranker_strict.StrictReranker (that is
    B3-Reliability, a separate future ablation) and does NOT change sort
    policy (that is B3-Sort, also separate and not activated here).
  - capture_pool_metadata is always True for sweep runs (this script's only
    reason to exist is to look at the resulting metadata) but that flag is
    observational-only and does not change what gets classified or how --
    see hierarchical_store.py's capture_pool_metadata docstring.

B2 operational eligibility rule
---------------------------------------------------------------------------
Reference configuration: K=5 (--reference-k) is the B1-compatible reference
run on the same dev set, hardware, model, prompt, timeout, input-ordering
policy, and logging settings as every candidate K -- all of that is already
guaranteed here because every K in the sweep runs in the same process,
against the same dev_rows, through the same build_system()/run_one_case()
call path, differing only in --reranker-candidates.

A candidate K is operationally eligible only if ALL of the following hold
(see check_eligibility()):

1. Hard failures -- no OOM, process crash, corrupted output file, or
   incomplete run. Any one of these rejects K outright, unconditionally.
   Operationalised as: KSweepResult.crashed (an exception escaped the
   sweep loop for this K -- see run_sweep_for_k()'s try/except), .
   incomplete_run (fewer cases completed than were in the dev set),
   .corrupted_output (the raw per-K CSV failed a read-back row-count
   check), or n_hard_failures > 0 (a per-case exception with no
   prediction produced, e.g. a Qdrant/network error unrelated to the
   reranker).

2. Timeout and invalid-output events -- raw counts AND rates are recorded
   (KSweepResult.n_timeouts / n_invalid_outputs / timeout_rate /
   invalid_output_rate). K is rejected if
   (n_timeouts + n_invalid_outputs) exceeds the reference K's
   (n_timeouts + n_invalid_outputs) by more than
   --max-additional-timeout-or-invalid (default 1). A timed-out or
   invalid-output case ALWAYS counts as incorrect in
   top1_accuracy_all_case (denominator = every case in the dev set), even
   though isco_classifier.py's existing fallback-on-failure behaviour
   still produces *some* prediction for that case (which could coincidentally
   equal gold) -- crediting a fallback guess as correct would misstate
   reliability. top1_accuracy_conditional is reported separately, computed
   only over cases that neither timed out nor produced invalid output
   (denominator = n_successful_cases), so "how good is the reranker when it
   actually runs" and "how reliable is the whole K configuration" don't get
   conflated into one number.

3. Latency -- median, P95, and max end_to_end_latency_ms are recorded per
   K, with the first --warmup-cases dev-set calls (default 2) run and
   discarded before the measured pass, per case, so cold-start latency
   (model/connection warm-up) doesn't distort the percentiles. K is
   rejected if its P95 latency exceeds --p95-ceiling-factor (default 1.5)
   times the reference K's P95 latency; the actual factor is always
   recorded in the report regardless of pass/fail.

4. Memory -- peak_memory_mb is sampled (via psutil, if installed; None
   with a logged reason if not) as the maximum RSS observed while that K's
   cases ran. K is rejected if peak_memory_mb exceeds --memory-budget-mb
   (no default -- if omitted, this specific check is skipped and the
   report says so explicitly rather than silently passing). Swap/thrashing
   is not independently detectable from this process without OS-level
   tooling this harness doesn't have -- a MemoryError raised during the
   run is caught and treated as `crashed` (see #1); anything short of an
   actual MemoryError is not claimed to be detected here, and the report
   says so.

5. Selection after eligibility -- among OPERATIONALLY ELIGIBLE K values
   only: maximise Candidate Recall@K; if recall is equivalent or nearly
   equivalent (within --recall-close-threshold, default 0.02), choose the
   best top1_accuracy_all_case (not conditional -- see #2 for why); if
   still tied, select the smaller K. eval/test_set_full130.csv is never
   read by this script, so it cannot factor into this decision even by
   accident.

Baseline config (--baseline-config, mandatory)
---------------------------------------------------------------------------
B2 is only meaningful as a K-only ablation if every other pipeline setting
is *exactly* what produced the frozen B1 confirmation result (54/130 on
eval/test_set_full130.csv, see eval/configs/b1_frozen.json's _source_csv).
--baseline-config points at a JSON file (see eval/configs/b1_frozen.json)
recording that frozen configuration. This script does NOT expose separate
--reranker-model/--beam/--stage1-mode/--disable-keyword-map flags -- those
values are read directly FROM the baseline file, so there is no separate
CLI value that could silently disagree with it. reranker_candidates (K) is
the only pipeline setting this script ever varies from the baseline.

What "load and assert" means concretely, since most of the frozen fields
have no independent value to compare against once they're sourced from the
file (see above): reranker_model/beam/stage1_mode/keyword_map_enabled are
*sourced from*, not *checked against*, the baseline (the strongest possible
guarantee that they can't drift). branch_collapse must literally be False
in the file (hard requirement -- a baseline with branch_collapse=true would
not describe B1 at all). llm_temperature, timeout_s, and
implementation_fingerprint have no CLI knob at all -- they're either
hardcoded constants (HARDCODED_LLM_TEMPERATURE here mirrors
run_eval.build_system()'s hardcoded 0.0; timeout_s is
backend.llm.llm_client._OLLAMA_INFERENCE_TIMEOUT) or a composite sha256
fingerprint over every B1 decision-critical function's live source (there
is no explicit prompt-version string anywhere in this codebase, so the
fingerprint is the closest available drift detector -- see
compute_composite_fingerprint() and task C in eval/configs/b1_frozen.json's
_notes for the exact component list and why each is/isn't included).
ollama_model_identity's digest is likewise checked live (task B) -- the
"latest" tag is mutable, so the pinned digest, not the tag, is what's
actually compared. assert_baseline_matches_codebase() recomputes all of
these against the CURRENT codebase/environment and raises
BaselineMismatchError, refusing to run any case, if the frozen file's
claims no longer hold -- e.g. someone edited the reranker prompt, bumped
the Ollama timeout, or re-pulled the model under the same tag, without
re-freezing eval/configs/b1_frozen.json. beam_evidence (task A) is a
separate, non-negotiable gate: see check_beam_evidence() and
--confirm-inferred-beam below.

Determinism (--seed)
---------------------------------------------------------------------------
--seed (default: run_eval.RANDOM_SEED, i.e. 42) drives two independent,
reproducible permutations, both recorded in the Markdown report:
  - dev-row execution order: the loaded dev set is shuffled ONCE with
    random.Random(seed) before the sweep starts, and every K runs through
    that SAME shuffled order (not a fresh shuffle per K) so cases stay
    directly comparable across K values case-by-case.
  - K execution order: the K values themselves run in
    random.Random(seed + 1)'s shuffled order (a distinct derived seed so
    the two permutations don't move in lockstep), not ascending numeric
    order -- guards against any systematic warm-up/ordering effect from
    always running small K before large K.
Both are pure functions of --seed -- re-running with the same seed
reproduces the same dev-row order and the same K order.

Usage
-----
    python eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv   # first!
    python eval/dev_sweep.py --dev-set eval/dev_set_v1.csv \
        --baseline-config eval/configs/b1_frozen.json --k-values 5,8,10,15,20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402
from backend.llm.llm_client import _OLLAMA_BASE_URL, _OLLAMA_HEALTH_TIMEOUT, _OLLAMA_INFERENCE_TIMEOUT  # noqa: E402
from backend.rag.hierarchical_store import HierarchicalISCOStore  # noqa: E402

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "dev_sweep"
DEFAULT_BASELINE_CONFIG = Path(__file__).resolve().parent / "configs" / "b1_frozen.json"

DEFAULT_K_VALUES = [5, 8, 10, 15, 20]
DEFAULT_REFERENCE_K = 5
DEFAULT_RECALL_CLOSE_THRESHOLD = 0.02
DEFAULT_P95_CEILING_FACTOR = 1.5
DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID = 1
DEFAULT_WARMUP_CASES = 2
DEFAULT_SWEEP_SEED = run_eval.RANDOM_SEED

# run_eval.build_system()'s hierarchical/flat branches hardcode
# ISCOClassifier(llm_temperature=0.0, ...) -- not a CLI-configurable value
# anywhere in this harness. Mirrored here so assert_baseline_matches_codebase()
# has something authoritative to check the frozen file's claim against.
HARDCODED_LLM_TEMPERATURE = 0.0

REQUIRED_BASELINE_FIELDS = [
    "reranker_model", "beam", "stage1_mode", "keyword_map_enabled",
    "branch_collapse", "llm_temperature", "timeout_s",
    "beam_evidence", "ollama_model_identity", "implementation_fingerprint",
    "baseline_validity",
]

# Conference I Reviewer #2, Task 04 (B1 baseline quarantine). A deliberately
# small, closed enum -- "historical_stale_requires_rerun" is the only status
# that exists today (every frozen baseline in this repo is exactly this,
# since none has ever been re-frozen against a post-refactor codebase).
# "current_verified_ready" is reserved for a FUTURE re-freeze that has both
# a passing implementation-fingerprint check AND b2_sweep_permitted=true --
# it is not used anywhere yet. No other status string is ever valid; an
# unrecognised status is a validate_baseline_shape() error, not a warning.
BASELINE_VALIDITY_STATUSES = ("historical_stale_requires_rerun", "current_verified_ready")

# The exact, sorted set of B1 decision-critical functions hashed into
# implementation_fingerprint. For the B1 hierarchical code path, candidate
# pooling/dedup and UnitCandidate serialisation-before-the-reranker both
# happen INLINE inside HierarchicalISCOStore._hierarchical_search() (see
# eval/configs/b1_frozen.json's _notes) -- there is no separate pooling
# helper or serialisation function to add for B1 specifically.
# backend/rag/candidate_pool.py is NOT part of this list: it is not called
# by the B1 code path (see that module's own docstring), so hashing it
# would create false-positive drift signals for B3-Sort-only changes.
IMPLEMENTATION_FINGERPRINT_TARGETS = {
    "HierarchicalISCOStore._hierarchical_search": lambda: HierarchicalISCOStore._hierarchical_search,
    "ISCOClassifier._llm_select_from_candidates": lambda: run_eval.ISCOClassifier._llm_select_from_candidates,
    "ISCOClassifier._parse_llm_response": lambda: run_eval.ISCOClassifier._parse_llm_response,
}


class BaselineMismatchError(Exception):
    """Raised by assert_baseline_matches_codebase() -- refuses to run any
    B2 case rather than produce a comparison that silently isn't actually
    comparable to the frozen B1 result."""


# ---------------------------------------------------------------------------
# Baseline config: load, shape-validate, and cross-check against the
# CURRENT codebase (see module docstring's "Baseline config" section for
# what's sourced-from vs. asserted-against and why).
# ---------------------------------------------------------------------------

def load_baseline_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_baseline_shape(baseline: dict) -> list:
    """Structural/type checks only -- no comparison against the codebase
    (that's assert_baseline_matches_codebase()) and no beam-evidence policy
    decision (that's check_beam_evidence()). Pure, unit-testable. Returns a
    list of error strings; empty means the shape is valid."""
    errors = []
    missing = [f for f in REQUIRED_BASELINE_FIELDS if f not in baseline]
    if missing:
        errors.append(f"missing required field(s): {missing}")
        return errors  # remaining checks assume presence

    model = baseline["reranker_model"]
    if not isinstance(model, str) or not (model.startswith("ollama/") or model.startswith("anthropic/")):
        errors.append(f"reranker_model={model!r} must be a string starting with 'ollama/' or 'anthropic/'")
    if not isinstance(baseline["beam"], int) or baseline["beam"] < 1:
        errors.append(f"beam={baseline['beam']!r} must be a positive integer")
    if baseline["stage1_mode"] not in ("description", "leaf_vote"):
        errors.append(f"stage1_mode={baseline['stage1_mode']!r} must be 'description' or 'leaf_vote'")
    if not isinstance(baseline["keyword_map_enabled"], bool):
        errors.append(f"keyword_map_enabled={baseline['keyword_map_enabled']!r} must be a bool")
    if not isinstance(baseline["branch_collapse"], bool):
        errors.append(f"branch_collapse={baseline['branch_collapse']!r} must be a bool")
    if not isinstance(baseline["llm_temperature"], (int, float)):
        errors.append(f"llm_temperature={baseline['llm_temperature']!r} must be numeric")
    if not isinstance(baseline["timeout_s"], (int, float)):
        errors.append(f"timeout_s={baseline['timeout_s']!r} must be numeric")

    beam_evidence = baseline["beam_evidence"]
    if not isinstance(beam_evidence, dict) or beam_evidence.get("status") not in ("confirmed", "inferred"):
        errors.append(
            f"beam_evidence={beam_evidence!r} must be a dict with status in "
            f"('confirmed', 'inferred')"
        )

    identity = baseline["ollama_model_identity"]
    if not isinstance(identity, dict) or not identity.get("tag") or not identity.get("digest"):
        errors.append(
            f"ollama_model_identity={identity!r} must be a dict with non-empty 'tag' and 'digest'"
        )

    fingerprint = baseline["implementation_fingerprint"]
    if (
        not isinstance(fingerprint, dict)
        or not isinstance(fingerprint.get("components"), dict)
        or not fingerprint.get("components")
        or not isinstance(fingerprint.get("composite_sha256"), str)
        or not fingerprint.get("composite_sha256")
    ):
        errors.append(
            f"implementation_fingerprint={fingerprint!r} must be a dict with a non-empty "
            f"'components' dict and a non-empty 'composite_sha256' string"
        )

    errors.extend(_validate_baseline_validity_shape(baseline.get("baseline_validity")))
    return errors


def _validate_baseline_validity_shape(validity) -> list:
    """Conference I Reviewer #2, Task 04. Structural/consistency checks for
    the baseline_validity object ONLY -- this never decides whether a sweep
    may proceed (that's check_baseline_validity_permits_sweep(), part of
    BASELINE_CODEBASE_CHECKS below); it only rejects a MALFORMED or
    internally-inconsistent baseline_validity block, the same way the rest
    of this function rejects a malformed beam_evidence/ollama_model_identity/
    implementation_fingerprint. Missing baseline_validity entirely is caught
    upstream by the REQUIRED_BASELINE_FIELDS check and never reaches here."""
    errors = []
    if not isinstance(validity, dict):
        errors.append(f"baseline_validity={validity!r} must be a dict")
        return errors

    status = validity.get("status")
    if status not in BASELINE_VALIDITY_STATUSES:
        errors.append(
            f"baseline_validity.status={status!r} must be one of {BASELINE_VALIDITY_STATUSES}"
        )

    permitted = validity.get("b2_sweep_permitted")
    if not isinstance(permitted, bool):
        errors.append(f"baseline_validity.b2_sweep_permitted={permitted!r} must be a bool")
    elif status == "historical_stale_requires_rerun" and permitted is True:
        errors.append(
            "baseline_validity.status='historical_stale_requires_rerun' cannot combine with "
            "b2_sweep_permitted=true -- internally inconsistent (a stale baseline can never "
            "permit a sweep)"
        )

    for field in ("reason", "permitted_use", "re_freeze_requires"):
        if not isinstance(validity.get(field), str) or not validity.get(field):
            errors.append(f"baseline_validity.{field} must be a non-empty string")

    return errors


# ---------------------------------------------------------------------------
# Implementation fingerprint: composite hash over every B1 decision-critical
# function's live source (task C). Sorted by component name before hashing
# so composite_sha256 never depends on dict/insertion order.
# ---------------------------------------------------------------------------

def compute_composite_fingerprint() -> dict:
    """Returns {"components": {name: 16-hex-char hash}, "composite_sha256":
    64-hex-char hash}. The composite is sha256 of the sorted
    "name:hash\\n"-joined component lines -- deterministic regardless of
    dict ordering, and any single function's edit changes both its own
    component hash and the composite."""
    components = {}
    for name in sorted(IMPLEMENTATION_FINGERPRINT_TARGETS):
        fn = IMPLEMENTATION_FINGERPRINT_TARGETS[name]()
        src = inspect.getsource(fn)
        components[name] = hashlib.sha256(src.encode()).hexdigest()[:16]

    composite_input = "\n".join(f"{name}:{components[name]}" for name in sorted(components))
    composite = hashlib.sha256(composite_input.encode()).hexdigest()
    return {"components": components, "composite_sha256": composite}


def compute_live_prompt_fingerprint() -> str:
    """Deprecated single-function fingerprint, kept only because
    compute_composite_fingerprint() is now the authoritative check --
    superseded by it, retained as a thin wrapper in case other code still
    imports this name. sha256 of ISCOClassifier._llm_select_from_candidates
    ()'s current source, truncated to 16 hex chars."""
    src = inspect.getsource(run_eval.ISCOClassifier._llm_select_from_candidates)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ollama model identity: a stable digest, not the mutable "latest" tag
# (task B). Metadata-only HTTP calls (GET /api/tags, GET /api/version) --
# never /api/generate or /api/chat, so no inference is ever performed here.
# ---------------------------------------------------------------------------

def resolve_ollama_model_identity(tag: str) -> dict:
    """Best-effort live lookup of *tag*'s installed digest via Ollama's
    /api/tags (metadata only, no inference). Returns
    {"status": "confirmed", "tag", "digest", "parameter_size",
    "quantization_level", "family", "size_bytes"} on success, or
    {"status": "unavailable", "tag", "reason"} if Ollama is unreachable or
    the tag isn't pulled -- callers must fail closed on "unavailable"
    (task B.4), never silently skip the check."""
    try:
        req = urllib.request.Request(f"{_OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=_OLLAMA_HEALTH_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"status": "unavailable", "tag": tag, "reason": f"{type(exc).__name__}: {exc}"}

    for m in data.get("models", []):
        if m.get("name") == tag or m.get("model") == tag:
            details = m.get("details", {}) or {}
            return {
                "status": "confirmed",
                "tag": tag,
                "digest": m.get("digest", ""),
                "parameter_size": details.get("parameter_size", ""),
                "quantization_level": details.get("quantization_level", ""),
                "family": details.get("family", ""),
                "size_bytes": m.get("size"),
            }
    return {"status": "unavailable", "tag": tag, "reason": f"tag {tag!r} not found in /api/tags response"}


def resolve_ollama_version() -> Optional[str]:
    """GET /api/version -- metadata only, no inference. Returns None
    (best-effort) if unreachable."""
    try:
        req = urllib.request.Request(f"{_OLLAMA_BASE_URL}/api/version")
        with urllib.request.urlopen(req, timeout=_OLLAMA_HEALTH_TIMEOUT) as resp:
            return json.loads(resp.read()).get("version")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def check_branch_collapse_false(baseline: dict) -> tuple:
    if baseline["branch_collapse"] is not False:
        return False, (
            f"branch_collapse={baseline['branch_collapse']!r} in the baseline config, but B2 "
            f"requires branch_collapse=False (B1 pooling) -- this file does not describe B1."
        )
    return True, "branch_collapse=False confirmed"


def check_llm_temperature(baseline: dict) -> tuple:
    if baseline["llm_temperature"] != HARDCODED_LLM_TEMPERATURE:
        return False, (
            f"llm_temperature={baseline['llm_temperature']!r} in the baseline config, but "
            f"run_eval.build_system() hardcodes {HARDCODED_LLM_TEMPERATURE!r} for every "
            f"hierarchical run (not a CLI-configurable value in this harness)."
        )
    return True, f"llm_temperature={HARDCODED_LLM_TEMPERATURE} confirmed against the hardcoded value"


def check_timeout(baseline: dict) -> tuple:
    if baseline["timeout_s"] != _OLLAMA_INFERENCE_TIMEOUT:
        return False, (
            f"timeout_s={baseline['timeout_s']!r} in the baseline config, but "
            f"backend.llm.llm_client._OLLAMA_INFERENCE_TIMEOUT is currently "
            f"{_OLLAMA_INFERENCE_TIMEOUT!r}."
        )
    return True, f"timeout_s={_OLLAMA_INFERENCE_TIMEOUT} confirmed against the live constant"


def check_implementation_fingerprint(baseline: dict) -> tuple:
    live_fingerprint = compute_composite_fingerprint()
    baseline_fingerprint = baseline["implementation_fingerprint"]
    if baseline_fingerprint.get("composite_sha256") != live_fingerprint["composite_sha256"]:
        changed = sorted(
            name for name in live_fingerprint["components"]
            if baseline_fingerprint.get("components", {}).get(name) != live_fingerprint["components"][name]
        )
        return False, (
            f"implementation_fingerprint.composite_sha256="
            f"{baseline_fingerprint.get('composite_sha256')!r} in the baseline config, but the "
            f"live composite is {live_fingerprint['composite_sha256']!r} -- component(s) with a "
            f"changed source hash: {changed or '(component set itself differs)'}. B1 decision-"
            f"critical logic has changed since the frozen B1 run. Re-run and re-freeze B1 before "
            f"trusting a B2 comparison."
        )
    return True, f"implementation_fingerprint.composite_sha256 confirmed ({live_fingerprint['composite_sha256'][:16]}...)"


def check_baseline_validity_permits_sweep(baseline: dict) -> tuple:
    """Conference I Reviewer #2, Task 04. A NEW, ADDITIVE check -- fails
    closed whenever the baseline's own self-reported validity metadata says
    it must not seed a B2 sweep, independent of (and in addition to) the
    implementation-fingerprint check below. Every check in
    BASELINE_CODEBASE_CHECKS is OR'd into one failure list by
    assert_baseline_matches_codebase() -- adding this one only makes that
    gate stricter, never weaker, and does not change how any existing check
    (especially check_implementation_fingerprint()) decides pass/fail."""
    validity = baseline["baseline_validity"]
    status = validity.get("status")
    permitted = validity.get("b2_sweep_permitted")
    if status != "current_verified_ready" or permitted is not True:
        return False, (
            f"baseline_validity.status={status!r}, b2_sweep_permitted={permitted!r} -- this "
            f"baseline is historical/stale and must not seed a B2 sweep. Reason: "
            f"{validity.get('reason', '(none recorded)')} Re-freeze requires: "
            f"{validity.get('re_freeze_requires', 'separate explicit approval and a fresh B1 run')}"
        )
    return True, "baseline_validity confirms this baseline is current and permitted for a B2 sweep"


def check_ollama_model_identity(baseline: dict) -> tuple:
    identity = resolve_ollama_model_identity(baseline["ollama_model_identity"]["tag"])
    if identity["status"] != "confirmed":
        return False, (
            f"could not confirm the live identity of Ollama model "
            f"{baseline['ollama_model_identity']['tag']!r}: {identity.get('reason')}. "
            f"Failing closed -- see task B.4."
        )
    if identity["digest"] != baseline["ollama_model_identity"]["digest"]:
        return False, (
            f"ollama_model_identity.digest={baseline['ollama_model_identity']['digest']!r} in the "
            f"baseline config, but the locally installed {baseline['ollama_model_identity']['tag']!r} "
            f"currently resolves to digest={identity['digest']!r} -- the 'latest' tag is mutable and "
            f"has moved (e.g. re-pulled) since the frozen B1 run. Re-run and re-freeze B1 with the "
            f"currently-installed model, or reinstall the exact frozen digest, before trusting a "
            f"B2 comparison."
        )
    return True, f"ollama_model_identity.digest confirmed for {identity['tag']!r}"


# The 5 codebase/environment cross-checks assert_baseline_matches_codebase()
# runs, in order -- also used individually by eval/pre_run_check.py so its
# checklist reports each as its own PASS/FAIL line rather than one bundled
# exception. Each is (name, fn(baseline) -> (ok, message)).
BASELINE_CODEBASE_CHECKS = [
    ("baseline_validity", check_baseline_validity_permits_sweep),
    ("branch_collapse", check_branch_collapse_false),
    ("llm_temperature", check_llm_temperature),
    ("timeout_s", check_timeout),
    ("implementation_fingerprint", check_implementation_fingerprint),
    ("ollama_model_identity", check_ollama_model_identity),
]


def assert_baseline_matches_codebase(baseline: dict) -> None:
    """Runs every check in BASELINE_CODEBASE_CHECKS -- the frozen baseline's
    own self-reported validity status (baseline_validity, Task 04), claims
    about non-CLI-configurable pipeline behaviour (temperature, timeout,
    implementation fingerprint, branch_collapse), and the pinned Ollama
    model identity, against the CURRENT codebase/environment.
    reranker_model/beam/stage1_mode/keyword_map_enabled are deliberately
    NOT checked here -- once this script sources them directly from the
    baseline file (see main()), there is no separate value left for them
    to disagree with. beam evidence policy is a separate concern, see
    check_beam_evidence(). Raises BaselineMismatchError (all mismatches
    listed at once, not just the first) if anything here no longer
    matches -- refuses to run any case rather than silently produce a
    comparison that isn't actually comparable to the frozen B1 result. A
    baseline can be blocked by ANY single check (including
    baseline_validity alone, even if every codebase/environment check would
    otherwise pass) -- this function never accepts or bypasses a mismatch
    from any check, and adding the baseline_validity check did not change
    how any pre-existing check decides pass/fail."""
    mismatches = [msg for _, check in BASELINE_CODEBASE_CHECKS for ok, msg in [check(baseline)] if not ok]
    if mismatches:
        raise BaselineMismatchError(
            f"{len(mismatches)} mismatch(es) between --baseline-config and the current "
            f"codebase/environment -- refusing to run B2 (would not be comparable to the frozen "
            f"B1 result):\n" + "\n".join(f"  - {m}" for m in mismatches)
        )


# ---------------------------------------------------------------------------
# Beam-provenance gate (task A). A separate policy decision from
# assert_baseline_matches_codebase() above -- this isn't "does the live
# codebase match the file", it's "is the file's own beam claim trustworthy
# enough to run on", which only a human with direct run-history access can
# ultimately resolve for an 'inferred' status.
# ---------------------------------------------------------------------------

def check_beam_evidence(baseline: dict, confirm_inferred_beam: Optional[int] = None) -> tuple:
    """Returns (ok: bool, message: str). ok=True either because
    beam_evidence.status == "confirmed", or because the caller passed
    --confirm-inferred-beam with EXACTLY baseline["beam"]'s value (a wrong
    override value is a hard failure, not silently accepted/ignored)."""
    status = baseline["beam_evidence"]["status"]
    beam = baseline["beam"]

    if status == "confirmed":
        return True, (
            f"beam_evidence.status='confirmed' (source: {baseline['beam_evidence'].get('source')})"
        )

    if confirm_inferred_beam is None:
        return False, (
            f"beam_evidence.status='inferred' for beam={beam} (source: "
            f"{baseline['beam_evidence'].get('source')}; detail: "
            f"{baseline['beam_evidence'].get('detail')}). Refusing to run by default -- pass "
            f"--confirm-inferred-beam {beam} after independently verifying beam={beam} from your "
            f"own run history to proceed."
        )

    if confirm_inferred_beam != beam:
        return False, (
            f"--confirm-inferred-beam {confirm_inferred_beam} does not match the baseline config's "
            f"beam={beam} -- refusing to run. Pass --confirm-inferred-beam {beam} (the exact "
            f"frozen value) if you have verified it, or fix the baseline config if beam={beam} "
            f"is itself wrong."
        )

    return True, (
        f"beam_evidence.status='inferred' for beam={beam}, but --confirm-inferred-beam {beam} "
        f"was passed -- proceeding on human-confirmed override. THIS OVERRIDE IS RECORDED IN "
        f"THE REPORT AND EVERY ROW'S JSONL METADATA."
    )


# ---------------------------------------------------------------------------
# Determinism: dev-row and K execution order, both pure functions of --seed.
# ---------------------------------------------------------------------------

def deterministic_shuffle(items: list, seed: int) -> list:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def deterministic_k_order(k_values: list, seed: int) -> list:
    # seed + 1, not seed, so the K-order permutation is a distinct stream
    # from the dev-row shuffle above rather than moving in lockstep with it.
    rng = random.Random(seed + 1)
    order = list(k_values)
    rng.shuffle(order)
    return order


# ---------------------------------------------------------------------------
# Provenance: real git commit and a real, run_eval-consistent config hash
# per K, replacing placeholder git_commit=""/cfg_hash=f"devsweep_k{k}"
# values from before the baseline-config mechanism existed.
# ---------------------------------------------------------------------------

def resolve_git_commit_full() -> str:
    """Full `git rev-parse HEAD`, best-effort. Returns "" (not a fabricated
    value) if git is unavailable or this isn't a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - best-effort bookkeeping, never fatal to the sweep
        return ""


def compute_k_config_hash(k: int, resolved_reranker_model: str, keyword_map_enabled: bool,
                           beam: int, stage1_mode: str, config_label: str) -> str:
    """The SAME hash run_eval.py itself computes for every B0/B1/B2 run
    (run_eval._config_hash()), applied to this K's actual configuration --
    replaces the placeholder f"devsweep_k{k}" string with a real,
    reproducible hash. branch_collapse is always False here (B2). Two
    different K values legitimately get two different hashes (K is baked
    into the hashed payload, same as every other run_eval.py invocation) --
    this is provenance for what actually ran, not a cross-K equality check.

    sre="on" / use_llm_reranker="on": B2's K-sweep never toggles either
    (those flags postdate this function -- added by Conference I Reviewer
    #2 Section E for ablation support) and always runs with both at their
    true default ("on"), matching run_one_case()'s own default behaviour
    (sre_enabled=True, use_llm_reranker=True) -- this is what every B2
    sweep case actually ran with, not an approximation."""
    fake_args = SimpleNamespace(
        system="hierarchical", beam=beam, stage1_mode=stage1_mode,
        reranker_candidates=k, branch_collapse=False, config=config_label,
        sre="on", use_llm_reranker="on",
    )
    return run_eval._config_hash(fake_args, resolved_reranker_model, keyword_map_enabled)


# ---------------------------------------------------------------------------
# Provenance (task D): environment facts recorded LIVE in every report --
# unlike b1_frozen.json's frozen fields, these describe the CURRENT run and
# would go stale if baked into the static baseline file.
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def check_git_tree_clean() -> tuple:
    """Returns (clean: bool, porcelain_output: str). porcelain_output is ""
    when clean; best-effort -- if git itself is unavailable, returns
    (False, "<reason>") so the caller fails closed rather than silently
    treating an unknown state as clean."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False, f"git status failed: {result.stderr.strip()}"
        output = result.stdout.strip()
        return (output == ""), output
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def resolve_qdrant_version() -> Optional[str]:
    """GET the Qdrant root endpoint for its version string -- service info
    only, not a /collections/*/points/search call, so this is NOT
    'calling Qdrant for retrieval'. Returns None (best-effort) if
    unreachable."""
    try:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = os.getenv("QDRANT_PORT", "6333")
        req = urllib.request.Request(f"http://{host}:{port}/")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read()).get("version")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def resolve_environment_provenance(requirements_path: Optional[Path] = None) -> dict:
    """Best-effort snapshot of the CURRENT execution environment (task D).
    Every field is independently best-effort -- a failure resolving one
    (e.g. no GPU, Qdrant not running) never raises or blocks the others.
    This function does NOT gate execution by itself; check_git_tree_clean()
    is what main() uses to decide whether to fail closed on a dirty tree."""
    git_clean, git_porcelain = check_git_tree_clean()
    requirements_path = requirements_path or (Path(__file__).resolve().parents[1] / "requirements.txt")

    ram_gb = None
    cpu = None
    gpu = None
    if _HAVE_PSUTIL:
        try:
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:  # noqa: BLE001
            pass
    try:
        import platform as _platform
        cpu = _platform.processor() or _platform.machine()
        os_summary = _platform.platform()
    except Exception:  # noqa: BLE001
        os_summary = None
    try:
        gpu_result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '"],
            capture_output=True, text=True, timeout=15,
        )
        gpu = gpu_result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        gpu = None

    return {
        "current_b2_git_commit": resolve_git_commit_full(),
        "git_tree_clean": git_clean,
        "git_tree_dirty_files": git_porcelain,
        "python_version": sys.version.split()[0],
        "requirements_path": str(requirements_path),
        "requirements_sha256": sha256_of_file(requirements_path),
        "ollama_version": resolve_ollama_version(),
        "qdrant_version": resolve_qdrant_version(),
        "os": os_summary,
        "cpu": cpu,
        "ram_gb": ram_gb,
        "gpu": gpu,
    }


@dataclass
class KSweepResult:
    k: int
    n_expected_cases: int
    n_cases: int                        # cases actually completed (post-warmup)
    n_hard_failures: int                # per-case exception, no prediction produced
    incomplete_run: bool                # n_cases < n_expected_cases
    crashed: bool                       # process-level exception escaped the sweep loop
    crash_message: str
    corrupted_output: bool              # raw per-K CSV failed a read-back sanity check
    n_timeouts: int
    n_invalid_outputs: int
    timeout_rate: float
    invalid_output_rate: float
    candidate_recall_at_k: float
    n_successful_cases: int             # n_cases - n_timeouts - n_invalid_outputs
    top1_accuracy_all_case: float       # denom n_cases; timeout/invalid always count as incorrect
    top1_accuracy_conditional: float    # denom n_successful_cases; timeout/invalid cases excluded
    median_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    peak_memory_mb: Optional[float]


@dataclass
class SelectionDecision:
    chosen_k: Optional[int]
    reference_k: int
    rejected: dict = field(default_factory=dict)     # k -> "; "-joined reason string
    rationale: list = field(default_factory=list)     # ordered human-readable steps


# ---------------------------------------------------------------------------
# Pure metric computation -- operates on plain objects/dicts exposing the
# same attributes as run_eval.CaseResult (gold_isco_4digit, pred_isco_4digit,
# gold_rank_in_pool, end_to_end_latency_ms, invalid_output_flag,
# timed_out_flag, error). Unit-testable without any live LLM/Qdrant call.
# ---------------------------------------------------------------------------

def _percentile(values: list, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))
    return s[idx]


def compute_metrics(
    case_results: list,
    k: int,
    n_expected_cases: Optional[int] = None,
    crashed: bool = False,
    crash_message: str = "",
    corrupted_output: bool = False,
    peak_memory_mb: Optional[float] = None,
) -> KSweepResult:
    n_cases = len(case_results)
    n_expected_cases = n_expected_cases if n_expected_cases is not None else n_cases

    if n_cases == 0:
        return KSweepResult(
            k=k, n_expected_cases=n_expected_cases, n_cases=0, n_hard_failures=0,
            incomplete_run=n_expected_cases > 0, crashed=crashed, crash_message=crash_message,
            corrupted_output=corrupted_output, n_timeouts=0, n_invalid_outputs=0,
            timeout_rate=0.0, invalid_output_rate=0.0, candidate_recall_at_k=0.0,
            n_successful_cases=0, top1_accuracy_all_case=0.0, top1_accuracy_conditional=0.0,
            median_latency_ms=0.0, p95_latency_ms=0.0, max_latency_ms=0.0, mean_latency_ms=0.0,
            peak_memory_mb=peak_memory_mb,
        )

    n_hard_failures = sum(
        1 for r in case_results if getattr(r, "error", "") and not getattr(r, "pred_isco_4digit", "")
    )
    n_recall_hits = sum(
        1 for r in case_results
        if getattr(r, "gold_rank_in_pool", None) is not None and r.gold_rank_in_pool <= k
    )
    n_timeouts = sum(1 for r in case_results if getattr(r, "timed_out_flag", False))
    # invalid_output_flag and timed_out_flag are mutually exclusive by construction in
    # run_eval.run_one_case(): timed_out_flag only arises from a reranker_error (kickoff()
    # itself raised); invalid_output_flag only arises from a successful kickoff() whose
    # response could not be parsed. A case cannot be both.
    n_invalid = sum(1 for r in case_results if getattr(r, "invalid_output_flag", False))
    n_unrecovered = n_timeouts + n_invalid
    n_successful = n_cases - n_unrecovered

    n_top1_hits_all_case = sum(
        1 for r in case_results
        if not getattr(r, "timed_out_flag", False) and not getattr(r, "invalid_output_flag", False)
        and getattr(r, "pred_isco_4digit", "") and r.pred_isco_4digit == getattr(r, "gold_isco_4digit", "")
    )
    latencies = [
        r.end_to_end_latency_ms for r in case_results
        if getattr(r, "end_to_end_latency_ms", None) is not None
    ]

    return KSweepResult(
        k=k,
        n_expected_cases=n_expected_cases,
        n_cases=n_cases,
        n_hard_failures=n_hard_failures,
        incomplete_run=n_cases < n_expected_cases,
        crashed=crashed,
        crash_message=crash_message,
        corrupted_output=corrupted_output,
        n_timeouts=n_timeouts,
        n_invalid_outputs=n_invalid,
        timeout_rate=n_timeouts / n_cases,
        invalid_output_rate=n_invalid / n_cases,
        candidate_recall_at_k=n_recall_hits / n_cases,
        n_successful_cases=n_successful,
        top1_accuracy_all_case=n_top1_hits_all_case / n_cases,
        top1_accuracy_conditional=(n_top1_hits_all_case / n_successful) if n_successful else 0.0,
        median_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        max_latency_ms=max(latencies) if latencies else 0.0,
        mean_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        peak_memory_mb=peak_memory_mb,
    )


def check_eligibility(
    candidate: KSweepResult,
    reference: KSweepResult,
    p95_ceiling_factor: float = DEFAULT_P95_CEILING_FACTOR,
    max_additional_timeout_or_invalid: int = DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID,
    memory_budget_mb: Optional[float] = None,
) -> tuple:
    """Returns (eligible: bool, reasons: list[str], latency_factor: Optional[float]).
    reasons is empty iff eligible. latency_factor (candidate P95 / reference P95) is
    always returned for reporting, even when eligible, since the report must show the
    actual factor regardless of pass/fail (see rule #3)."""
    reasons = []

    # 1. Hard failures -- unconditional, no threshold.
    if candidate.crashed:
        reasons.append(f"process crash during sweep: {candidate.crash_message or '(no message captured)'}")
    if candidate.incomplete_run:
        reasons.append(
            f"incomplete run: completed {candidate.n_cases}/{candidate.n_expected_cases} case(s)"
        )
    if candidate.corrupted_output:
        reasons.append("raw output CSV failed its read-back sanity check")
    if candidate.n_hard_failures > 0:
        reasons.append(
            f"{candidate.n_hard_failures} hard case failure(s) (exception during classify(), "
            f"no prediction produced)"
        )

    # 2. Timeout + invalid-output delta vs. reference.
    candidate_unrecovered = candidate.n_timeouts + candidate.n_invalid_outputs
    reference_unrecovered = reference.n_timeouts + reference.n_invalid_outputs
    delta = candidate_unrecovered - reference_unrecovered
    if delta > max_additional_timeout_or_invalid:
        reasons.append(
            f"timeout+invalid_output count {candidate_unrecovered} (timeouts={candidate.n_timeouts}, "
            f"invalid={candidate.n_invalid_outputs}) exceeds reference K={reference.k}'s count "
            f"{reference_unrecovered} by {delta}, more than the allowed "
            f"{max_additional_timeout_or_invalid}"
        )

    # 3. P95 latency vs. reference.
    latency_factor = (
        candidate.p95_latency_ms / reference.p95_latency_ms if reference.p95_latency_ms > 0 else None
    )
    if latency_factor is not None and latency_factor > p95_ceiling_factor:
        reasons.append(
            f"P95 latency {candidate.p95_latency_ms:.1f}ms is {latency_factor:.2f}x reference "
            f"K={reference.k}'s P95 ({reference.p95_latency_ms:.1f}ms), exceeding the ceiling "
            f"{p95_ceiling_factor}x"
        )

    # 4. Memory budget (only checked if a budget was actually provided).
    if memory_budget_mb is not None:
        if candidate.peak_memory_mb is not None and candidate.peak_memory_mb > memory_budget_mb:
            reasons.append(
                f"peak_memory_mb={candidate.peak_memory_mb:.1f} exceeds "
                f"memory_budget_mb={memory_budget_mb}"
            )

    return (len(reasons) == 0, reasons, latency_factor)


def select_k(
    results: list,
    reference_k: int = DEFAULT_REFERENCE_K,
    recall_close_threshold: float = DEFAULT_RECALL_CLOSE_THRESHOLD,
    p95_ceiling_factor: float = DEFAULT_P95_CEILING_FACTOR,
    max_additional_timeout_or_invalid: int = DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID,
    memory_budget_mb: Optional[float] = None,
) -> SelectionDecision:
    """Pure selection logic over a list of KSweepResult, implementing the B2
    operational eligibility rule (module docstring). Deterministic given the
    same inputs and thresholds -- never looks at anything beyond the
    KSweepResult list passed in (in particular, never at
    eval/test_set_full130.csv)."""
    decision = SelectionDecision(chosen_k=None, reference_k=reference_k)
    if not results:
        decision.rationale.append("No K results supplied -- nothing to select.")
        return decision

    by_k = {r.k: r for r in results}
    if reference_k not in by_k:
        decision.rationale.append(
            f"Reference K={reference_k} is not among the supplied results "
            f"({sorted(by_k.keys())}) -- cannot apply the eligibility rule "
            f"without a reference. Include --reference-k in --k-values."
        )
        return decision
    reference = by_k[reference_k]

    if memory_budget_mb is None:
        decision.rationale.append(
            "No --memory-budget-mb was set -- the memory-budget eligibility check "
            "was NOT applied to any K (peak_memory_mb is still recorded for every K)."
        )
    if not _HAVE_PSUTIL:
        decision.rationale.append(
            "psutil is not installed -- peak_memory_mb could not be measured for any "
            "K (recorded as None); the memory-budget check could not be evaluated "
            "even if --memory-budget-mb was set."
        )

    eligible = []
    for r in results:
        ok, reasons, latency_factor = check_eligibility(
            r, reference, p95_ceiling_factor, max_additional_timeout_or_invalid, memory_budget_mb,
        )
        factor_note = f" (P95 latency factor vs. reference: {latency_factor:.2f}x)" if latency_factor is not None else ""
        if ok:
            eligible.append(r)
            decision.rationale.append(f"K={r.k}: eligible.{factor_note}")
        else:
            decision.rejected[r.k] = "; ".join(reasons)
            decision.rationale.append(f"K={r.k}: REJECTED -- {'; '.join(reasons)}{factor_note}")

    if not eligible:
        decision.rationale.append(
            "All K values were rejected by the operational eligibility rule -- no K "
            "can be selected from this sweep. Investigate the rejection reasons above "
            "(and re-run the reference K itself if it also failed) rather than relaxing "
            "thresholds silently."
        )
        return decision

    max_recall = max(r.candidate_recall_at_k for r in eligible)
    decision.rationale.append(
        f"Step 5a: max Candidate Recall@K among eligible K values = {max_recall:.4f}."
    )

    close = [r for r in eligible if (max_recall - r.candidate_recall_at_k) <= recall_close_threshold]
    decision.rationale.append(
        f"Step 5b: K values within recall_close_threshold={recall_close_threshold} of the max: "
        f"{sorted(r.k for r in close)}."
    )

    max_top1 = max(r.top1_accuracy_all_case for r in close)
    top1_tied = [r for r in close if abs(r.top1_accuracy_all_case - max_top1) < 1e-9]
    decision.rationale.append(
        f"Step 5c: among those, max all-case Top-1 accuracy = {max_top1:.4f}, achieved by K="
        f"{sorted(r.k for r in top1_tied)}."
    )

    chosen = min(top1_tied, key=lambda r: r.k)
    decision.rationale.append(f"Step 5d: smallest-K tie-break -> chosen K={chosen.k}.")
    decision.chosen_k = chosen.k
    return decision


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def render_markdown_report(
    decision: SelectionDecision,
    results: list,
    dev_set_path: str,
    reranker_model: str,
    n_dev_cases: int,
    generated_at: str,
    p95_ceiling_factor: float = DEFAULT_P95_CEILING_FACTOR,
    max_additional_timeout_or_invalid: int = DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID,
    memory_budget_mb: Optional[float] = None,
    baseline_config_path: str = "",
    baseline: Optional[dict] = None,
    git_commit: str = "",
    seed: Optional[int] = None,
    dev_row_order: Optional[list] = None,
    k_execution_order: Optional[list] = None,
    beam_evidence_result: Optional[tuple] = None,
    confirm_inferred_beam: Optional[int] = None,
    environment: Optional[dict] = None,
) -> str:
    baseline = baseline or {}
    fingerprint = baseline.get("implementation_fingerprint", {})
    identity = baseline.get("ollama_model_identity", {})
    provenance = baseline.get("provenance", {})
    environment = environment or {}
    beam_ev = baseline.get("beam_evidence", {})
    beam_evidence_ok, beam_evidence_msg = beam_evidence_result if beam_evidence_result else (None, "")

    lines = [
        "# B2 dev-set K-sweep report -- operational eligibility rule",
        "",
        f"- Dev set: `{dev_set_path}` ({n_dev_cases} cases)",
        f"- Baseline config: `{baseline_config_path}`",
        f"- Reranker model: `{reranker_model}` (sourced from baseline config)",
        f"- Beam: {baseline.get('beam', 'n/a')} (sourced from baseline config)",
        f"- Stage-1 mode: {baseline.get('stage1_mode', 'n/a')} (sourced from baseline config)",
        f"- Keyword map enabled: {baseline.get('keyword_map_enabled', 'n/a')} (sourced from baseline config)",
        f"- Pooling: B1 (branch_collapse=False, asserted against baseline config)",
        f"- LLM temperature: {baseline.get('llm_temperature', 'n/a')} (asserted against hardcoded value)",
        f"- Reranker timeout (s): {baseline.get('timeout_s', 'n/a')} (asserted against live constant)",
        f"- Baseline source run: `{baseline.get('_source_csv', 'n/a')}` "
        f"({baseline.get('_source_top1_accuracy', 'n/a')} top-1)",
        f"- git commit (this sweep): `{git_commit or '(unresolved)'}`",
        f"- Seed: {seed}",
        f"- Reference K: {decision.reference_k}",
        f"- P95 latency ceiling: {p95_ceiling_factor}x reference",
        f"- Max additional timeout+invalid_output events vs. reference: {max_additional_timeout_or_invalid}",
        f"- Memory budget: {memory_budget_mb if memory_budget_mb is not None else '(not set -- check skipped)'} MB",
        f"- Generated: {generated_at}",
        "",
        "## Beam-provenance gate",
        "",
        f"- beam_evidence.status (baseline config): `{beam_ev.get('status', 'n/a')}`",
        f"- beam_evidence.source: {beam_ev.get('source', 'n/a')}",
        f"- Gate result: {'PASS' if beam_evidence_ok else 'FAIL' if beam_evidence_ok is False else 'not evaluated'}",
        f"- Gate message: {beam_evidence_msg or 'n/a'}",
        f"- --confirm-inferred-beam override passed: "
        f"{confirm_inferred_beam if confirm_inferred_beam is not None else '(not passed)'}"
        + (" **-- HUMAN OVERRIDE IN EFFECT FOR THIS RUN**" if confirm_inferred_beam is not None else ""),
        "",
        "## Ollama model identity (task B)",
        "",
        f"- Baseline tag: `{identity.get('tag', 'n/a')}`",
        f"- Baseline digest: `{identity.get('digest', 'n/a')}`",
        f"- Baseline parameter_size / quantization: "
        f"{identity.get('parameter_size', 'n/a')} / {identity.get('quantization_level', 'n/a')}",
        f"- Live Ollama version (this run): {environment.get('ollama_version', 'n/a')}",
        "",
        "## Implementation fingerprint (task C)",
        "",
        f"- Composite SHA-256: `{fingerprint.get('composite_sha256', 'n/a')}`",
        "- Components (sorted):",
    ] + [
        f"  - `{name}`: `{h}`" for name, h in sorted(fingerprint.get("components", {}).items())
    ] + [
        "",
        "## Provenance (task D)",
        "",
        f"- b1_result_csv: `{provenance.get('b1_result_csv', 'n/a')}`",
        f"- b1_result_csv_sha256: `{provenance.get('b1_result_csv_sha256', 'n/a')}`",
        f"- b1_baseline_git_commit: {provenance.get('b1_baseline_git_commit') or '(unrecoverable -- see b1_frozen.json _notes)'}",
        f"- current_b2_git_commit: `{environment.get('current_b2_git_commit', 'n/a')}`",
        f"- git tree clean: {environment.get('git_tree_clean', 'n/a')}"
        + (f" (dirty files:\n```\n{environment.get('git_tree_dirty_files')}\n```)"
           if environment.get("git_tree_dirty_files") else ""),
        f"- python_version: {environment.get('python_version', 'n/a')}",
        f"- requirements_sha256: `{environment.get('requirements_sha256', 'n/a')}` "
        f"({environment.get('requirements_path', 'n/a')})",
        f"- qdrant_version: {environment.get('qdrant_version', 'n/a')}",
        f"- os: {environment.get('os', 'n/a')}",
        f"- cpu: {environment.get('cpu', 'n/a')}",
        f"- ram_gb: {environment.get('ram_gb', 'n/a')}",
        f"- gpu: {environment.get('gpu', 'n/a')}",
        "",
        "## Determinism",
        "",
        f"- K execution order (actual, seed-derived, not ascending): "
        f"{k_execution_order if k_execution_order is not None else 'n/a'}",
        f"- Dev-row execution order (seed-derived shuffle, same order used for every K): "
        f"{', '.join(dev_row_order) if dev_row_order else 'n/a'}",
        "",
        "## Per-K metrics",
        "",
        "| K | n | hard_fail | incomplete | crashed | corrupted | timeouts | invalid | "
        "Recall@K | Top-1 (all-case) | Top-1 (conditional) | median (ms) | P95 (ms) | max (ms) | peak mem (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r.k):
        mem = f"{r.peak_memory_mb:.1f}" if r.peak_memory_mb is not None else "n/a"
        lines.append(
            f"| {r.k} | {r.n_cases}/{r.n_expected_cases} | {r.n_hard_failures} | "
            f"{'yes' if r.incomplete_run else 'no'} | {'yes' if r.crashed else 'no'} | "
            f"{'yes' if r.corrupted_output else 'no'} | {r.n_timeouts} ({r.timeout_rate:.3f}) | "
            f"{r.n_invalid_outputs} ({r.invalid_output_rate:.3f}) | {r.candidate_recall_at_k:.4f} | "
            f"{r.top1_accuracy_all_case:.4f} | {r.top1_accuracy_conditional:.4f} | "
            f"{r.median_latency_ms:.1f} | {r.p95_latency_ms:.1f} | {r.max_latency_ms:.1f} | {mem} |"
        )

    lines += ["", "## Eligibility and selection rationale", ""]
    for step in decision.rationale:
        lines.append(f"- {step}")

    lines += ["", "## Decision", ""]
    if decision.chosen_k is not None:
        lines.append(
            f"**Chosen K = {decision.chosen_k}**, frozen for the single B2 "
            f"confirmation run on `eval/test_set_full130.csv`. Per the final-test "
            f"rule, K was selected without looking at that set, and that set will "
            f"be run exactly once with this K."
        )
    else:
        lines.append(
            "**No K selected.** See the rejection reasons above -- every candidate "
            "K failed the operational eligibility rule, the reference K was missing "
            "from the sweep, or no results were supplied."
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI: actually runs the sweep (NOT executed by this session -- see the B2
# spec's explicit "do not run the K-sweep... yet" instruction; infra only).
# ---------------------------------------------------------------------------

def _load_dev_rows_as_test_set(dev_set_path: Path) -> list[dict]:
    """Translate dev_set_v1.csv's schema (eval/dev_set_schema.md) into the
    column names run_eval.run_one_case() expects (input_text,
    input_language, gold_isco_4digit) -- kept as an explicit, visible
    mapping rather than silently aliasing column names, since the two
    schemas exist for different audiences (human annotators vs. the
    harness) and should stay allowed to diverge further later."""
    with open(dev_set_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "case_id": row["case_id"],
            "input_text": row["respondent_text"],
            "input_language": row["language"],
            "gold_isco_4digit": row["gold_isco_code"],
        }
        for row in rows
    ]


def _peak_rss_mb() -> Optional[float]:
    if not _HAVE_PSUTIL:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 - best-effort sampling, never fatal to the sweep
        return None


def run_sweep_for_k(dev_rows: list[dict], k: int, reranker_model: str, beam: int,
                     stage1_mode: str, disable_keyword_map: bool, git_commit: str, seed: int,
                     warmup_cases: int = DEFAULT_WARMUP_CASES) -> tuple:
    """Runs the dev set once through B1's pooling pipeline with
    --reranker-candidates=k. branch_collapse is hardcoded False -- see
    module docstring's experiment-separation constraints. dev_rows is
    expected to already be in its final execution order (shuffled by the
    caller via deterministic_shuffle()) -- this function does not reorder it.

    Returns (case_results, crashed, crash_message, peak_memory_mb). Catches
    exceptions at the sweep level (not just per-case, which run_one_case()
    already handles) so a crash on one K doesn't take down the whole sweep
    -- the crash is recorded and that K is rejected by check_eligibility()
    rather than losing every other K's results too."""
    crashed = False
    crash_message = ""
    peak_mb = _peak_rss_mb()
    case_results = []

    try:
        clf = run_eval.build_system(
            "hierarchical", reranker_model=reranker_model,
            disable_keyword_map=disable_keyword_map, beam=beam, stage1_mode=stage1_mode,
            reranker_candidates=k, branch_collapse=False, capture_pool_metadata=True,
        )
        resolved_reranker_model = getattr(clf, "reranker_model_resolved", reranker_model)
        keyword_map_enabled = not disable_keyword_map
        isic_clf = run_eval.ISICClassifier()
        isced_clf = run_eval.ISCEDClassifier()
        sre = run_eval.SemanticRelationEngine(use_llm=False)
        cfg_hash = compute_k_config_hash(
            k, resolved_reranker_model, keyword_map_enabled, beam, stage1_mode, f"devsweep_k{k}",
        )

        # Warm-up phase: discarded entirely, never enters case_results, so it
        # cannot affect recall/top1/latency stats -- see rule #3.
        for i in range(min(warmup_cases, len(dev_rows))):
            row = dev_rows[i]
            run_eval.run_one_case(
                clf, sre, isic_clf, isced_clf, row_index=i,
                case_id=f"warmup_{row['case_id']}", input_text=row["input_text"],
                input_language=row["input_language"], gold_isco_4digit=row["gold_isco_4digit"],
                gold_isic="", gold_isced="", config_hash=cfg_hash, system="hierarchical",
                keyword_map_enabled=keyword_map_enabled, branch_collapse_enabled=False,
                capture_pool_metadata_enabled=True, run_id=f"devsweep_k{k}_warmup",
                git_commit=git_commit, seed=seed,
            )
            sample = _peak_rss_mb()
            if sample is not None:
                peak_mb = sample if peak_mb is None else max(peak_mb, sample)

        for i, row in enumerate(dev_rows):
            r = run_eval.run_one_case(
                clf, sre, isic_clf, isced_clf, row_index=i,
                case_id=row["case_id"], input_text=row["input_text"],
                input_language=row["input_language"], gold_isco_4digit=row["gold_isco_4digit"],
                gold_isic="", gold_isced="", config_hash=cfg_hash, system="hierarchical",
                keyword_map_enabled=keyword_map_enabled, branch_collapse_enabled=False,
                capture_pool_metadata_enabled=True, run_id=f"devsweep_k{k}",
                git_commit=git_commit, seed=seed, input_order_position=i,
            )
            case_results.append(r)
            sample = _peak_rss_mb()
            if sample is not None:
                peak_mb = sample if peak_mb is None else max(peak_mb, sample)
    except Exception as exc:  # noqa: BLE001 - see docstring: a crash on one K must not kill the sweep
        crashed = True
        crash_message = f"{type(exc).__name__}: {exc}"

    return case_results, crashed, crash_message, peak_mb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-set", required=True, type=Path)
    parser.add_argument(
        "--baseline-config", required=True, type=Path,
        help=(
            "Mandatory. JSON file (see eval/configs/b1_frozen.json) recording the exact "
            "frozen B1 configuration -- reranker_model/beam/stage1_mode/keyword_map_enabled "
            "are read directly from this file (not separately CLI-configurable in this "
            "script), and branch_collapse/llm_temperature/timeout_s/implementation_fingerprint/"
            "ollama_model_identity are asserted against it before any case runs. See the module "
            "docstring's 'Baseline config' section."
        ),
    )
    parser.add_argument("--k-values", type=str, default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--reference-k", type=int, default=DEFAULT_REFERENCE_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SWEEP_SEED,
                         help="Drives the deterministic dev-row shuffle and K execution order "
                              "(both recorded in the report). See module docstring.")
    parser.add_argument("--warmup-cases", type=int, default=DEFAULT_WARMUP_CASES)
    parser.add_argument("--recall-close-threshold", type=float, default=DEFAULT_RECALL_CLOSE_THRESHOLD)
    parser.add_argument("--p95-ceiling-factor", type=float, default=DEFAULT_P95_CEILING_FACTOR)
    parser.add_argument("--max-additional-timeout-or-invalid", type=int,
                         default=DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID)
    parser.add_argument("--memory-budget-mb", type=float, default=None,
                         help="If omitted, the memory-budget eligibility check is skipped "
                              "(peak_memory_mb is still recorded for every K).")
    parser.add_argument(
        "--confirm-inferred-beam", type=int, default=None,
        help=(
            "Required to proceed if the baseline config's beam_evidence.status is 'inferred' "
            "(see task A / eval/configs/b1_frozen.json's beam_evidence field). Must equal the "
            "baseline's beam value EXACTLY -- pass this only after independently verifying the "
            "beam width from your own run history. Recorded prominently in the report and "
            "metadata JSONL."
        ),
    )
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help=(
            "By default, a dirty git working tree (uncommitted changes) fails closed before "
            "any case runs -- a B2 run's code provenance must be traceable to a specific commit. "
            "Pass this to override; the dirty-tree files are still recorded prominently in the "
            "report."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if not args.dev_set.exists():
        parser.error(f"Dev set not found: {args.dev_set}")
    if not args.baseline_config.exists():
        parser.error(f"--baseline-config not found: {args.baseline_config}")
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    if not k_values:
        parser.error("--k-values produced an empty list")
    if args.reference_k not in k_values:
        parser.error(
            f"--reference-k={args.reference_k} must be included in --k-values={k_values} "
            f"-- the eligibility rule compares every K against this reference run."
        )
    if not _HAVE_PSUTIL:
        print("WARNING: psutil is not installed -- peak_memory_mb will be None for every K "
              "and the memory-budget eligibility check cannot be evaluated.", file=sys.stderr)

    baseline = load_baseline_config(args.baseline_config)
    shape_errors = validate_baseline_shape(baseline)
    if shape_errors:
        parser.error(
            f"--baseline-config {args.baseline_config} is malformed:\n" +
            "\n".join(f"  - {e}" for e in shape_errors)
        )
    try:
        assert_baseline_matches_codebase(baseline)
    except BaselineMismatchError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Baseline config OK: {args.baseline_config} matches the current codebase "
          f"(branch_collapse=False, llm_temperature, timeout_s, implementation_fingerprint, "
          f"and ollama_model_identity all verified).")

    beam_evidence_ok, beam_evidence_msg = check_beam_evidence(baseline, args.confirm_inferred_beam)
    print(f"Beam-provenance gate: {'PASS' if beam_evidence_ok else 'FAIL'} -- {beam_evidence_msg}")
    if not beam_evidence_ok:
        print(f"FATAL: {beam_evidence_msg}", file=sys.stderr)
        sys.exit(1)
    if args.confirm_inferred_beam is not None:
        print(f"*** HUMAN OVERRIDE IN EFFECT: --confirm-inferred-beam {args.confirm_inferred_beam} ***")

    git_clean, git_dirty_files = check_git_tree_clean()
    if not git_clean and not args.allow_dirty_tree:
        print(
            f"FATAL: working tree is dirty -- refusing to run (code provenance for this B2 run "
            f"would not be traceable to a single commit). Files:\n{git_dirty_files}\n"
            f"Commit/stash your changes, or pass --allow-dirty-tree to override (recorded "
            f"prominently in the report).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not git_clean and args.allow_dirty_tree:
        print(f"*** WARNING: --allow-dirty-tree override in effect. Dirty files:\n{git_dirty_files} ***")

    reranker_model = baseline["reranker_model"]
    beam = baseline["beam"]
    stage1_mode = baseline["stage1_mode"]
    disable_keyword_map = not baseline["keyword_map_enabled"]

    git_commit = resolve_git_commit_full()
    if not git_commit:
        print("WARNING: could not resolve `git rev-parse HEAD` -- git_commit will be blank "
              "in every row and in the report.", file=sys.stderr)

    dev_rows_loaded = _load_dev_rows_as_test_set(args.dev_set)
    dev_rows = deterministic_shuffle(dev_rows_loaded, args.seed)
    k_order = deterministic_k_order(k_values, args.seed)
    dev_row_order = [r["case_id"] for r in dev_rows]
    print(f"Loaded {len(dev_rows)} dev case(s) from {args.dev_set}")
    print(f"seed={args.seed}  dev-row execution order (seed-derived shuffle): {dev_row_order}")
    print(f"K execution order (seed-derived shuffle, not ascending): {k_order}  (reference K={args.reference_k})")
    print(f"reranker_model={reranker_model} beam={beam} stage1_mode={stage1_mode} "
          f"keyword_map_enabled={baseline['keyword_map_enabled']}  (all sourced from --baseline-config)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_metrics = []
    for k in k_order:
        print(f"\n=== K={k} ===")
        case_results, crashed, crash_message, peak_mb = run_sweep_for_k(
            dev_rows, k, reranker_model, beam, stage1_mode, disable_keyword_map,
            git_commit, args.seed, warmup_cases=args.warmup_cases,
        )
        if crashed:
            print(f"K={k}: CRASHED -- {crash_message}")

        raw_csv_path = args.output_dir / f"{timestamp}_devsweep_k{k}_raw.csv"
        corrupted_output = False
        fieldnames = list(run_eval.CaseResult.__dataclass_fields__.keys())
        try:
            with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in case_results:
                    writer.writerow(r.__dict__)
            with open(raw_csv_path, newline="", encoding="utf-8") as f:
                written_rows = list(csv.DictReader(f))
            if len(written_rows) != len(case_results):
                corrupted_output = True
        except Exception as exc:  # noqa: BLE001
            corrupted_output = True
            print(f"K={k}: output write/read-back failed: {exc}")

        metrics = compute_metrics(
            case_results, k, n_expected_cases=len(dev_rows), crashed=crashed,
            crash_message=crash_message, corrupted_output=corrupted_output, peak_memory_mb=peak_mb,
        )
        all_metrics.append(metrics)
        print(
            f"K={k}: recall@K={metrics.candidate_recall_at_k:.4f} "
            f"top1_all_case={metrics.top1_accuracy_all_case:.4f} "
            f"top1_conditional={metrics.top1_accuracy_conditional:.4f} "
            f"timeouts={metrics.n_timeouts} invalid={metrics.n_invalid_outputs} "
            f"P95={metrics.p95_latency_ms:.1f}ms peak_mem={metrics.peak_memory_mb}"
        )

    decision = select_k(
        all_metrics,
        reference_k=args.reference_k,
        recall_close_threshold=args.recall_close_threshold,
        p95_ceiling_factor=args.p95_ceiling_factor,
        max_additional_timeout_or_invalid=args.max_additional_timeout_or_invalid,
        memory_budget_mb=args.memory_budget_mb,
    )

    environment = resolve_environment_provenance()

    report_md = render_markdown_report(
        decision, all_metrics, str(args.dev_set), reranker_model,
        len(dev_rows), datetime.now(timezone.utc).isoformat(),
        p95_ceiling_factor=args.p95_ceiling_factor,
        max_additional_timeout_or_invalid=args.max_additional_timeout_or_invalid,
        memory_budget_mb=args.memory_budget_mb,
        baseline_config_path=str(args.baseline_config),
        baseline=baseline,
        git_commit=git_commit,
        seed=args.seed,
        dev_row_order=dev_row_order,
        k_execution_order=k_order,
        beam_evidence_result=(beam_evidence_ok, beam_evidence_msg),
        confirm_inferred_beam=args.confirm_inferred_beam,
        environment=environment,
    )
    report_path = args.output_dir / f"{timestamp}_devsweep_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    metadata_path = args.output_dir / f"{timestamp}_devsweep_metadata.jsonl"
    metadata_record = {
        "timestamp": timestamp,
        "dev_set": str(args.dev_set),
        "baseline_config": str(args.baseline_config),
        "baseline": baseline,
        "seed": args.seed,
        "dev_row_order": dev_row_order,
        "k_execution_order": k_order,
        "reference_k": args.reference_k,
        "beam_evidence_status": baseline["beam_evidence"]["status"],
        "confirm_inferred_beam_passed": args.confirm_inferred_beam,
        "beam_override_used": args.confirm_inferred_beam is not None,
        "allow_dirty_tree_used": (not git_clean) and args.allow_dirty_tree,
        "environment": environment,
        "chosen_k": decision.chosen_k,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")

    print(f"\nWrote report to {report_path}")
    print(f"Wrote metadata JSONL to {metadata_path}")
    print(f"Chosen K: {decision.chosen_k}")


if __name__ == "__main__":
    main()
