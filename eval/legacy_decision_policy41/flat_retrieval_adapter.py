"""
eval/legacy_decision_policy41/flat_retrieval_adapter.py

Task 42: wires Task 41's historical decision-policy compatibility
component (`policy.py`, unmodified) to the current maintained flat
retrieval caller, and runs it -- read-only, zero LLM calls -- against the
WISCO v2 DEV split only.

This module never constructs a CrewAI agent, never imports an LLM/
provider SDK, and never invokes a reranker itself: `run_dev_preflight`
supplies `reranker=None` to `policy.classify_with_policy()` and
translates the resulting `ValueError` (Task 41's existing, correct
fail-closed behavior for a *required* reranker) into a
`PENDING_RERANK_NO_LLM_CALLED` row outcome. It calls
`HierarchicalISCOStore._embed_query()`/`._query()` directly (not
`search_flat_only()`/`_flat_search()`) so it can read all 5 raw hits --
`_flat_search()` only exposes the top 3 via `top_candidates` -- without
modifying `hierarchical_store.py` in any way.

Known gap this module works around, not fixes: `_flat_search()` reads
`payload.get("label_en", "")`/`payload.get("label_ar", "")`, but the
official flat collection's payload
(`backend/rag/build_official_isco08_collections.py::_build_payload()`)
only ever writes `title_en` (no `label_en`/`label_ar` key at all). This
module reads `title_en` directly and correctly instead.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from eval.legacy_decision_policy41.policy import (
    METHOD_SEMANTIC,
    PolicyCandidate,
    classify_with_policy,
)

REQUIRED_CANDIDATE_COUNT = 5
OUTCOME_SEMANTIC = "semantic"
OUTCOME_PENDING_RERANK = "PENDING_RERANK_NO_LLM_CALLED"


class InsufficientCandidatesError(Exception):
    """Raised when the flat collection returns fewer than
    REQUIRED_CANDIDATE_COUNT hits for a query -- a signal worth
    surfacing (fail closed), never silently padded or tolerated."""


@dataclass
class DevRow:
    case_id: str
    input_text: str
    input_language: str
    gold_isco_4digit: str


@dataclass
class DevRowResult:
    case_id: str
    top1_code: str
    top1_confidence: float
    outcome: str  # OUTCOME_SEMANTIC | OUTCOME_PENDING_RERANK


@dataclass
class DevPreflightReport:
    rows: list[DevRowResult] = field(default_factory=list)
    n_total: int = 0
    n_semantic_fast_path: int = 0
    n_pending_rerank: int = 0
    n_semantic_fast_path_exact_match: int = 0
    run_manifest: dict = field(default_factory=dict)


def fetch_five_official_flat_candidates(store, query_text: str) -> list[PolicyCandidate]:
    """
    Queries *store*'s profile-resolved flat collection directly (via the
    store's own `_embed_query`/`_query` methods -- no new embedding or
    retrieval logic) and returns exactly 5 `PolicyCandidate`s, in the
    collection's own ranked order.

    `title_ar` and `description` are always `""` -- the official
    catalogue has neither field, and this function never fabricates,
    translates, or copies `title_en` into them. `level` is always `4`
    (the flat collection is 4-digit unit-groups only, by construction --
    see FLAT_BASELINE_COVERAGE_AUDIT.md).
    """
    query_vec = store._embed_query(query_text)
    hits = store._query(collection=store._col_flat, query_vec=query_vec, limit=REQUIRED_CANDIDATE_COUNT)
    if len(hits) < REQUIRED_CANDIDATE_COUNT:
        raise InsufficientCandidatesError(
            f"expected {REQUIRED_CANDIDATE_COUNT} candidates from flat collection "
            f"{store._col_flat!r}, got {len(hits)} for query {query_text!r}"
        )
    return [
        PolicyCandidate(
            code=hit.payload.get("code", ""),
            title_en=hit.payload.get("title_en", ""),
            title_ar="",
            level=4,
            confidence=float(hit.score),
            description="",
        )
        for hit in hits[:REQUIRED_CANDIDATE_COUNT]
    ]


def _git_commit_at(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, check=True,
    )
    return result.stdout.decode("utf-8").strip()


def build_run_manifest(store, dev_csv_path: Path, repo_root: Path) -> dict:
    from backend.rag.hierarchical_store import MODEL_NAME

    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_at(repo_root),
        "embedding_model": MODEL_NAME,
        "collection": store._col_flat,
        "profile": store.profile,
        "dev_csv_path": str(dev_csv_path),
        "dev_csv_sha256": hashlib.sha256(dev_csv_path.read_bytes()).hexdigest(),
    }


def run_dev_preflight(
    store,
    dev_rows: list[DevRow],
    limit: Optional[int] = None,
    run_manifest: Optional[dict] = None,
) -> DevPreflightReport:
    """
    Runs the historical decision policy (Task 41's `policy.py`,
    unmodified) against *dev_rows* using real candidates fetched from
    *store*. Never calls a reranker -- `reranker=None` is passed to
    `classify_with_policy` for every row; a row whose top candidate is
    below the 0.92 threshold gets outcome `OUTCOME_PENDING_RERANK`
    instead of an LLM call.

    Any exception other than the expected "reranker is required" one
    stops the run immediately (fail closed) -- it is never caught,
    logged, and skipped so the loop can continue, since a partial run
    would make the aggregate counts untrustworthy.
    """
    rows = dev_rows[:limit] if limit is not None else dev_rows
    report = DevPreflightReport(run_manifest=run_manifest or {})

    for row in rows:
        candidates = fetch_five_official_flat_candidates(store, row.input_text)
        try:
            result = classify_with_policy(
                job_title=row.input_text,
                candidates=candidates,
                lang=row.input_language,
                reranker=None,
            )
        except ValueError as exc:
            if "reranker callable is required" not in str(exc):
                raise
            top = candidates[0]
            row_result = DevRowResult(
                case_id=row.case_id, top1_code=top.code,
                top1_confidence=top.confidence, outcome=OUTCOME_PENDING_RERANK,
            )
            report.n_pending_rerank += 1
        else:
            assert result.method == METHOD_SEMANTIC  # reranker=None guarantees this on the non-raising path
            row_result = DevRowResult(
                case_id=row.case_id, top1_code=result.primary.code,
                top1_confidence=result.primary.confidence, outcome=OUTCOME_SEMANTIC,
            )
            report.n_semantic_fast_path += 1
            if result.primary.code == row.gold_isco_4digit:
                report.n_semantic_fast_path_exact_match += 1

        report.rows.append(row_result)
        report.n_total += 1

    return report


def load_dev_rows(csv_path: Path) -> list[DevRow]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            DevRow(
                case_id=r["case_id"], input_text=r["input_text"],
                input_language=r["input_language"], gold_isco_4digit=r["gold_isco_4digit"],
            )
            for r in reader
        ]
