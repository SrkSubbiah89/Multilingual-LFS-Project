"""Read-only ISCO retrieval with lexical/dense fusion and optional soft paths.

This is an opt-in retrieval alternative, not a replacement for the published
flat or parent-filtered hierarchical baselines. Lexical text comes exclusively
from the caller's authoritative catalogue records, never labelled queries.
Weighted reciprocal-rank fusion uses ``weight / (k + rank)`` with one-based
ranks and the original RRF default k=60. Fusion/path scores are ranking signals,
not calibrated probabilities or automatic human-review decisions.

``rank`` accepts cached dense scores for CPU/batch evaluation. ``search`` uses
an injected Qdrant client and query embedder without loading another model,
creating collections, or writing points. Ancestors supply soft support to a
global leaf pool; a wrong major-group match can never remove a unit candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence
import unicodedata

from backend.agents.classifier_methods import (
    ISCO_HYBRID_RRF,
    ISCO_HYBRID_SOFT_HIERARCHY_RRF,
    ISCO_HYBRID_SPARSE_FALLBACK,
    ISCO_HYBRID_UNAVAILABLE,
)
from backend.rag.official_isco08_catalogue import (
    DEFAULT_METADATA_PATH,
    ENRICHED_E5LARGE_PROFILE,
    ENRICHED_PROFILE,
    OfficialCatalogueRecord,
    PROFILE_COLLECTION_NAMES,
    embedding_config_for_profile,
    load_enriched_catalogue,
    load_official_catalogue,
)

_LEVELS = ("major", "submajor", "minor", "unit")
_ANCESTORS = _LEVELS[:-1]


def unicode_tokens(text: str) -> list[str]:
    """Keep Unicode letters/numbers and their combining marks, including Hindi.

    NFKC/casefold handles presentation forms and case without translating or
    rewriting the respondent's words. No English-only stemming or stop list is
    imposed on multilingual inputs.
    """
    words: list[str] = []
    current: list[str] = []
    for char in unicodedata.normalize("NFKC", text).casefold():
        category = unicodedata.category(char)[0]
        if category in ("L", "N") or (category == "M" and current):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


class _BM25Field:
    """Small inverted BM25 index; no model, network, or optional dependency."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.count = len(documents)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, document in enumerate(documents):
            terms = unicode_tokens(document)
            self.lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / self.count if self.count else 0.0

    def scores(self, query_terms: Sequence[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        if not self.average_length:
            return scores
        for term in sorted(set(query_terms)):
            postings = self.postings.get(term, ())
            if not postings:
                continue
            idf = math.log1p((self.count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / self.average_length
                )
                scores[index] += idf * frequency * (self.k1 + 1.0) / denominator
        return scores


@dataclass
class HybridCandidate:
    code: str
    label_en: str
    label_ar: str
    dense_score: Optional[float]
    lexical_score: float
    fusion_score: float
    dense_rank: Optional[int]
    lexical_rank: Optional[int]
    hierarchy_support: float
    hierarchy_path: list[str] = field(default_factory=list)


@dataclass
class HybridResult:
    code: str
    label_en: str
    label_ar: str
    ranking_score: float
    top_candidates: list[HybridCandidate]
    method: str
    requested_method: str
    profile: str
    source_catalogue_sha256: str
    hierarchy_path: list[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


def _positive_integer(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _finite_weight(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


class HybridISCORetriever:
    """Fusion over one explicit official ISCO catalogue/profile.

    ``records`` should come from the existing official catalogue loaders; the
    ``from_catalogue_file`` convenience constructor performs their validation.
    All records must carry the same profile and source digest. Title-only and
    definition-enriched catalogues remain distinct inputs/profiles.

    ``embed_query`` receives the raw query and must apply its model's query
    preprocessing (for E5, the ``query: `` prefix and normalization). Providing
    a precomputed vector to ``search`` or score lists to ``rank`` avoids calling
    it. The injected client owns its actual connection/read timeout; the
    request timeout supplied here is also passed to Qdrant operations.
    """

    def __init__(
        self,
        records: Sequence[OfficialCatalogueRecord],
        *,
        profile: str,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        hierarchy_weight: float = 0.2,
        title_weight: float = 2.0,
        body_weight: float = 1.0,
        rrf_k: int = 60,
        candidate_k: int = 50,
        client=None,
        embed_query: Optional[Callable[[str], Sequence[float]]] = None,
        query_timeout_seconds: float = 10,
    ):
        if profile not in PROFILE_COLLECTION_NAMES:
            raise ValueError(f"Unknown official ISCO catalogue profile: {profile!r}")
        _positive_integer("rrf_k", rrf_k, 10000)
        _positive_integer("candidate_k", candidate_k, 436)
        self.lexical_weight = _finite_weight("lexical_weight", lexical_weight)
        self.dense_weight = _finite_weight("dense_weight", dense_weight)
        self.hierarchy_weight = _finite_weight("hierarchy_weight", hierarchy_weight)
        self.title_weight = _finite_weight("title_weight", title_weight)
        self.body_weight = _finite_weight("body_weight", body_weight)
        if not self.lexical_weight + self.dense_weight:
            raise ValueError("At least one of lexical_weight/dense_weight must be positive")
        if self.lexical_weight and not self.title_weight + self.body_weight:
            raise ValueError("Positive lexical_weight requires a positive lexical field weight")
        timeout = _finite_weight("query_timeout_seconds", query_timeout_seconds)
        if not timeout:
            raise ValueError("query_timeout_seconds must be positive")
        self.query_timeout_seconds = timeout
        self.profile, self.rrf_k, self.candidate_k = profile, rrf_k, candidate_k
        self._client, self._embed_query = client, embed_query
        self._collections = dict(PROFILE_COLLECTION_NAMES[profile])
        self.embedding_model_identity, self.embedding_vector_dim = embedding_config_for_profile(profile)
        self._records: dict[str, OfficialCatalogueRecord] = {}
        self._by_level: dict[str, set[str]] = {level: set() for level in _LEVELS}
        digests: set[str] = set()
        for record in records:
            if record.profile != profile:
                raise ValueError("Catalogue records and requested profile differ")
            if record.level not in self._by_level or not record.code or record.code in self._records:
                raise ValueError("Catalogue has an invalid level or duplicate/blank code")
            if not record.title_en or not record.source_catalogue_sha256:
                raise ValueError("Catalogue records require a title and source digest")
            self._records[record.code] = record
            self._by_level[record.level].add(record.code)
            digests.add(record.source_catalogue_sha256)
        if len(digests) != 1 or not self._by_level["unit"]:
            raise ValueError("One nonempty source digest and unit catalogue are required")
        self.source_catalogue_sha256 = next(iter(digests))
        for level_index, level in enumerate(_LEVELS):
            for code in self._by_level[level]:
                parent = self._records[code].parent_code
                if (level_index == 0 and parent) or (
                    level_index > 0 and parent not in self._by_level[_LEVELS[level_index - 1]]
                ):
                    raise ValueError("Catalogue has a missing or wrong-level parent")
        self._units = [self._records[code] for code in sorted(self._by_level["unit"])]
        titles = [record.title_en + " " + getattr(record, "title_ar", "") for record in self._units]
        bodies: list[str] = []
        for record in self._units:
            # The loader's exact code/title prefix is not a second title field.
            prefix = f"{record.code} {record.title_en}"
            body = record.embedding_text
            if body.startswith(prefix):
                body = body[len(prefix):].lstrip(". ")
            bodies.append(body)
        self._title_index, self._body_index = _BM25Field(titles), _BM25Field(bodies)
        self._has_body = any(unicode_tokens(body) for body in bodies)
        self._paths = {record.code: self._path(record.code) for record in self._units}

    @classmethod
    def from_catalogue_file(cls, catalogue_path: Path, *, profile: str,
                            metadata_path: Path = DEFAULT_METADATA_PATH, **kwargs):
        loader = load_enriched_catalogue if profile in (
            ENRICHED_PROFILE, ENRICHED_E5LARGE_PROFILE
        ) else load_official_catalogue
        records = loader(Path(catalogue_path), metadata_path=metadata_path, profile=profile)
        return cls(records, profile=profile, **kwargs)

    def _path(self, code: str) -> list[str]:
        path = []
        while code:
            path.append(code)
            code = self._records[code].parent_code
        return list(reversed(path))

    def _lexical_scores(self, query: str) -> dict[str, float]:
        tokens = unicode_tokens(query)
        title_scores = self._title_index.scores(tokens) if self.title_weight else {}
        body_scores = self._body_index.scores(tokens) if self.body_weight else {}
        return {
            self._units[index].code: self.title_weight * title_scores.get(index, 0.0)
            + self.body_weight * body_scores.get(index, 0.0)
            for index in title_scores.keys() | body_scores.keys()
        }

    def lexical_scores(self, query: str) -> list[tuple[str, float]]:
        """Positive field-weighted BM25 scores, sorted deterministically.

        This uses the same catalogue fields/index as fusion and performs no
        dense inference. An empty list is evidence of no lexical match, rather
        than a list of arbitrarily ordered zero-score codes.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonblank text")
        return sorted(
            ((code, score) for code, score in self._lexical_scores(query).items() if score > 0),
            key=lambda item: (-item[1], item[0]),
        )

    def rank(
        self,
        query: str,
        *,
        dense_scores_by_level: Mapping[str, Sequence[tuple[str, float]]],
        top_k: int = 5,
        trace: Optional[dict] = None,
    ) -> HybridResult:
        """Rank cached scores, preserving absent lexical evidence and provenance.

        Scores may contain all official codes or only a retrieved subset. Input
        ordering has no effect. Duplicate codes count once (highest score wins).
        Unknown/wrong-level codes and non-finite scores fail closed rather than
        entering a catalogue-independent prediction. Empty unit dense evidence
        uses an explicitly labelled sparse fallback; without positive lexical
        evidence the result is unavailable, not the first arbitrary code.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonblank text")
        _positive_integer("top_k", top_k, 436)
        unknown_levels = set(dense_scores_by_level) - set(_LEVELS)
        if unknown_levels:
            raise ValueError(f"Unknown dense levels: {sorted(unknown_levels)}")
        dense_scores: dict[str, dict[str, float]] = {}
        dense_ranks: dict[str, dict[str, int]] = {}
        for level in _LEVELS:
            scores: dict[str, float] = {}
            for code, raw_score in dense_scores_by_level.get(level, ()):
                if code not in self._by_level[level]:
                    raise ValueError(f"Dense {level} code {code!r} is outside the selected catalogue")
                if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
                    raise ValueError("Dense scores must be finite numbers")
                if not math.isfinite(raw_score):
                    raise ValueError("Dense scores must be finite numbers")
                scores[code] = max(float(raw_score), scores.get(code, -math.inf))
            dense_scores[level] = scores
            ordered = sorted(scores, key=lambda code: (-scores[code], code))
            dense_ranks[level] = {code: index for index, code in enumerate(ordered, start=1)}

        lexical_scores = self._lexical_scores(query) if self.lexical_weight else {}
        lexical_order = sorted(
            (code for code, score in lexical_scores.items() if score > 0),
            key=lambda code: (-lexical_scores[code], code),
        )[:self.candidate_k]
        lexical_ranks = {code: index for index, code in enumerate(lexical_order, start=1)}
        unit_ranks = {
            code: rank for code, rank in dense_ranks["unit"].items()
            if rank <= self.candidate_k and self.dense_weight
        }
        pool = unit_ranks.keys() | lexical_ranks.keys()
        dense_available = bool(dense_scores["unit"]) and bool(self.dense_weight)
        requested = ISCO_HYBRID_SOFT_HIERARCHY_RRF if self.hierarchy_weight else ISCO_HYBRID_RRF
        fallback_used = not dense_available
        method = requested if dense_available else (
            ISCO_HYBRID_SPARSE_FALLBACK if lexical_order else ISCO_HYBRID_UNAVAILABLE
        )
        fallback_reason = "unit_dense_evidence_unavailable" if fallback_used else None
        candidates = []
        for code in pool:
            dense_rank, lexical_rank = unit_ranks.get(code), lexical_ranks.get(code)
            support = 0.0
            if dense_available and self.hierarchy_weight:
                support = sum(
                    1.0 / (self.rrf_k + dense_ranks[level][parent])
                    for level, parent in zip(_ANCESTORS, self._paths[code][:-1])
                    if parent in dense_ranks[level]
                ) / len(_ANCESTORS)
            fusion = (self.dense_weight / (self.rrf_k + dense_rank) if dense_rank else 0.0)
            fusion += (self.lexical_weight / (self.rrf_k + lexical_rank) if lexical_rank else 0.0)
            fusion += self.hierarchy_weight * support
            record = self._records[code]
            candidates.append(HybridCandidate(
                code=code, label_en=record.title_en, label_ar=getattr(record, "title_ar", ""),
                dense_score=dense_scores["unit"].get(code), lexical_score=lexical_scores.get(code, 0.0),
                fusion_score=fusion, dense_rank=dense_rank, lexical_rank=lexical_rank,
                hierarchy_support=support, hierarchy_path=list(self._paths[code]),
            ))
        candidates.sort(key=lambda candidate: (-candidate.fusion_score, candidate.code))
        best = candidates[0] if candidates else None
        result = HybridResult(
            code=best.code if best else "", label_en=best.label_en if best else "",
            label_ar=best.label_ar if best else "", ranking_score=best.fusion_score if best else 0.0,
            top_candidates=candidates[:top_k], method=method, requested_method=requested,
            profile=self.profile, source_catalogue_sha256=self.source_catalogue_sha256,
            hierarchy_path=list(best.hierarchy_path) if best else [], fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        if trace is not None:
            trace.update({
                "method": method, "requested_method": requested, "profile": self.profile,
                "source_catalogue_sha256": self.source_catalogue_sha256,
                "ranking_score_calibrated": False, "rrf_rank_indexing": "one_based",
                "parameters": {"rrf_k": self.rrf_k, "candidate_k": self.candidate_k,
                               "lexical_weight": self.lexical_weight, "dense_weight": self.dense_weight,
                               "hierarchy_weight": self.hierarchy_weight,
                               "title_weight": self.title_weight, "body_weight": self.body_weight},
                "lexical_fields": "title_and_official_body" if self._has_body else "title_only",
                "lexical_evidence_present": bool(lexical_order),
                "dense_rankings": {level: [
                    {"code": code, "score": dense_scores[level][code], "rank": rank}
                    for code, rank in ranks.items()
                ] for level, ranks in dense_ranks.items()},
                "lexical_ranking": [{"code": code, "score": lexical_scores[code], "rank": rank}
                                    for code, rank in lexical_ranks.items()],
                "missing_dense_levels": [level for level in _LEVELS if not dense_scores[level]],
                "fused_candidates": [asdict(candidate) for candidate in candidates],
                "fallback_used": fallback_used, "fallback_reason": fallback_reason,
            })
        return result

    def search(self, query: str, *, query_vector: Optional[Sequence[float]] = None,
               top_k: int = 5, trace: Optional[dict] = None) -> HybridResult:
        """Read-only global Qdrant queries plus optional unfiltered ancestor scores.

        Query failures retain an explicit sparse-only fallback. Missing ancestor
        evidence is reported without discarding global unit results. Supplied
        client/model identity must match the selected existing vector profile;
        payload profile/digest/code membership are checked before accepting hits.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonblank text")
        _positive_integer("top_k", top_k, 436)
        failures: dict[str, str] = {}
        dense: dict[str, list[tuple[str, float]]] = {}
        vector = None
        if self._client is not None:
            try:
                if query_vector is None:
                    if self._embed_query is None:
                        raise RuntimeError("query embedder is unavailable")
                    query_vector = self._embed_query(query)
                vector = [float(value) for value in query_vector]
                if len(vector) != self.embedding_vector_dim or not all(math.isfinite(value) for value in vector):
                    raise ValueError("Query vector does not match the selected embedding profile")
            except Exception as error:
                vector = None
                failures["embedding"] = type(error).__name__
        else:
            failures["client"] = "Unavailable"
        if vector is not None:
            levels = ("unit", *_ANCESTORS) if self.hierarchy_weight else ("unit",)
            for level in levels:
                try:
                    response = self._client.query_points(
                        collection_name=self._collections[level], query=vector, query_filter=None,
                        limit=min(self.candidate_k, len(self._by_level[level])) if level == "unit"
                        else len(self._by_level[level]),
                        with_payload=True, timeout=self.query_timeout_seconds,
                    )
                    accepted = []
                    for point in response.points:
                        payload = point.payload or {}
                        if (payload.get("profile") != self.profile
                                or payload.get("source_catalogue_sha256") != self.source_catalogue_sha256
                                or payload.get("code") not in self._by_level[level]):
                            raise ValueError("Retrieved payload is outside the selected catalogue/profile")
                        score = point.score
                        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                            raise ValueError("Retrieved score must be a finite number")
                        accepted.append((payload["code"], float(score)))
                    dense[level] = accepted
                except Exception as error:
                    failures[level] = type(error).__name__
        result = self.rank(query, dense_scores_by_level=dense, top_k=top_k, trace=trace)
        if trace is not None:
            trace["retrieval_failures"] = failures
            trace["embedding_model"] = self.embedding_model_identity
            trace["embedding_vector_dim"] = self.embedding_vector_dim
            trace["query_timeout_seconds"] = self.query_timeout_seconds
        return result
