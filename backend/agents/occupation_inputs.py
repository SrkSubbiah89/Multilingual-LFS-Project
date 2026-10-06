"""Apply respondent-supplied duties consistently without changing frozen RAG.

The title-only invocation stays identical when no distinct real duties exist.
Combined inputs are a separate runtime condition: published title-only WISCO
accuracy does not measure their accuracy. Only an explicitly supplied duties
field enters the retrieval query; auxiliary language/industry context retains
the classifier's existing contract.
"""
from __future__ import annotations

from copy import copy
from dataclasses import is_dataclass, replace
import unicodedata

MAX_DUTIES_CHARS = 1600
PARENT_METHOD = "isco_parent_document_rag"
DUTIES_PARENT_METHOD = "isco_parent_document_duties_rag"
_MISSING = {"", "n/a", "na", "none", "null", "unknown", "not applicable",
            "not_applicable", "not provided", "not_provided", "never_worked",
            "refused", "prefer_not_to_say", "prefer not to say"}


def _comparison(text: str) -> str:
    # Compare canonical equivalents without rewriting the respondent's text.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _query(title: str, duties: str | None) -> tuple[str, bool, bool]:
    if duties is None:
        duties = ""
    if not isinstance(duties, str):
        raise ValueError("Occupation duties must be text")
    cleaned = duties.strip()
    if (_comparison(cleaned) in _MISSING or not any(char.isalpha() for char in cleaned)
            or (isinstance(title, str) and _comparison(cleaned) == _comparison(title))):
        return title, False, False
    if not isinstance(title, str) or not title.strip():
        # The classifier owns its ordinary title validation. Duties cannot
        # create an occupation when the respondent provided no title.
        return title, False, False
    if _comparison(title) in _MISSING:
        return title, False, False
    truncated = len(cleaned) > MAX_DUTIES_CHARS
    bounded = cleaned[:MAX_DUTIES_CHARS].rstrip()
    if not any(char.isalpha() for char in bounded):
        return title, False, truncated
    return title.strip() + " " + bounded, True, truncated


def _copy_result(result, updates: dict):
    if hasattr(result, "model_copy"):
        return result.model_copy(update=updates)
    if is_dataclass(result):
        return replace(result, **updates)
    copied = copy(result)
    for name, value in updates.items():
        setattr(copied, name, value)
    return copied


def classify_occupation_input(classifier, title, *, duties="", **kwargs):
    """Classify once; preserve the reported title, scores and review decision.

    Optional ``trace`` records actual retrieval text and explicit truncation.
    It is caller-owned diagnostic data and should not be publicly published.
    The no-duties result is returned untouched, preserving prior object identity.
    """
    query, used, truncated = _query(title, duties)
    trace = kwargs.get("trace")
    if trace is not None:
        trace.update(duties_used=used, duties_truncated=truncated,
                     query_mode="title_and_duties" if used else "title_only",
                     retrieval_query=query, title_only_input=not used,
                     duties_accuracy_evaluated=False)
    result = classifier.classify(query, **kwargs)
    if not used or result is None:
        return result
    updates = {"query": title.strip()}
    if getattr(result, "method", None) == PARENT_METHOD:
        updates["method"] = DUTIES_PARENT_METHOD
        if trace is not None:
            trace["method"] = DUTIES_PARENT_METHOD
    return _copy_result(result, updates)
