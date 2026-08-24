"""
eval/legacy_thesis_ch6/evaluate.py

Thesis Chapter 6 — Evaluation Framework
=========================================
Compares three ISCO-08 classification approaches on a 100-item synthetic
test corpus of occupation descriptions:

    1. BM25Baseline       — rank-bm25 sparse retrieval, no embeddings
    2. FlatVectorBaseline — dense retrieval against flat isco_occupations
    3. HierarchicalRAG    — 4-stage hierarchical pipeline (thesis method)

Metrics reported per system:
    top1_accuracy   — exact 4-digit ISCO code match at rank-1
    top3_accuracy   — exact match anywhere in top-3 results
    cohen_kappa     — sklearn inter-rater agreement at major-group level
    hitl_rate       — fraction of predictions with confidence < 0.70
    avg_latency_ms  — mean prediction time in milliseconds

Usage
-----
    python -m eval.legacy_thesis_ch6.evaluate
    python -m eval.legacy_thesis_ch6.evaluate --system bm25
    python -m eval.legacy_thesis_ch6.evaluate --system flat
    python -m eval.legacy_thesis_ch6.evaluate --system hierarchical
    python -m eval.legacy_thesis_ch6.evaluate --output results.json
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError as _err:
    raise ImportError(
        "rank-bm25 is required for the evaluation framework.\n"
        "Install it with:  pip install rank-bm25"
    ) from _err

from sklearn.metrics import cohen_kappa_score

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HITL_THRESHOLD  = 0.70      # matches ISCOClassifier.HITL_THRESHOLD
_DEFAULT_TOP_K  = 3
_CSV_DEFAULT    = "eval/results/legacy_thesis_ch6/results.csv"

# ---------------------------------------------------------------------------
# 100-item synthetic test corpus
# Each entry: (job_description, correct_isco_4digit_code)
#
# Difficulty legend
#   [E] Easy   — plain job title, single obvious match
#   [M] Medium — short description requiring minor inference
#   [H] Hard   — vague / ambiguous / cross-domain wording
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str]] = [

    # ── Major 0: Armed Forces ──────────────────────────────────────────────
    ("Army officer commanding a battalion",                              "0110"),  # [E]
    ("Military soldier serving in a ground combat unit",                "0310"),  # [E]
    ("I protect my country and follow military orders",                 "0310"),  # [M]

    # ── Major 1: Managers ─────────────────────────────────────────────────
    ("Chief Executive Officer of a multinational company",              "1112"),  # [E]
    ("General Manager overseeing daily operations",                     "1211"),  # [E]
    ("Financial Manager responsible for budgets and forecasting",       "1211"),  # [E]
    ("Human Resources Manager developing talent strategies",            "1212"),  # [E]
    ("Sales Manager directing a regional sales team",                   "1221"),  # [E]
    ("IT Manager overseeing network infrastructure",                    "1330"),  # [E]
    ("Hotel General Manager directing hospitality services",            "1411"),  # [E]
    ("Operations Manager in a manufacturing plant",                     "1321"),  # [E]
    ("I manage a team of engineers and set delivery timelines",         "1223"),  # [M]
    ("I run a small restaurant and handle all the staff",               "1412"),  # [M]
    ("I oversee marketing campaigns and brand strategy for our firm",   "1221"),  # [M]

    # ── Major 2: Professionals ────────────────────────────────────────────
    ("Software Developer writing backend APIs in Python",               "2512"),  # [E]
    ("software developer",                                              "2512"),  # [E]
    ("Medical Doctor diagnosing and treating patients",                 "2211"),  # [E]
    ("Civil Engineer designing bridges and road infrastructure",        "2142"),  # [E]
    ("Registered Nurse providing patient care in hospital",             "2221"),  # [E]
    ("Accountant preparing financial statements and tax returns",       "2411"),  # [E]
    ("Lawyer providing legal advice and representing clients",          "2611"),  # [E]
    ("High School Teacher teaching mathematics",                        "2330"),  # [E]
    ("Data Scientist building machine learning models",                 "2511"),  # [E]
    ("Architect designing residential and commercial buildings",        "2161"),  # [E]
    ("Pharmacist dispensing medications and counselling patients",      "2262"),  # [E]
    ("Dentist performing dental examinations",                          "2261"),  # [E]
    ("Economist researching labour market trends",                      "2631"),  # [E]
    ("Financial Analyst evaluating investment opportunities",           "2413"),  # [E]
    ("Psychologist providing counselling and therapy",                  "2634"),  # [E]
    ("I analyse data and build predictive models for business",         "2511"),  # [M]
    ("I write code for web applications and fix bugs",                  "2512"),  # [M]
    ("I do things with computers at a hospital",                        "2512"),  # [H]
    ("I help people feel better by talking through their problems",     "2634"),  # [H]
    ("I work with numbers and help companies save money",               "2411"),  # [H]

    # ── Major 3: Technicians & Associate Professionals ────────────────────
    ("Computer Network Technician maintaining LAN and WAN",            "3511"),  # [E]
    ("Medical Laboratory Technician performing blood tests",           "3212"),  # [E]
    ("Civil Engineering Technician assisting with site surveys",       "3112"),  # [E]
    ("Air Traffic Controller managing aircraft movements",             "3154"),  # [E]
    ("Radiographer operating X-ray and CT scan equipment",             "3211"),  # [E]
    ("Real Estate Agent helping clients buy and sell properties",      "3334"),  # [E]
    ("Insurance Agent selling life and property insurance policies",   "3322"),  # [E]
    ("Legal Paralegal supporting attorneys with research",             "3411"),  # [E]
    ("I set up and troubleshoot networks and servers",                 "3511"),  # [M]
    ("I test blood samples and report results to doctors",             "3212"),  # [M]
    ("I inspect construction sites and write safety reports",          "3257"),  # [M]

    # ── Major 4: Clerical Support Workers ─────────────────────────────────
    ("Office Secretary managing schedules and correspondence",         "4120"),  # [E]
    ("Data Entry Clerk inputting records into databases",              "4132"),  # [E]
    ("Receptionist greeting visitors and answering calls",             "4226"),  # [E]
    ("Customer Service Representative handling complaints",            "4225"),  # [E]
    ("Bank Teller processing deposits and withdrawals",                "4211"),  # [E]
    ("Payroll Clerk processing employee salary payments",              "4313"),  # [E]
    ("Call Centre Agent handling inbound customer calls",              "4225"),  # [E]
    ("I answer phones, greet visitors and manage meeting rooms",       "4226"),  # [M]
    ("I type documents, file papers and manage the office calendar",   "4120"),  # [M]
    ("I sit at a desk and enter information into a computer all day",  "4132"),  # [H]

    # ── Major 5: Service and Sales Workers ────────────────────────────────
    ("Retail Sales Assistant helping customers in a clothing store",   "5223"),  # [E]
    ("Cook preparing meals in a restaurant kitchen",                   "5120"),  # [E]
    ("Security Guard patrolling premises and monitoring CCTV",         "5414"),  # [E]
    ("Waiter serving food and beverages in a restaurant",              "5131"),  # [E]
    ("Hairdresser cutting and styling clients hair",                   "5141"),  # [E]
    ("Police Officer maintaining public order and safety",             "5412"),  # [E]
    ("Firefighter responding to fire emergencies",                     "5411"),  # [E]
    ("Cashier processing payments at a supermarket",                   "5221"),  # [E]
    ("Home Care Aide assisting elderly with daily activities",         "5322"),  # [E]
    ("Tour Guide leading tourists through historical sites",           "5113"),  # [E]
    ("I sell things to customers and handle the register",             "5223"),  # [H]

    # ── Major 6: Agricultural, Forestry and Fishery Workers ───────────────
    ("Farmer cultivating crops and managing irrigation",               "6111"),  # [E]
    ("Livestock Farmer raising cattle and poultry",                    "6121"),  # [E]
    ("Fisherman operating fishing nets in coastal waters",             "6161"),  # [E]
    ("Greenhouse Grower producing vegetables hydroponically",          "6112"),  # [E]
    ("I work the land and grow vegetables to sell at the market",      "6111"),  # [M]

    # ── Major 7: Craft and Related Trades ─────────────────────────────────
    ("Electrician installing and maintaining electrical wiring",       "7411"),  # [E]
    ("Plumber fitting and repairing water pipes",                      "7126"),  # [E]
    ("Carpenter making furniture and fitting wooden structures",       "7115"),  # [E]
    ("Welder joining metal components using arc welding",              "7212"),  # [E]
    ("Automotive Mechanic diagnosing and repairing vehicles",          "7231"),  # [E]
    ("Mason laying bricks and blocks for construction",                "7112"),  # [E]
    ("Painter applying paint and finishes to buildings",               "7131"),  # [E]
    ("I fix cars and motorcycles at a garage",                         "7231"),  # [M]
    ("I wire houses and install circuit breakers",                     "7411"),  # [M]
    ("I use tools to build and repair things at construction sites",   "7112"),  # [H]

    # ── Major 8: Plant and Machine Operators ──────────────────────────────
    ("Forklift Operator moving goods in a warehouse",                  "8344"),  # [E]
    ("Truck Driver transporting goods across the country",             "8332"),  # [E]
    ("Crane Operator lifting heavy materials on construction sites",   "8343"),  # [E]
    ("Bus Driver transporting passengers on urban routes",             "8331"),  # [E]
    ("Chemical Plant Operator monitoring refinery processes",          "8131"),  # [E]
    ("Assembly Line Worker assembling electronic components",          "8211"),  # [E]
    ("Mining Machine Operator drilling boreholes",                     "8112"),  # [E]
    ("I drive a large vehicle and deliver products to warehouses",     "8332"),  # [M]
    ("I operate heavy machinery and move large items around a site",   "8343"),  # [M]
    ("I work in a factory and operate machines all day",               "8211"),  # [H]

    # ── Major 9: Elementary Occupations ───────────────────────────────────
    ("Cleaner sweeping and mopping office floors",                     "9112"),  # [E]
    ("Domestic Helper cleaning houses and doing laundry",              "9111"),  # [E]
    ("Garbage Collector collecting waste from residential areas",      "9129"),  # [E]
    ("Agricultural Labourer digging and planting in fields",          "9211"),  # [E]
    ("Kitchen Helper washing dishes and preparing ingredients",        "9412"),  # [E]
    ("Delivery Worker delivering parcels door-to-door",               "9333"),  # [E]
    ("Building Caretaker maintaining communal areas",                  "9141"),  # [E]
    ("I wash and iron clothes at a laundry",                           "9121"),  # [M]
    ("I carry things, clean up and do whatever is needed",             "9333"),  # [H]
]

assert len(TEST_CASES) == 100, f"Expected 100 test cases, got {len(TEST_CASES)}"

# ---------------------------------------------------------------------------
# 30-item Arabic test corpus (Gulf Arabic + MSA) — UAE LFS focus
# Covers all 10 ISCO major groups; Gulf dialect terms included
# ---------------------------------------------------------------------------

ARABIC_TEST_CASES: list[tuple[str, str]] = [
    # Major 0 — Armed Forces
    ("ضابط في القوات المسلحة الإماراتية",                            "0110"),  # [E]
    ("جندي يؤدي واجبات الحراسة العسكرية",                           "0310"),  # [E]

    # Major 1 — Managers
    ("مدير تنفيذي لشركة متعددة الجنسيات في دبي",                    "1112"),  # [E]
    ("مدير الموارد البشرية في مؤسسة حكومية",                         "1212"),  # [E]
    ("مدير فندق خمس نجوم في أبوظبي",                                 "1411"),  # [E]

    # Major 2 — Professionals
    ("مهندس برمجيات يطور تطبيقات الجوال",                            "2512"),  # [E]
    ("طبيب متخصص في الجراحة العامة",                                  "2211"),  # [E]
    ("محامٍ متخصص في قانون الشركات",                                  "2611"),  # [E]
    ("مدرس رياضيات في مدرسة ثانوية",                                  "2330"),  # [E]
    ("محاسب قانوني معتمد يراجع الميزانيات",                           "2411"),  # [E]

    # Major 3 — Technicians
    ("فني مختبر يجري الفحوصات الطبية",                               "3212"),  # [E]
    ("مساعد مهندس يشرف على أعمال البناء",                            "3112"),  # [M]
    ("ممرض يعمل في وحدة العناية المركزة",                            "3221"),  # [E]

    # Major 4 — Clerical
    ("موظف استقبال في مستشفى حكومي",                                 "4226"),  # [E]
    ("كاتب بيانات يدخل المعلومات في الحاسوب",                        "4132"),  # [E]
    ("أبي يشتغل في مكتب ويرد على المكالمات",                         "4224"),  # [M] Gulf Arabic

    # Major 5 — Service & Sales
    ("بائع في محل تجاري بمركز تسوق",                                 "5223"),  # [E]
    ("سائق تاكسي يعمل في شوارع دبي",                                 "5322"),  # [E]
    ("طباخ في مطعم فندقي يعد الوجبات الخليجية",                      "5120"),  # [E]
    ("حارس أمن في مبنى تجاري",                                        "5414"),  # [E]

    # Major 6 — Agriculture
    ("عامل زراعي يرعى النخيل في المزارع",                            "6121"),  # [E]
    ("صياد أسماك في موانئ عجمان",                                     "6221"),  # [E]

    # Major 7 — Craft & Trades
    ("نجار يصنع الأثاث الخشبي",                                       "7422"),  # [E]
    ("كهربائي يقوم بتمديد الأسلاك في المباني",                        "7411"),  # [E]
    ("سباك يصلح أنابيب المياه",                                        "7126"),  # [E]

    # Major 8 — Operators
    ("سائق شاحنة ثقيلة لنقل البضائع",                                "8332"),  # [E]
    ("مشغل آلات في مصنع للبلاستيك",                                   "8131"),  # [M]

    # Major 9 — Elementary
    ("عامل نظافة في مراكز التسوق",                                    "9112"),  # [E]
    ("عامل بناء يحمل مواد في مواقع البناء",                          "9313"),  # [E]
    ("موزع بريد يوصل الرسائل والطرود",                                "9621"),  # [E]
]

assert len(ARABIC_TEST_CASES) == 30, f"Expected 30 Arabic test cases, got {len(ARABIC_TEST_CASES)}"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """One system's prediction for one test case."""
    predicted_code: str          # 4-digit ISCO code
    top3_codes: list[str]
    confidence: float
    latency_ms: float


_MAJOR_LABELS: dict[str, str] = {
    "0": "Armed Forces",
    "1": "Managers",
    "2": "Professionals",
    "3": "Technicians",
    "4": "Clerical",
    "5": "Service & Sales",
    "6": "Agriculture",
    "7": "Craft & Trade",
    "8": "Operators",
    "9": "Elementary",
}


@dataclass
class MajorGroupMetrics:
    """Per-major-group precision, recall, and F1."""
    major_code: str
    label: str
    n_true: int          # support (ground truth count)
    n_predicted: int     # how many times this major was predicted
    tp: int              # true positives
    precision: float
    recall: float
    f1: float


@dataclass
class SystemMetrics:
    """Aggregated evaluation metrics for one classification system."""
    system_name: str
    top1_accuracy: float
    top3_accuracy: float
    cohen_kappa: float           # major-group level, sklearn
    hitl_rate: float             # fraction with confidence < HITL_THRESHOLD
    avg_latency_ms: float
    p95_latency_ms: float        # 95th-percentile latency
    p99_latency_ms: float        # 99th-percentile latency
    n_evaluated: int
    errors: int = 0
    hallucination_count: int = 0  # predictions not found in ISCO-08 code set
    per_major: list[MajorGroupMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. BM25 Baseline  (rank-bm25 library)
# ---------------------------------------------------------------------------

class BM25Baseline:
    """
    Sparse BM25 retrieval over ISCO-08 title + description strings.

    Uses the curated ``_ISCO_DATA`` list from vector_store.py so the
    retrieval vocabulary is identical to the other baselines.
    Confidence is the normalised BM25 score of the top hit.
    """

    def __init__(self) -> None:
        from backend.rag.vector_store import _ISCO_DATA
        self._entries = [e for e in _ISCO_DATA if len(e["code"]) == 4]
        self._codes   = [e["code"] for e in self._entries]
        corpus_tokens = [
            self._tokenise(
                f"{e['code']} {e['title_en']} {e.get('title_ar', '')} "
                f"{e.get('description', '')}"
            )
            for e in self._entries
        ]
        self._bm25 = BM25Okapi(corpus_tokens)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        import re
        return re.findall(r"\w+", text.lower())

    def predict(self, text: str, top_k: int = _DEFAULT_TOP_K) -> PredictionResult:
        t0     = time.perf_counter()
        tokens = self._tokenise(text)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(scores, self._codes), reverse=True)
        ms     = (time.perf_counter() - t0) * 1_000

        top_codes  = [c for _, c in ranked[:top_k]]
        top_score  = ranked[0][0] if ranked else 0.0
        max_score  = max(s for s, _ in ranked) if ranked else 1.0
        confidence = float(min(top_score / max(max_score, 1e-9), 1.0))

        return PredictionResult(
            predicted_code=top_codes[0] if top_codes else "",
            top3_codes=top_codes,
            confidence=round(confidence, 4),
            latency_ms=round(ms, 2),
        )


# ---------------------------------------------------------------------------
# 2. Flat Vector Baseline
# ---------------------------------------------------------------------------

class FlatVectorBaseline:
    """
    Dense retrieval against the flat ``isco_occupations`` Qdrant collection.
    Uses the multilingual-e5-small model embedded in VectorStore.
    """

    def __init__(self) -> None:
        from backend.rag.vector_store import get_vector_store
        self._vs = get_vector_store()

    def predict(self, text: str, top_k: int = _DEFAULT_TOP_K) -> PredictionResult:
        t0      = time.perf_counter()
        matches = self._vs.search(text, top_k=top_k)
        ms      = (time.perf_counter() - t0) * 1_000

        top_codes  = [m.code for m in matches]
        confidence = matches[0].confidence if matches else 0.0

        return PredictionResult(
            predicted_code=top_codes[0] if top_codes else "",
            top3_codes=top_codes,
            confidence=round(float(confidence), 4),
            latency_ms=round(ms, 2),
        )


# ---------------------------------------------------------------------------
# 3. Hierarchical RAG  (4-stage pipeline — thesis method)
# ---------------------------------------------------------------------------

class HierarchicalRAG:
    """
    Full 4-stage hierarchical Qdrant pipeline via ISCOClassifier.

    Falls back to flat search transparently when hierarchical
    collections are not yet populated.
    """

    def __init__(self) -> None:
        from backend.agents.isco_classifier import ISCOClassifier
        self._clf = ISCOClassifier()

    def predict(self, text: str, top_k: int = _DEFAULT_TOP_K) -> PredictionResult:
        t0     = time.perf_counter()
        result = self._clf.classify(text)
        ms     = (time.perf_counter() - t0) * 1_000

        primary    = result.primary.code
        alts       = [a.code for a in result.alternatives]
        top_codes  = ([primary] + alts)[:top_k]
        confidence = result.primary.confidence

        return PredictionResult(
            predicted_code=primary,
            top3_codes=top_codes,
            confidence=round(float(confidence), 4),
            latency_ms=round(ms, 2),
        )


# ---------------------------------------------------------------------------
# 4. No-RAG Baseline  (pure LLM, zero retrieval)
# ---------------------------------------------------------------------------

class NoRAGBaseline:
    """
    Pure LLM classification with absolutely no retrieval augmentation.

    The LLM receives the raw job description and must produce a 4-digit
    ISCO-08 code from its parametric knowledge alone.  This baseline
    establishes the floor for accuracy and measures:

    * Hallucination rate — how often the LLM returns a code that does not
      exist in the ISCO-08 standard (a direct measure of confabulation).
    * Latency — LLM inference time with no vector-search overhead.

    Calls Ollama's /api/generate endpoint directly (no CrewAI overhead)
    for speed and reliability.  Falls back to a structured LLM call if
    the direct HTTP route fails.
    """

    _PROMPT_TEMPLATE = (
        "You are an ISCO-08 (International Standard Classification of "
        "Occupations) expert.  Given the job description below, return the "
        "single most appropriate 4-digit ISCO-08 code.\n\n"
        "Job description: {text}\n\n"
        "Rules:\n"
        "- Use ONLY real ISCO-08 4-digit codes that exist in the standard.\n"
        "- Return ONLY a JSON object, nothing else.\n"
        "- Format: {{\"code\": \"XXXX\", \"confidence\": 0.00}}\n"
    )

    def __init__(self) -> None:
        import os
        from backend.rag.vector_store import _ISCO_DATA
        self._valid      = frozenset(
            e["code"] for e in _ISCO_DATA if len(e["code"]) == 4
        )
        self._ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._model      = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    def _call_ollama(self, prompt: str, timeout: int = 30) -> str:
        """Direct Ollama /api/generate call — no CrewAI overhead."""
        import json as _json
        import urllib.request
        payload = _json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80},
        }).encode()
        req = urllib.request.Request(
            f"{self._ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = _json.loads(resp.read())
            return body.get("response", "")

    def predict(self, text: str, top_k: int = _DEFAULT_TOP_K) -> PredictionResult:
        import json
        import re as _re

        prompt = self._PROMPT_TEMPLATE.format(text=text)
        t0     = time.perf_counter()
        try:
            raw = self._call_ollama(prompt, timeout=30)
        except Exception as exc:
            _logger.debug("Ollama direct call failed: %s", exc)
            raw = ""
        ms = (time.perf_counter() - t0) * 1_000

        code       = ""
        confidence = 0.5
        try:
            clean = _re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")
            # Extract first {...} block
            m = _re.search(r"\{[^}]+\}", clean)
            if m:
                obj  = json.loads(m.group(0))
                code = str(obj.get("code", "")).strip()[:4]
                confidence = float(obj.get("confidence", 0.5))
        except Exception:
            pass

        if not code:
            # Last-resort: grab first 4-digit sequence in the response
            m = _re.search(r"\b(\d{4})\b", raw)
            if m:
                code = m.group(1)

        # Mark hallucinated codes explicitly
        if code and self._valid and code not in self._valid:
            confidence = 0.1

        return PredictionResult(
            predicted_code=code,
            top3_codes=[code] if code else [],
            confidence=round(confidence, 4),
            latency_ms=round(ms, 2),
        )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _get_valid_isco_codes() -> frozenset:
    """
    Return the full set of valid 4-digit ISCO-08 codes.

    Tries three sources in order:
    1. Qdrant isco08_unit_groups collection (full 436 unit groups)
    2. Qdrant isco_occupations flat collection
    3. _ISCO_DATA fallback (~110 curated entries)

    Using only _ISCO_DATA would produce false hallucination positives for
    RAG systems that correctly return codes outside the curated subset.
    """
    valid: set[str] = set()

    # Source 1 & 2: Qdrant collections (authoritative, full coverage)
    try:
        from qdrant_client import QdrantClient
        import os
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
        )
        for collection in ("isco08_unit_groups", "isco_occupations"):
            try:
                result, _ = client.scroll(
                    collection_name=collection,
                    limit=500,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in result:
                    code = str(pt.payload.get("code", ""))
                    if len(code) == 4 and code.isdigit():
                        valid.add(code)
                if valid:
                    break
            except Exception:
                continue
    except Exception:
        pass

    # Source 3: _ISCO_DATA fallback
    if not valid:
        try:
            from backend.rag.vector_store import _ISCO_DATA
            valid = {e["code"] for e in _ISCO_DATA if len(e["code"]) == 4}
        except Exception:
            pass

    return frozenset(valid)


def evaluate_system(
    system_name: str,
    predict_fn,
    test_cases: list[tuple[str, str]],
    top_k: int = _DEFAULT_TOP_K,
) -> SystemMetrics:
    """
    Run *predict_fn* over all test cases and compute metrics.

    Parameters
    ----------
    system_name : str
    predict_fn  : callable(text: str, top_k: int) -> PredictionResult
    test_cases  : list of (job_description, true_isco_4digit)
    top_k       : candidates per query
    """
    top1_hits = top3_hits = hitl_count = halluc_count = 0
    latencies: list[float] = []
    pred_majors: list[str] = []
    true_majors: list[str] = []
    errors = 0
    _valid_codes = _get_valid_isco_codes()

    # Per-major-group counters: {major_code: {"tp", "fp", "fn"}}
    from collections import defaultdict
    _major_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for text, true_code in test_cases:
        try:
            res = predict_fn(text, top_k)
        except Exception as exc:
            _logger.warning("Predict error for %r: %s", text[:50], exc)
            errors += 1
            pred_majors.append("?")
            true_majors.append(true_code[:1])
            continue

        latencies.append(res.latency_ms)
        pred       = res.predicted_code
        pred_major = pred[:1] if pred else "?"
        true_major = true_code[:1]

        top1_hits    += int(pred == true_code)
        top3_hits    += int(true_code in res.top3_codes)
        hitl_count   += int(res.confidence < HITL_THRESHOLD)
        halluc_count += int(bool(pred) and _valid_codes and pred not in _valid_codes)

        pred_majors.append(pred_major)
        true_majors.append(true_major)

        # Per-major TP / FP / FN
        if pred_major == true_major:
            _major_stats[true_major]["tp"] += 1
        else:
            _major_stats[true_major]["fn"] += 1
            if pred_major != "?":
                _major_stats[pred_major]["fp"] += 1

    n       = len(test_cases) - errors
    avg_lat = sum(latencies) / max(len(latencies), 1)

    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = (pct / 100) * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 2)

    p95_lat = _percentile(latencies, 95)
    p99_lat = _percentile(latencies, 99)

    # Cohen's Kappa at major-group level (sklearn)
    if n >= 2 and len(set(true_majors)) > 1:
        kappa = float(cohen_kappa_score(true_majors, pred_majors))
    elif n >= 2:
        kappa = 1.0 if pred_majors == true_majors else 0.0
    else:
        kappa = 0.0

    # Build per-major-group metrics
    per_major: list[MajorGroupMetrics] = []
    all_true_major_counts: dict[str, int] = defaultdict(int)
    all_pred_major_counts: dict[str, int] = defaultdict(int)
    for tm, pm in zip(true_majors, pred_majors):
        all_true_major_counts[tm] += 1
        all_pred_major_counts[pm] += 1

    for code in sorted(_major_stats.keys()):
        stats = _major_stats[code]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        prec   = tp / max(tp + fp, 1)
        rec    = tp / max(tp + fn, 1)
        f1     = 2 * prec * rec / max(prec + rec, 1e-9)
        n_true = all_true_major_counts.get(code, 0)
        n_pred = all_pred_major_counts.get(code, 0)
        per_major.append(MajorGroupMetrics(
            major_code=code,
            label=_MAJOR_LABELS.get(code, "Unknown"),
            n_true=n_true,
            n_predicted=n_pred,
            tp=tp,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
        ))

    return SystemMetrics(
        system_name=system_name,
        top1_accuracy=round(top1_hits / max(n, 1), 4),
        top3_accuracy=round(top3_hits / max(n, 1), 4),
        cohen_kappa=round(kappa, 4),
        hitl_rate=round(hitl_count / max(n, 1), 4),
        avg_latency_ms=round(avg_lat, 2),
        p95_latency_ms=p95_lat,
        p99_latency_ms=p99_lat,
        n_evaluated=n,
        errors=errors,
        hallucination_count=halluc_count,
        per_major=per_major,
    )


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    top_k: int = _DEFAULT_TOP_K,
    systems: Optional[list[str]] = None,
    test_cases: Optional[list[tuple[str, str]]] = None,
) -> dict[str, SystemMetrics]:
    """Instantiate and evaluate all (or selected) systems."""
    registry = {
        "bm25":         BM25Baseline,
        "flat":         FlatVectorBaseline,
        "hierarchical": HierarchicalRAG,
    }
    selected = systems or list(registry.keys())
    cases = test_cases if test_cases is not None else TEST_CASES
    results: dict[str, SystemMetrics] = {}

    for name in selected:
        if name not in registry:
            _logger.warning("Unknown system %r — skipping.", name)
            continue
        print(f"\n{'='*60}\nEvaluating: {name.upper()}\n{'='*60}")
        try:
            sys_obj = registry[name]()
        except Exception as exc:
            print(f"  ERROR initialising {name}: {exc}")
            continue

        metrics = evaluate_system(
            system_name=name,
            predict_fn=sys_obj.predict,
            test_cases=cases,
            top_k=top_k,
        )
        results[name] = metrics
        _print_metrics(metrics)
        _print_per_major_breakdown(metrics)

    return results


def _print_metrics(m: SystemMetrics) -> None:
    print(f"  n_evaluated   : {m.n_evaluated}  (errors: {m.errors})")
    print(f"  top1_accuracy : {m.top1_accuracy:.2%}")
    print(f"  top3_accuracy : {m.top3_accuracy:.2%}")
    print(f"  cohen_kappa   : {m.cohen_kappa:.4f}")
    print(f"  hitl_rate     : {m.hitl_rate:.2%}  (conf < {HITL_THRESHOLD})")
    print(f"  avg_latency   : {m.avg_latency_ms:.1f} ms  "
          f"(p95={m.p95_latency_ms:.1f} ms  p99={m.p99_latency_ms:.1f} ms)")


def _print_per_major_breakdown(m: SystemMetrics) -> None:
    """Print per-major-group precision / recall / F1 table (thesis Table 4)."""
    if not m.per_major:
        return
    print(f"\n  Per-Major-Group Breakdown — {m.system_name.upper()}")
    hdr = f"  {'Major':<4} {'Label':<20} {'Support':>7} {'Pred':>5} {'TP':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}"
    print(f"  {'-'*len(hdr.strip())}")
    print(hdr)
    print(f"  {'-'*len(hdr.strip())}")
    for mg in m.per_major:
        print(
            f"  {mg.major_code:<4} {mg.label:<20} {mg.n_true:>7} "
            f"{mg.n_predicted:>5} {mg.tp:>4} {mg.precision:>6.2%} "
            f"{mg.recall:>6.2%} {mg.f1:>6.2%}"
        )
    macro_f1 = sum(mg.f1 for mg in m.per_major) / max(len(m.per_major), 1)
    print(f"  {'-'*len(hdr.strip())}")
    print(f"  {'Macro avg':<25} {'':<28} {macro_f1:>6.2%}")


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

class _AblationNoKeyword(HierarchicalRAG):
    """Hierarchical RAG without the keyword major-group pre-filter."""

    def predict(self, text: str, top_k: int = _DEFAULT_TOP_K) -> PredictionResult:
        t0 = time.perf_counter()
        import re as _re
        arabic_chars = len(_re.findall(r"[\u0600-\u06FF]", text))
        lang_hint = "ar" if arabic_chars / max(len(text), 1) > 0.3 else "en"
        result = self._clf._classify_hierarchical(text, "", lang_hint, top_k, major_hint="")
        ms = (time.perf_counter() - t0) * 1_000
        primary   = result.primary.code
        alts      = [a.code for a in result.alternatives]
        top_codes = ([primary] + alts)[:top_k]
        return PredictionResult(
            predicted_code=primary,
            top3_codes=top_codes,
            confidence=round(float(result.primary.confidence), 4),
            latency_ms=round(ms, 2),
        )


class _AblationNoBeam(HierarchicalRAG):
    """
    Hierarchical RAG with beam width forced to 1 (strict greedy top-1).

    Demonstrates the cascade-error problem that beam search solves.
    Uses the new ``beam`` parameter added to ``HierarchicalISCOStore.search()``.
    """

    def predict(self, text: str, top_k: int = _DEFAULT_TOP_K) -> PredictionResult:
        t0    = time.perf_counter()
        store = self._clf._hierarchical_store
        if store is None:
            return super().predict(text, top_k)

        import re as _re
        arabic_chars = len(_re.findall(r"[\u0600-\u06FF]", text))
        lang_hint = "ar" if arabic_chars / max(len(text), 1) > 0.3 else "en"

        # Use beam=1 (greedy top-1 at every intermediate stage)
        from backend.agents.isco_classifier import _keyword_major_hint
        major_hint = _keyword_major_hint(text)
        h = store.search(text, top_k=top_k, major_hint=major_hint, beam=1)
        ms = (time.perf_counter() - t0) * 1_000

        primary   = h.code
        top_codes = [c.code for c in h.top_candidates[:top_k]]
        if primary and primary not in top_codes:
            top_codes.insert(0, primary)
        return PredictionResult(
            predicted_code=primary,
            top3_codes=top_codes[:top_k],
            confidence=round(float(h.confidence), 4),
            latency_ms=round(ms, 2),
        )


def run_ablation(top_k: int = _DEFAULT_TOP_K) -> dict[str, SystemMetrics]:
    """
    Ablation study: measure the contribution of each thesis component.

    Variants evaluated
    ------------------
    bm25       : BM25 sparse retrieval (keyword-only baseline)
    flat       : Single-stage dense vector search (flat baseline)
    no_keyword : Hierarchical RAG without major-group keyword pre-filter
    full       : Full system (hierarchical + keyword hint + beam width=2)

    The delta rows show the exact accuracy gain attributable to each
    architectural decision, which forms the empirical backbone of the
    thesis contribution chapter.
    """
    print(f"\n{'='*60}\nABLATION STUDY — Thesis Table 5\n{'='*60}")
    print("Component contributions:")
    print("  bm25        — sparse keyword retrieval        (baseline A)")
    print("  flat        — dense flat vector search        (baseline B)")
    print("  no_keyword  — hierarchical RAG, no keyword hint")
    print("  no_beam     — hierarchical RAG, beam=1 greedy (no beam search)")
    print("  full        — hierarchical RAG, keyword + beam=2 (proposed)")

    results: dict[str, SystemMetrics] = {}

    ablation_systems: list[tuple[str, type]] = [
        ("bm25",       BM25Baseline),
        ("flat",       FlatVectorBaseline),
        ("no_keyword", _AblationNoKeyword),
        ("no_beam",    _AblationNoBeam),
        ("full",       HierarchicalRAG),
    ]

    for name, cls in ablation_systems:
        print(f"\n  [{name}] initialising...")
        try:
            sys_obj = cls()
        except Exception as exc:
            print(f"  ERROR initialising {name}: {exc}")
            continue
        metrics = evaluate_system(
            system_name=name,
            predict_fn=sys_obj.predict,
            test_cases=TEST_CASES,
            top_k=top_k,
        )
        results[name] = metrics
        _print_metrics(metrics)

    # Delta summary
    print(f"\n  {'-'*50}")
    print("  Component delta (vs bm25 baseline):")
    base = results.get("bm25")
    if base:
        for name in ("flat", "no_keyword", "full"):
            m = results.get(name)
            if m:
                print(
                    f"    {name:<12}  Top-1 d={m.top1_accuracy - base.top1_accuracy:+.2%}"
                    f"  Kappa d={m.cohen_kappa - base.cohen_kappa:+.4f}"
                )

    if "full" in results and "no_keyword" in results:
        f  = results["full"]
        nk = results["no_keyword"]
        print(f"\n  Keyword-hint contribution (full - no_keyword):")
        print(f"    Top-1 d={f.top1_accuracy - nk.top1_accuracy:+.2%}  "
              f"Kappa d={f.cohen_kappa - nk.cohen_kappa:+.4f}")

    if "full" in results and "no_beam" in results:
        f  = results["full"]
        nb = results["no_beam"]
        print(f"\n  Beam-search contribution (full - no_beam):")
        print(f"    Top-1 d={f.top1_accuracy - nb.top1_accuracy:+.2%}  "
              f"Kappa d={f.cohen_kappa - nb.cohen_kappa:+.4f}")

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def run_approach_comparison(top_k: int = _DEFAULT_TOP_K) -> dict[str, SystemMetrics]:
    """
    Evaluate all four approaches for the thesis approach-comparison table.

    Approaches
    ----------
    no_rag       : Pure LLM, zero retrieval (baseline)
    standard_rag : Flat dense vector search (standard RAG)
    bm25         : BM25 sparse keyword retrieval (keyword baseline)
    hierarchical : 4-stage hierarchical pipeline (thesis contribution)

    Note: Fine-tuned model results are sourced from published literature
    (Gweon et al. 2017; Boselli et al. 2018) and shown separately in
    the thesis table, not evaluated here due to training-data requirements.
    """
    systems: list[tuple[str, type]] = [
        ("no_rag",       NoRAGBaseline),
        ("standard_rag", FlatVectorBaseline),
        ("bm25",         BM25Baseline),
        ("hierarchical", HierarchicalRAG),
    ]
    results: dict[str, SystemMetrics] = {}
    for name, cls in systems:
        print(f"\n{'='*60}\nApproach: {name.upper()}\n{'='*60}")
        try:
            sys_obj = cls()
        except Exception as exc:
            print(f"  ERROR initialising {name}: {exc}")
            continue
        metrics = evaluate_system(
            system_name=name,
            predict_fn=sys_obj.predict,
            test_cases=TEST_CASES,
            top_k=top_k,
        )
        results[name] = metrics
        _print_metrics(metrics)
    return results


# Scalability labels per approach (static thesis analysis)
_SCALABILITY: dict[str, str] = {
    "no_rag":       "N/A",
    "bm25":         "Limited",
    "standard_rag": "Limited",
    "fine_tuned":   "Complex",
    "hierarchical": "Excellent",
}

# Hallucination risk label based on hallucination_count / n_evaluated
def _halluc_label(count: int, n: int) -> str:
    if n == 0:
        return "Unknown"
    rate = count / n
    if rate > 0.15:
        return "HIGH"
    if rate > 0.05:
        return "MODERATE"
    if rate > 0.01:
        return "LOW"
    return "VERY LOW"


def print_thesis_approach_table(results: dict[str, SystemMetrics]) -> None:
    """
    Print the thesis Approach Comparison Table with actual measured values.

    Columns: Approach | Top-1 | Top-3 | Hallucinations | Kappa | Latency | Scalability
    """
    W = 90
    print()
    print("=" * W)
    print("  THESIS TABLE: Approach Comparison -- ISCO-08 Classification (n=100)")
    print("=" * W)

    # Header
    print(
        f"  {'Approach':<22} {'Top-1':>7} {'Top-3':>7} "
        f"{'Halluc':>10} {'Kappa':>7} {'Latency':>10} {'Scalability':<12} {'Risk Label'}"
    )
    print(f"  {'-'*88}")

    # Rows from measured results
    order = ["no_rag", "bm25", "standard_rag", "hierarchical"]
    base_acc = None
    for name in order:
        m = results.get(name)
        if m is None:
            continue
        if base_acc is None and m.n_evaluated > 0:
            base_acc = m.top1_accuracy  # first successful row is baseline
        halluc_lbl = _halluc_label(m.hallucination_count, m.n_evaluated)
        scale      = _SCALABILITY.get(name, "-")
        if m.n_evaluated == 0:
            print(
                f"  {name:<22} {'N/A':>6}  {'N/A':>7}  "
                f"  {'err':>3}/{m.errors:<4} "
                f"  {'N/A':>7}  {'N/A':>7}ms {scale:<12} N/A (all errors)"
            )
        else:
            print(
                f"  {name:<22} {m.top1_accuracy:>6.1%} {m.top3_accuracy:>7.1%} "
                f"  {m.hallucination_count:>3}/{m.n_evaluated:<4} "
                f"{m.cohen_kappa:>7.3f} {m.avg_latency_ms:>8.1f}ms {scale:<12} {halluc_lbl}"
            )

    # Fine-tuned reference row from literature
    print(
        f"  {'fine_tuned (lit.)*':<22} {'~45%':>6s} {'~65%':>7s} "
        f"  {'est.':>8s}  {'~0.72':>7s} {'~5-50ms':>9s} {'Complex':<12} MODERATE"
    )
    print(f"  {'-'*88}")

    # Relative improvement over No-RAG baseline
    no_rag = results.get("no_rag")
    hier   = results.get("hierarchical")
    std    = results.get("standard_rag")
    if no_rag and no_rag.top1_accuracy > 0:
        print()
        print("  Relative accuracy improvement over No-RAG baseline:")
        for name, m in results.items():
            if name == "no_rag":
                continue
            delta_pp  = (m.top1_accuracy - no_rag.top1_accuracy) * 100
            delta_pct = (m.top1_accuracy - no_rag.top1_accuracy) / max(no_rag.top1_accuracy, 1e-9) * 100
            sign      = "+" if delta_pp >= 0 else ""
            print(f"    {name:<20}: {sign}{delta_pp:.1f}pp Top-1  ({sign}{delta_pct:.1f}% relative)")

    # Hallucination highlight
    print()
    print("  Hallucination (invalid ISCO-08 codes produced by each system):")
    for name in order:
        m = results.get(name)
        if m:
            rate = m.hallucination_count / max(m.n_evaluated, 1)
            bar  = "#" * m.hallucination_count + "." * (20 - min(m.hallucination_count, 20))
            print(f"    {name:<20}: {m.hallucination_count:>3} / {m.n_evaluated}  ({rate:.1%})  [{bar}]")

    # Scalability summary
    print()
    print("  Scalability analysis:")
    print("    No RAG          : N/A      -- No index; latency grows with model size")
    print("    BM25            : Limited  -- Index is flat; no multilingual embeddings")
    print("    Standard RAG    : Limited  -- Single flat collection; no hierarchy")
    print("    Fine-tuned      : Complex  -- Requires labeled corpus; retraining per language")
    print("    Hierarchical RAG: Excellent-- 4-layer index; O(log N) search; language-agnostic")
    print()
    print("  * Literature estimates: Gweon et al. (2017) ISCO fine-tuned classifier;")
    print("    Boselli et al. (2018) job-ad classification with fine-tuned embeddings.")
    print("=" * W)


def print_comparison_table(results: dict[str, SystemMetrics]) -> None:
    """Print a side-by-side comparison table."""
    systems = list(results.values())
    if not systems:
        print("No results to display.")
        return

    header = f"{'Metric':<22} | " + " | ".join(f"{s.system_name:>14}" for s in systems)
    sep    = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    rows = [
        ("Top-1 Accuracy",   lambda s: f"{s.top1_accuracy:.2%}"),
        ("Top-3 Accuracy",   lambda s: f"{s.top3_accuracy:.2%}"),
        ("Cohen's Kappa",    lambda s: f"{s.cohen_kappa:.4f}"),
        ("HITL Rate",        lambda s: f"{s.hitl_rate:.2%}"),
        ("Avg Latency (ms)", lambda s: f"{s.avg_latency_ms:.1f}"),
        ("P95 Latency (ms)", lambda s: f"{s.p95_latency_ms:.1f}"),
        ("P99 Latency (ms)", lambda s: f"{s.p99_latency_ms:.1f}"),
        ("N Evaluated",      lambda s: str(s.n_evaluated)),
    ]
    for label, fmt in rows:
        row = f"{label:<22} | " + " | ".join(f"{fmt(s):>14}" for s in systems)
        print(row)
    print(sep)


def save_results_csv(
    results: dict[str, SystemMetrics],
    csv_path: str = _CSV_DEFAULT,
) -> None:
    """Write one row per system to a CSV file for thesis tables."""
    fields = [
        "system_name", "top1_accuracy", "top3_accuracy",
        "cohen_kappa", "hitl_rate", "avg_latency_ms",
        "p95_latency_ms", "p99_latency_ms",
        "n_evaluated", "errors",
    ]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in results.values():
            writer.writerow({k: getattr(m, k) for k in fields})
    print(f"\nResults saved to: {csv_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Evaluate ISCO-08 classification systems for UAE LFS thesis."
    )
    parser.add_argument(
        "--system", choices=["bm25", "flat", "hierarchical", "all"],
        default="all",
    )
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path.")
    parser.add_argument("--csv", type=str, default=_CSV_DEFAULT,
                        help="CSV output path.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study instead of full comparison.")
    parser.add_argument("--arabic", action="store_true",
                        help="Evaluate on Arabic-only test corpus (30 Gulf/MSA cases).")
    parser.add_argument("--approach-compare", action="store_true",
                        help="Run thesis approach comparison: No-RAG vs BM25 vs Standard RAG vs Hierarchical RAG.")
    args = parser.parse_args()

    if args.approach_compare:
        results = run_approach_comparison(top_k=args.top_k)
        print_thesis_approach_table(results)
        save_results_csv(results, args.csv.replace(".csv", "_approach.csv"))
    elif args.ablation:
        results = run_ablation(top_k=args.top_k)
        print_comparison_table(results)
    elif args.arabic:
        selected = None if args.system == "all" else [args.system]
        results  = run_comparison(top_k=args.top_k, systems=selected,
                                  test_cases=ARABIC_TEST_CASES)
        print("\n  [Arabic-only evaluation — 30 Gulf Arabic + MSA test cases]")
        print_comparison_table(results)
        save_results_csv(results, args.csv.replace(".csv", "_arabic.csv"))
    else:
        selected = None if args.system == "all" else [args.system]
        results  = run_comparison(top_k=args.top_k, systems=selected)
        print_comparison_table(results)
        save_results_csv(results, args.csv)

    if args.output:
        import json
        from dataclasses import asdict
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"JSON results saved to: {args.output}")


if __name__ == "__main__":
    main()
