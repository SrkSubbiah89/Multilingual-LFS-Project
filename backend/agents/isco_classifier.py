"""
backend/agents/isco_classifier.py

Two-stage ISCO-08 occupation classifier for the LFS survey.

Pipeline
--------
Stage 1 – Hierarchical semantic search (HierarchicalISCOStore)
    Traverses four Qdrant collections — major groups → sub-major groups →
    minor groups → unit groups — using filtered vector search at each level.
    Returns the best unit-group code and a weighted confidence score.
    Falls back automatically to the flat ``isco_occupations`` collection if
    the hierarchical collections are not yet populated.

Stage 2 – LLM re-ranking (Claude 3.5 Sonnet, TaskType.CRITICAL)
    A CrewAI agent picks the single best candidate from the top-3 unit groups
    and provides one-sentence reasoning.
    Skipped when the top hierarchical match is unambiguous (≥ 0.92).

HITL gate
---------
When the final confidence is below ``HITL_THRESHOLD`` (0.70) the result
carries ``hitl_required=True`` so downstream quality-management workflows
can route the classification to a human reviewer.

Usage
-----
from backend.agents.isco_classifier import ISCOClassifier

clf    = ISCOClassifier()
result = clf.classify("software engineer")
result = clf.classify("مهندس برمجيات")
result = clf.classify("I fix broken pipes", context="construction sector")

print(result.primary.code)           # e.g. "2512"
print(result.primary.title_en)       # "Software Developers"
print(result.primary.confidence)     # 0.8741
print(result.method)                 # "hierarchical_llm" | "hierarchical_semantic"
                                     # | "flat_llm" | "flat_semantic"
print(result.hitl_required)          # False
print(result.hierarchy_path)         # ["2", "25", "251", "2512"]
print(result.stage_confidences)      # {"stage1": 0.91, "stage2": 0.88, ...}
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from crewai import Agent, Crew, Task
from pydantic import BaseModel

from backend.llm import TaskType, get_llm, get_llm_strict
from backend.rag import OccupationMatch, get_vector_store
from backend.rag.hierarchical_store import (
    HierarchicalResult,
    UnitCandidate,
    get_hierarchical_store,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword → ISCO major group pre-filter
# ---------------------------------------------------------------------------
# Maps common job-title tokens to the correct 1-digit ISCO-08 major group.
# This anchors stage 1 of the hierarchical search so semantic drift cannot
# place "chef" inside "Professionals" (group 2) because of the French meaning
# of the word, for example.
#
# Rules:
#   - All keys are lowercase, no punctuation.
#   - A job title is tokenised to lowercase words; each word is looked up.
#   - If ≥1 word matches, the most-frequently-matched major group wins.
#   - Ties go to the entry that appears earliest in the title.
# ---------------------------------------------------------------------------

_MAJOR_KEYWORD_MAP: dict[str, str] = {
    # =========================================================
    # Group 1: Managers / المديرون
    # =========================================================
    "ceo": "1", "coo": "1", "cfo": "1", "cto": "1", "cmo": "1",
    "director": "1", "manager": "1", "executive": "1",
    "president": "1", "vp": "1", "vice president": "1",
    "superintendent": "1", "administrator": "1",
    "principal": "1",          # school principal → 1345
    "commissioner": "1", "governor": "1", "minister": "1",
    "chairman": "1", "chairperson": "1",
    "chief": "1",              # chief executive / chief officer
    "head of": "1",
    "general manager": "1", "country manager": "1",
    "operations manager": "1", "branch manager": "1",
    "hotel manager": "1", "restaurant manager": "1",
    "retail manager": "1", "store manager": "1",
    "project manager": "1", "program manager": "1",
    "product manager": "1", "portfolio manager": "1",
    "marketing manager": "1", "sales manager": "1",
    "hr manager": "1", "human resources manager": "1",
    "finance manager": "1", "it manager": "1",
    "managing director": "1",
    # AR
    "مدير": "1", "رئيس": "1", "مسؤول": "1", "مدير عام": "1",
    "مدير تنفيذي": "1", "رئيس تنفيذي": "1",

    # =========================================================
    # Group 2: Professionals / المهنيون
    # =========================================================
    # Engineering
    "engineer": "2", "engineering": "2",
    "civil engineer": "2", "mechanical engineer": "2",
    "electrical engineer": "2", "chemical engineer": "2",
    "structural engineer": "2", "software engineer": "2",
    "network engineer": "2", "systems engineer": "2",
    "biomedical engineer": "2", "petroleum engineer": "2",
    "aerospace engineer": "2", "marine engineer": "2",
    "architect": "2", "urban planner": "2", "surveyor": "2",
    "designer": "2", "interior designer": "2",
    # Medicine
    "doctor": "2", "physician": "2", "surgeon": "2",
    "specialist": "2", "consultant physician": "2",
    "gp": "2", "general practitioner": "2",
    "radiologist": "2", "anaesthesiologist": "2", "anesthesiologist": "2",
    "cardiologist": "2", "neurologist": "2", "oncologist": "2",
    "orthopedic": "2", "dermatologist": "2", "psychiatrist": "2",
    "ophthalmologist": "2", "urologist": "2",
    "dentist": "2", "dental surgeon": "2",
    "pharmacist": "2", "veterinarian": "2", "vet": "2",
    "physiotherapist": "2", "occupational therapist": "2",
    "psychologist": "2", "counsellor": "2",
    "dietitian": "2", "nutritionist": "2", "optometrist": "2",
    # Law
    "lawyer": "2", "attorney": "2", "solicitor": "2",
    "barrister": "2", "judge": "2", "legal counsel": "2",
    "notary": "2",
    # Finance / Business Professionals
    "accountant": "2", "auditor": "2", "actuary": "2",
    "economist": "2", "statistician": "2", "financial analyst": "2",
    "investment analyst": "2", "tax consultant": "2",
    # IT Professionals
    "programmer": "2", "developer": "2", "coder": "2",
    "software developer": "2", "web developer": "2",
    "mobile developer": "2", "app developer": "2",
    "data scientist": "2", "data engineer": "2",
    "machine learning": "2", "ai engineer": "2",
    "devops": "2", "cloud engineer": "2", "security analyst": "2",
    "cybersecurity": "2", "database administrator": "2",
    "dba": "2", "systems analyst": "2", "business analyst": "2",
    # Teaching / Research
    "professor": "2", "lecturer": "2", "associate professor": "2",
    "researcher": "2", "scientist": "2", "geologist": "2",
    "biologist": "2", "chemist": "2", "physicist": "2",
    "mathematician": "2",
    # Other Professionals
    "journalist": "2", "editor": "2", "author": "2", "writer": "2",
    "translator": "2", "interpreter": "2",
    "social worker": "2", "counselor": "2",
    "librarian": "2", "archivist": "2",
    # AR
    "مهندس": "2", "طبيب": "2", "جراح": "2",
    "محامي": "2", "قاضي": "2",
    "محاسب": "2", "مراجع": "2",
    "مطور": "2", "مبرمج": "2", "باحث": "2",
    "أستاذ": "2", "محاضر": "2",
    "صيدلاني": "2", "طبيب بيطري": "2",
    "مهندس معماري": "2",

    # =========================================================
    # Group 3: Technicians & Associate Professionals
    # =========================================================
    "technician": "3", "tech": "3",
    "lab technician": "3", "laboratory technician": "3",
    "medical technician": "3", "x-ray technician": "3",
    "dental technician": "3", "pharmacy technician": "3",
    "paramedic": "3", "ambulance": "3", "emt": "3",
    "nurse": "3", "registered nurse": "3", "rn": "3",
    "enrolled nurse": "3", "midwife": "3",
    "draughtsman": "3", "draftsman": "3", "cad": "3",
    "it support": "3", "helpdesk": "3", "support engineer": "3",
    "network technician": "3", "radio technician": "3",
    "broadcast technician": "3", "sound technician": "3",
    "electrician technician": "3",
    "inspector": "3", "quality inspector": "3", "quality control": "3",
    "supervisor": "3", "foreman": "3", "overseer": "3",
    "broker": "3", "insurance broker": "3",
    "real estate agent": "3", "property agent": "3",
    "travel agent": "3", "customs agent": "3",
    "fitness trainer": "3", "personal trainer": "3",
    "sports coach": "3", "coach": "3",
    "photographer": "3", "cameraman": "3",
    "social media": "3",
    # AR
    "فني": "3", "ممرض": "3", "مشرف": "3", "مراقب": "3",
    "مفتش": "3", "وكيل": "3",

    # =========================================================
    # Group 4: Clerical Support / الدعم الكتابي
    # =========================================================
    "clerk": "4", "secretary": "4", "receptionist": "4",
    "typist": "4", "bookkeeper": "4", "teller": "4",
    "data entry": "4", "data entry clerk": "4",
    "office clerk": "4", "general clerk": "4",
    "administrative assistant": "4", "admin assistant": "4",
    "personal assistant": "4", "pa": "4",
    "executive assistant": "4",
    "customer service": "4", "call centre": "4", "call center": "4",
    "front desk": "4", "front office": "4",
    "payroll clerk": "4", "accounts clerk": "4",
    "hr assistant": "4", "recruitment coordinator": "4",
    "filing clerk": "4", "records clerk": "4",
    "library clerk": "4", "mail clerk": "4",
    "bank clerk": "4", "loan officer": "4",
    "cashier": "4",           # retail cashier → 4 (office) vs 5 resolved by context
    # AR
    "كاتب": "4", "سكرتير": "4", "موظف": "4",
    "مساعد إداري": "4", "موظف استقبال": "4",

    # =========================================================
    # Group 5: Service and Sales / الخدمات والمبيعات
    # =========================================================
    # Food service
    "chef": "5", "cook": "5", "kitchen": "5",
    "sous chef": "5", "head chef": "5", "executive chef": "5",
    "pastry chef": "5", "pastry cook": "5", "confectioner": "5",
    "baker": "5", "bread maker": "5",
    "barista": "5", "coffee maker": "5",
    "waiter": "5", "waitress": "5", "server": "5",
    "bartender": "5", "mixologist": "5",
    "line cook": "5", "prep cook": "5", "grill cook": "5",
    "sushi chef": "5", "bbq": "5",
    "dishwasher": "5", "kitchen helper": "5",
    "butcher": "5", "fishmonger": "5",
    "food service": "5", "catering": "5",
    # Personal services
    "hairdresser": "5", "beautician": "5", "barber": "5",
    "stylist": "5", "hair stylist": "5", "nail technician": "5",
    "makeup artist": "5", "cosmetician": "5", "esthetician": "5",
    "massage therapist": "5", "spa therapist": "5",
    "tattoo artist": "5",
    # Sales
    "salesperson": "5", "sales representative": "5",
    "sales executive": "5", "sales associate": "5",
    "shop assistant": "5", "retail assistant": "5",
    "shopkeeper": "5", "merchant": "5",
    "insurance agent": "5",
    # Security / protection
    "security guard": "5", "guard": "5", "security officer": "5",
    "doorman": "5", "bouncer": "5",
    # Travel / hospitality services
    "flight attendant": "5", "cabin crew": "5", "steward": "5",
    "stewardess": "5", "air hostess": "5",
    "travel guide": "5", "tour guide": "5",
    "concierge": "5", "bellboy": "5", "porter hotel": "5",
    # Care
    "nanny": "5", "babysitter": "5", "childminder": "5",
    "caregiver": "5", "carer": "5", "home carer": "5",
    "housemaid": "5", "domestic worker": "5", "maid": "5",
    # AR
    "طاهٍ": "5", "طاهي": "5", "شيف": "5",
    "نادل": "5", "بائع": "5", "حارس": "5",
    "مضيف": "5", "حلاق": "5", "مصفف": "5",

    # =========================================================
    # Group 6: Agricultural / الزراعة
    # =========================================================
    "farmer": "6", "agriculture": "6", "agricultural worker": "6",
    "fisherman": "6", "fisher": "6", "aquaculture": "6",
    "horticulturist": "6", "gardener": "6", "landscaper": "6",
    "livestock": "6", "animal farmer": "6", "poultry farmer": "6",
    "dairy farmer": "6", "beekeeper": "6",
    "forester": "6", "forestry worker": "6", "lumberjack": "6",
    "crop farmer": "6", "rice farmer": "6",
    # AR
    "مزارع": "6", "صياد": "6", "بستاني": "6",

    # =========================================================
    # Group 7: Craft and Trades / الحرف اليدوية
    # =========================================================
    "plumber": "7", "pipefitter": "7",
    "electrician": "7", "electrical installer": "7",
    "welder": "7", "metalworker": "7",
    "carpenter": "7", "joiner": "7", "cabinetmaker": "7",
    "bricklayer": "7", "mason": "7", "stonemason": "7",
    "plasterer": "7", "tiler": "7", "floor layer": "7",
    "roofer": "7", "insulation worker": "7",
    "painter decorator": "7", "painter": "7",
    "glazier": "7", "window installer": "7",
    "mechanic": "7", "auto mechanic": "7", "car mechanic": "7",
    "motorcycle mechanic": "7",
    "air conditioning": "7", "hvac": "7", "ac technician": "7",
    "refrigeration": "7",
    "blacksmith": "7", "tool maker": "7",
    "tailor": "7", "seamstress": "7", "dressmaker": "7",
    "shoemaker": "7", "cobbler": "7",
    "jeweller": "7", "goldsmith": "7",
    "printer": "7", "bookbinder": "7",
    "upholsterer": "7",
    "maintenance technician": "7", "maintenance worker": "7",
    "handyman": "7",
    # AR
    "سباك": "7", "كهربائي": "7", "لحام": "7",
    "نجار": "7", "ميكانيكي": "7", "خياط": "7",
    "بناء": "7", "حداد": "7",

    # =========================================================
    # Group 8: Plant and Machine Operators / مشغلو الآلات
    # =========================================================
    "driver": "8", "chauffeur": "8",
    "truck driver": "8", "lorry driver": "8",
    "bus driver": "8", "coach driver": "8",
    "taxi driver": "8", "cab driver": "8", "uber driver": "8",
    "delivery driver": "8", "courier": "8",
    "forklift": "8", "forklift operator": "8",
    "crane operator": "8", "crane driver": "8",
    "excavator": "8", "bulldozer": "8", "grader": "8",
    "machine operator": "8", "plant operator": "8",
    "press operator": "8", "lathe operator": "8",
    "factory operator": "8", "production operator": "8",
    "assembler": "8", "production line": "8",
    "locomotive": "8", "train driver": "8",
    "ship": "8", "sailor": "8", "seafarer": "8",
    "packaging": "8", "packing machine": "8",
    "food machine operator": "8",
    # AR
    "سائق": "8", "مشغل": "8", "مشغل آلات": "8",

    # =========================================================
    # Group 9: Elementary / المهن الأولية
    # =========================================================
    "cleaner": "9", "janitor": "9", "sweeper": "9",
    "housekeeper": "9", "domestic cleaner": "9",
    "office cleaner": "9", "hotel cleaner": "9",
    "labourer": "9", "laborer": "9",
    "construction labourer": "9", "site labourer": "9",
    "general labourer": "9", "unskilled worker": "9",
    "helper": "9", "assistant helper": "9",
    "porter": "9", "luggage porter": "9",
    "packer": "9", "picker": "9", "warehouse worker": "9",
    "shelf stacker": "9", "shelf filler": "9",
    "messenger": "9", "errand boy": "9",
    "garbage collector": "9", "refuse worker": "9",
    "street cleaner": "9", "road sweeper": "9",
    "farm labourer": "9", "agricultural labourer": "9",
    "fast food worker": "9", "kitchen assistant": "9",
    "vending machine": "9",
    # AR
    "عامل": "9", "عمال": "9", "منظف": "9",
    "فراش": "9", "بواب": "9",
}

# Pre-index multi-word phrases (longest first so "general manager" beats "manager")
_MULTI_WORD_HINTS: list[tuple[str, str]] = sorted(
    [(k, v) for k, v in _MAJOR_KEYWORD_MAP.items() if " " in k],
    key=lambda x: -len(x[0]),
)

# Single-token index (pre-built from non-phrase entries only, for O(1) lookup)
_SINGLE_TOKEN_MAP: dict[str, str] = {
    k: v for k, v in _MAJOR_KEYWORD_MAP.items() if " " not in k
}


def _keyword_major_hint(job_title: str) -> str:
    """
    Return the most likely ISCO-08 1-digit major group code for *job_title*,
    or "" if no keyword matches.

    Checks multi-word phrases first (longest first), then single tokens.
    """
    lower = job_title.lower().strip()

    # Multi-word phrases (e.g. "sous chef", "flight attendant")
    for phrase, code in _MULTI_WORD_HINTS:
        if phrase in lower:
            return code

    # Single-word tokens (use pre-built single-token index)
    scores: dict[str, int] = {}
    for token in re.findall(r"[a-z\u0600-\u06ff]{2,}", lower):
        if token in _SINGLE_TOKEN_MAP:
            grp = _SINGLE_TOKEN_MAP[token]
            scores[grp] = scores.get(grp, 0) + 1

    if not scores:
        return ""
    return max(scores, key=scores.__getitem__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Top match at or above this score → skip LLM re-ranking.
_HIGH_CONFIDENCE_THRESHOLD = 0.92

# Below this floor the primary result should be treated as unreliable.
MIN_USABLE_CONFIDENCE = 0.35

# Confidence below which a human reviewer should inspect the result.
HITL_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class ISCOMatch(BaseModel):
    """A single ISCO-08 match (primary or alternative)."""
    code: str
    title_en: str
    title_ar: str
    confidence: float


class ISCOClassification(BaseModel):
    """Structured result from ``ISCOClassifier.classify()``."""

    query: str
    language: str                           # "en" | "ar" | "mixed" | "other"
    primary: ISCOMatch                      # best-matching ISCO occupation
    alternatives: list[ISCOMatch]           # up to 2 alternatives
    method: str                             # see module docstring
    stage_confidences: Optional[dict] = None  # {stage1, stage2, stage3, stage4}
    hierarchy_path: Optional[list] = None     # e.g. ["2", "25", "251", "2512"]
    hitl_required: bool = False
    reasoning: str


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ISCOClassifier:
    """
    Classify a free-text job title to an ISCO-08 occupation code.

    Attributes
    ----------
    MIN_USABLE_CONFIDENCE : float
        Confidence floor below which the primary result may be unreliable.
        Callers can check ``result.primary.confidence < MIN_USABLE_CONFIDENCE``
        and ask the respondent to clarify.
    HITL_THRESHOLD : float
        Confidence below which ``result.hitl_required`` is set to ``True``.
    """

    MIN_USABLE_CONFIDENCE = MIN_USABLE_CONFIDENCE
    HITL_THRESHOLD        = HITL_THRESHOLD

    def __init__(
        self,
        llm_temperature: Optional[float] = None,
        force_flat: bool = False,
        reranker_model: Optional[str] = None,
        disable_keyword_map: bool = False,
        beam: int = 2,
        stage1_mode: str = "description",
        reranker_candidates: int = 5,
        branch_collapse: bool = False,
        capture_pool_metadata: bool = False,
        enable_llm: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        llm_temperature : float, optional
            Override the re-ranking LLM's temperature (default: the
            TaskType.GENERAL default, 0.3). Existing callers that omit this
            see identical behaviour to before. Added for the evaluation
            harness (eval/run_eval.py), which needs temperature=0 for
            reproducible runs — this is an additive constructor parameter,
            not a change to default production behaviour.
        force_flat : bool, default False
            Skip hierarchical store initialisation and always use the flat
            single-stage vector store, even when the hierarchical
            collections are available. Existing callers that omit this
            (default False) see identical behaviour to before — the
            hierarchical store is still tried first as always. Added so
            the evaluation harness's "Flat RAG" baseline can reuse
            _classify_flat()'s real reranking logic (same
            _llm_select_from_candidates() call, same 0.92 threshold as the
            hierarchical path) instead of a separately-reimplemented flat
            baseline that would silently drift from production behaviour.
        reranker_model : str, optional
            Pin the re-ranking LLM to exactly this model (e.g.
            "ollama/llama3.2:1b" or "anthropic/claude-3-5-sonnet-20241022")
            via get_llm_strict() instead of get_llm(). get_llm() silently
            substitutes Claude for GENERAL tasks when Ollama is
            unreachable; get_llm_strict() never does — it raises instead,
            and that exception is NOT caught here (it propagates out of
            __init__), so a run against an unavailable pinned model aborts
            before any case is classified rather than silently answering
            with a different model for some subset of cases. Existing
            callers that omit this (default None) get the exact previous
            behaviour: get_llm(TaskType.GENERAL, ...), fallback-on-down,
            failures logged and swallowed (agent_available=False). Added
            for the evaluation harness, where every case in a run must be
            answered by the same, known model or the run's numbers
            describe a system that never existed end-to-end.
        disable_keyword_map : bool, default False
            When True, _keyword_major_hint() is never consulted -- stage 1
            always uses semantic retrieval over the major-group collection,
            regardless of what _MAJOR_KEYWORD_MAP would have returned.
            Nothing else changes: same beam widths, same top_k, same
            reranker, same 0.92 threshold. Existing callers that omit this
            (default False) see identical behaviour to before. Added for
            the evaluation harness's two-arm keyword-map measurement
            (eval/run_eval.py --disable-keyword-map) so stage-1 semantic
            retrieval can be measured on a full test set instead of the
            handful of cases that happen not to match any dictionary entry.
        beam : int, default 2
            Passed straight through to HierarchicalISCOStore.search(beam=).
            Existing callers that omit this see identical behaviour to
            before (default 2, matching the previous hardcoded value --
            no caller ever overrode it prior to this parameter existing).
        stage1_mode : str, default "description"
            Passed straight through to HierarchicalISCOStore.search(
            stage1_mode=). "description" is the previous, unchanged
            behaviour. See that method's docstring for "leaf_vote".
        reranker_candidates : int, default 5
            Passed straight through to HierarchicalISCOStore.search(
            reranker_candidates=). How many pooled candidates reach the
            reranker; see that method's docstring for the pooling fix.
        branch_collapse : bool, default False
            Passed straight through to HierarchicalISCOStore.search(
            branch_collapse=). True reproduces the pre-fix behaviour
            (winning branch only); default False is the fixed, pooled
            behaviour. See that method's docstring for the quantified
            defect this fixes (49/82 errors on the 130-case full set).
        capture_pool_metadata : bool, default False
            Passed straight through to HierarchicalISCOStore.search(
            capture_pool_metadata=). B2 instrumentation, observational
            only -- see that method's docstring. Existing callers that
            omit this (default False) see identical behaviour to before.
        enable_llm : bool, default True
            Task 09: when False, Stage 2 (LLM agent for re-ranking) is
            skipped entirely -- get_llm_strict(), get_llm(), and
            _build_reranker_agent() are never called, no LLM/agent is
            constructed, _agent_available stays False, and
            reranker_model_resolved is set to the explicit, non-model
            string "none (reranking disabled)" (never a fabricated model
            identity). classify() already only takes its LLM re-ranking
            branch when self._agent_available is True, so this alone makes
            every classify() call route through the existing semantic-only
            retrieval path -- no new retrieval algorithm, no fabricated
            reranking trace. Retrieval store initialisation (Stage 1a/1b)
            is unaffected either way. Existing callers that omit this
            (default True) see identical behaviour to before -- added for
            the evaluation harness's genuinely retrieval-only runs
            (eval/run_eval.py --use-llm-reranker off), which must not
            initialise an LLM at all, not just skip calling it.
        """
        self._disable_keyword_map   = disable_keyword_map
        self._beam                  = beam
        self._stage1_mode           = stage1_mode
        self._reranker_candidates   = reranker_candidates
        self._branch_collapse       = branch_collapse
        self._capture_pool_metadata = capture_pool_metadata
        self._agent_available       = False
        self._hierarchical_store    = None
        self._flat_store            = None
        self._llm_temperature       = llm_temperature
        self._reranker_model_pin    = reranker_model
        self.reranker_model_resolved = ""

        # Stage 1a: hierarchical vector store (preferred, unless force_flat)
        if not force_flat:
            try:
                self._hierarchical_store = get_hierarchical_store()
            except Exception as exc:
                _logger.warning(
                    "ISCOClassifier: hierarchical store unavailable (%s). "
                    "Will attempt flat store instead.",
                    exc,
                )

        # Stage 1b: flat vector store (fallback when hierarchical store fails
        # to initialise entirely — e.g. Qdrant completely unreachable — or
        # when force_flat=True was requested explicitly)
        if self._hierarchical_store is None:
            try:
                self._flat_store = get_vector_store()
            except Exception as exc:
                _logger.warning(
                    "ISCOClassifier: flat vector store also unavailable (%s). "
                    "ISCO classification will be unavailable until Qdrant is reachable.",
                    exc,
                )
                return  # no point loading LLM if there is no store at all

        # Stage 2: LLM agent for re-ranking
        if not enable_llm:
            # Retrieval-only mode (Task 09): never construct an LLM/agent.
            # _agent_available stays False (its __init__ default), so
            # classify() always takes the existing semantic-only path --
            # see this constructor's enable_llm docstring.
            self.reranker_model_resolved = "none (reranking disabled)"
        elif self._reranker_model_pin:
            # Strict mode (eval harness): no fallback, no swallowed
            # exception — an unavailable pinned model aborts __init__.
            self._llm = get_llm_strict(
                self._reranker_model_pin,
                temperature=self._llm_temperature if self._llm_temperature is not None else 0.3,
            )
            self._agent = self._build_reranker_agent()
            self._agent_available = True
            self.reranker_model_resolved = getattr(self._llm, "model", self._reranker_model_pin)
        else:
            # Default mode (production callers): graceful degradation —
            # unchanged from previous behaviour.
            try:
                self._llm   = get_llm(TaskType.GENERAL, temperature=self._llm_temperature)
                self._agent = self._build_reranker_agent()
                self._agent_available = True
                self.reranker_model_resolved = getattr(self._llm, "model", "")
            except Exception as exc:
                _logger.warning(
                    "ISCOClassifier: LLM agent unavailable (%s). "
                    "Falling back to semantic-only classification (no LLM re-ranking).",
                    exc,
                )

    def _build_reranker_agent(self) -> Agent:
        return Agent(
            role="ISCO-08 Occupation Classification Specialist",
            goal=(
                "Select the single most accurate ISCO-08 occupation code for a "
                "given job title from a shortlist of semantic search candidates. "
                "Prefer the most specific (unit-group) code the title clearly supports."
            ),
            backstory=(
                "You are an expert in the International Standard Classification "
                "of Occupations (ISCO-08) and have classified thousands of job "
                "titles for national statistics offices. You understand formal and "
                "informal job descriptions in both English and Arabic, including "
                "code-switched text common in the Arab world's labour market."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        job_title: str,
        context: str = "",
        language: str = "",
        top_k: int = 5,
        use_llm: bool = True,
        trace: Optional[dict] = None,
    ) -> ISCOClassification:
        """
        Classify a free-text job title to the best ISCO-08 occupation.

        Parameters
        ----------
        job_title : str
            Raw job title (English, Arabic, or code-switched).
        context : str
            Optional free-text context that aids disambiguation, e.g.
            "works in a hospital" or "construction sector, 10 years experience".
        language : str
            Optional ISO-639 hint ("en", "ar", "mixed").  When empty the
            script is detected automatically from *job_title*.
        top_k : int
            Number of unit-group candidates to retrieve in stage 4 (default 5).
        trace : dict, optional
            Instrumentation-only passthrough to
            HierarchicalISCOStore.search(trace=...) — see that method's
            docstring. No effect when omitted (default None).

        Returns
        -------
        ISCOClassification
            Always returns a result.  Check ``primary.confidence`` against
            ``MIN_USABLE_CONFIDENCE`` if you need to gate on reliability.
            Check ``hitl_required`` to decide whether a human reviewer is needed.
        """
        job_title = job_title.strip()
        if not job_title:
            return self._empty_result(job_title)

        if self._hierarchical_store is None and self._flat_store is None:
            raise RuntimeError("ISCOClassifier: no vector store initialised.")

        lang = language.strip() or _detect_script(job_title)

        # Keyword pre-filter → anchor major group before semantic search
        major_hint = "" if self._disable_keyword_map else _keyword_major_hint(job_title)
        if major_hint:
            _logger.debug(
                "ISCOClassifier: keyword hint major_group=%r for %r", major_hint, job_title
            )

        # ── Hierarchical path ──────────────────────────────────────────────
        if self._hierarchical_store is not None:
            return self._classify_hierarchical(
                job_title, context, lang, top_k, major_hint=major_hint, use_llm=use_llm, trace=trace,
            )

        # ── Flat fallback path (hierarchical store failed at init time, or
        #    force_flat=True was passed to __init__) ─────────────────────────
        return self._classify_flat(job_title, context, lang, top_k, use_llm=use_llm, trace=trace)

    # ------------------------------------------------------------------
    # Hierarchical classification
    # ------------------------------------------------------------------

    def _classify_hierarchical(
        self,
        job_title: str,
        context: str,
        lang: str,
        top_k: int,
        major_hint: str = "",
        use_llm: bool = True,
        trace: Optional[dict] = None,
    ) -> ISCOClassification:
        """Run the 4-stage hierarchical pipeline, then optionally re-rank with LLM.

        Context-enriched query embedding (thesis contribution):
        Rather than embedding only the bare job title, we concatenate any
        available contextual keywords (job duties, industry, sector) so the
        stage-4 unit-group cosine similarity is computed against a semantically
        richer representation.  The major-group keyword hint still anchors
        stage 1, preventing semantic drift regardless of the enriched text.
        """
        # Build enriched embedding query: job_title + context keywords.
        # Strip boilerplate tokens ("language=en") so they don't add noise.
        context_clean = " ".join(
            tok for tok in context.split()
            if not tok.startswith("language=")
        ).strip()
        enriched_query = f"{job_title} {context_clean}".strip() if context_clean else job_title

        h: HierarchicalResult = self._hierarchical_store.search(
            enriched_query, top_k=top_k, major_hint=major_hint,
            beam=self._beam, stage1_mode=self._stage1_mode,
            reranker_candidates=self._reranker_candidates,
            branch_collapse=self._branch_collapse,
            capture_pool_metadata=self._capture_pool_metadata, trace=trace,
        )

        if not h.code:
            return self._empty_result(job_title, lang)

        primary_match = ISCOMatch(
            code=h.code,
            title_en=h.label_en,
            title_ar=h.label_ar,
            confidence=h.confidence,
        )

        alternatives = [
            ISCOMatch(
                code=c.code,
                title_en=c.label_en,
                title_ar=c.label_ar,
                confidence=c.score,
            )
            for c in h.top_candidates[1:3]  # skip index 0 (that's the primary)
        ]

        method_prefix = "flat" if h.fallback_used else "hierarchical"

        # Fast path: unambiguous — skip LLM
        if h.confidence >= _HIGH_CONFIDENCE_THRESHOLD:
            if trace is not None:
                trace["reranker_fired"] = False
            return ISCOClassification(
                query=job_title,
                language=lang,
                primary=primary_match,
                alternatives=alternatives,
                method=f"{method_prefix}_semantic",
                stage_confidences=h.stage_confidences,
                hierarchy_path=h.hierarchy_path,
                hitl_required=h.hitl_required,
                reasoning=(
                    f"Unambiguous {'hierarchical' if not h.fallback_used else 'flat'} "
                    f"semantic match (score {h.confidence:.2%})."
                ),
            )

        # LLM re-ranking when available, enabled, AND confidence < 0.92
        if use_llm and self._agent_available and h.top_candidates:
            if trace is not None:
                trace["reranker_fired"] = True
                trace["reranker_input_candidates"] = [
                    {"code": c.code, "label_en": c.label_en, "score": c.score}
                    for c in h.top_candidates
                ]
            primary_match, reasoning = self._llm_select_from_candidates(
                job_title=job_title,
                candidates=h.top_candidates,
                context=context,
                lang=lang,
                stage_confidences=h.stage_confidences,
                trace=trace,
            )
            if trace is not None:
                trace["reranker_output"] = {"code": primary_match.code, "reasoning": reasoning}
            # Recalculate alternatives excluding the selected primary
            alternatives = [
                ISCOMatch(
                    code=c.code,
                    title_en=c.label_en,
                    title_ar=c.label_ar,
                    confidence=c.score,
                )
                for c in h.top_candidates[:3]
                if c.code != primary_match.code
            ][:2]
            method = f"{method_prefix}_llm"
        else:
            if trace is not None:
                trace["reranker_fired"] = False
            reasoning = (
                f"Top {'hierarchical' if not h.fallback_used else 'flat'} semantic match "
                f"(score {h.confidence:.2%})"
                + ("; LLM agent unavailable." if not self._agent_available else ".")
            )
            method = f"{method_prefix}_semantic"

        # Re-evaluate hitl after potential LLM reselection
        hitl = primary_match.confidence < HITL_THRESHOLD

        return ISCOClassification(
            query=job_title,
            language=lang,
            primary=primary_match,
            alternatives=alternatives,
            method=method,
            stage_confidences=h.stage_confidences,
            hierarchy_path=h.hierarchy_path,
            hitl_required=hitl,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Flat classification (init-level fallback only)
    # ------------------------------------------------------------------

    def _classify_flat(
        self,
        job_title: str,
        context: str,
        lang: str,
        top_k: int,
        use_llm: bool = True,
        trace: Optional[dict] = None,
    ) -> ISCOClassification:
        """
        Use the flat VectorStore when the hierarchical store failed to
        initialise (e.g. Qdrant is reachable but hierarchical collections
        are not yet created), or when __init__(force_flat=True) was used to
        request this path deliberately (the eval harness's "Flat RAG"
        baseline — single-stage retrieval, same reranking rule as the
        hierarchical path since both call _llm_select_from_candidates()).
        """
        if trace is not None:
            # This system has no stages 1-3 (single-stage retrieval) -- set
            # explicitly to None so the eval harness records them as JSON
            # null, distinguishable from the hierarchical path's "ran but
            # returned zero candidates" case, which is an empty list.
            trace["stage1"], trace["stage2"], trace["stage3"] = None, None, None
            trace["stage1_latency_ms"], trace["stage2_latency_ms"], trace["stage3_latency_ms"] = None, None, None
            # Flat retrieval never consults _MAJOR_KEYWORD_MAP (classify()
            # only threads major_hint into _classify_hierarchical()) -- every
            # flat case is genuine semantic retrieval, disclosed explicitly
            # so a reader comparing hierarchical vs flat accuracy can see
            # that hierarchical's stage1 sometimes wasn't retrieval at all.
            trace["stage1_source"] = "not_applicable (flat has no stage1; retrieval is always semantic)"

        t0 = time.perf_counter()
        candidates: list[OccupationMatch] = self._flat_store.search(
            job_title, top_k=top_k
        )
        if trace is not None:
            trace["stage4_latency_ms"] = (time.perf_counter() - t0) * 1000
            trace["stage4"] = [
                {"code": c.code, "label_en": c.title_en, "score": round(float(c.confidence), 4)}
                for c in candidates
            ]

        if not candidates:
            return self._empty_result(job_title, lang)

        best = candidates[0]
        primary_match = _occupation_match_to_isco(best)
        alternatives  = [_occupation_match_to_isco(c) for c in candidates[1:3]]

        from backend.rag.hierarchical_store import _infer_path
        hierarchy_path = _infer_path(best.code)

        flat_confidences = {
            "stage1": round(best.confidence, 4),
            "stage2": round(best.confidence, 4),
            "stage3": round(best.confidence, 4),
            "stage4": round(best.confidence, 4),
        }

        # Fast path
        if best.confidence >= _HIGH_CONFIDENCE_THRESHOLD:
            if trace is not None:
                trace["reranker_fired"] = False
            return ISCOClassification(
                query=job_title,
                language=lang,
                primary=primary_match,
                alternatives=alternatives,
                method="flat_semantic",
                stage_confidences=flat_confidences,
                hierarchy_path=hierarchy_path,
                hitl_required=best.confidence < HITL_THRESHOLD,
                reasoning=f"Unambiguous flat semantic match (score {best.confidence:.2%}).",
            )

        # LLM re-ranking
        if use_llm and self._agent_available:
            if trace is not None:
                trace["reranker_fired"] = True
                trace["reranker_input_candidates"] = trace["stage4"][:5]
            unit_candidates = [
                UnitCandidate(
                    code=c.code,
                    label_en=c.title_en,
                    label_ar=c.title_ar,
                    score=c.confidence,
                )
                for c in candidates[:5]
            ]
            primary_match, reasoning = self._llm_select_from_candidates(
                job_title=job_title,
                candidates=unit_candidates,
                context=context,
                lang=lang,
                stage_confidences=flat_confidences,
                trace=trace,
            )
            if trace is not None:
                trace["reranker_output"] = {"code": primary_match.code, "reasoning": reasoning}
            alternatives = [
                ISCOMatch(
                    code=c.code,
                    title_en=c.title_en,
                    title_ar=c.title_ar,
                    confidence=c.confidence,
                )
                for c in candidates[:3]
                if c.code != primary_match.code
            ][:2]
            method = "flat_llm"
        else:
            if trace is not None:
                trace["reranker_fired"] = False
            reasoning = (
                f"Top flat semantic match (score {best.confidence:.2%}); "
                "LLM agent unavailable."
            )
            method = "flat_semantic"

        return ISCOClassification(
            query=job_title,
            language=lang,
            primary=primary_match,
            alternatives=alternatives,
            method=method,
            stage_confidences=flat_confidences,
            hierarchy_path=hierarchy_path,
            hitl_required=primary_match.confidence < HITL_THRESHOLD,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # LLM re-ranking (shared by hierarchical and flat paths)
    # ------------------------------------------------------------------

    def _llm_select_from_candidates(
        self,
        job_title: str,
        candidates: list[UnitCandidate],
        context: str,
        lang: str,
        stage_confidences: dict,
        trace: Optional[dict] = None,
    ) -> tuple[ISCOMatch, str]:
        """Ask the re-ranking LLM to pick the best candidate from *candidates*.

        (Docstring said "Claude 3.5 Sonnet" but this agent is built with
        get_llm(TaskType.GENERAL) in __init__, which routes to Ollama by
        default and only falls back to Claude if Ollama is unreachable —
        corrected here; trace["reranker_model"] records which one actually
        answered for a given call, so this isn't left as an assumption.)
        """
        candidate_block = "\n".join(
            f"{i + 1}. [{c.code}] {c.label_en} / {c.label_ar}\n"
            f"   Semantic score: {c.score:.2%}"
            for i, c in enumerate(candidates)
        )

        lang_note = {
            "ar":    "The job title is written in Arabic.",
            "mixed": "The job title is code-switched (Arabic and English).",
        }.get(lang, "The job title is written in English.")

        context_line = f"\nAdditional context: {context}" if context.strip() else ""

        task = Task(
            description=(
                "You are an ISCO-08 classification specialist for a national "
                "Labour Force Survey.\n\n"
                f'Job title: "{job_title}"\n'
                f"{lang_note}{context_line}\n\n"
                f"Candidates (from hierarchical semantic search):\n{candidate_block}\n\n"
                "Select the single best ISCO-08 match. Prefer the most specific "
                "code (unit group over sub-major over major group) when the title "
                "clearly supports it.\n\n"
                "Return ONLY a valid JSON object — no markdown fences, no extra text:\n"
                '{"selected_code": "<isco_code>", "reasoning": "<one sentence>"}'
            ),
            expected_output=(
                'JSON: {"selected_code": "<code>", "reasoning": "<sentence>"}'
            ),
            agent=self._agent,
        )

        if trace is not None:
            trace["reranker_model"] = getattr(self._llm, "model", "unknown")
        try:
            crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
            raw  = str(crew.kickoff()).strip()
            if trace is not None:
                try:
                    usage = crew.calculate_usage_metrics()
                    trace["reranker_prompt_tokens"] = usage.prompt_tokens
                    trace["reranker_completion_tokens"] = usage.completion_tokens
                    trace["reranker_total_tokens"] = usage.total_tokens
                except Exception as _usage_exc:
                    _logger.debug("Could not read CrewAI usage metrics: %s", _usage_exc)
            return self._parse_llm_response(raw, candidates)
        except Exception as exc:
            _logger.warning("ISCO LLM re-ranking failed: %s. Using top candidate.", exc)
            if trace is not None:
                trace["reranker_error"] = f"{type(exc).__name__}: {exc}"
            top = candidates[0]
            return (
                ISCOMatch(
                    code=top.code,
                    title_en=top.label_en,
                    title_ar=top.label_ar,
                    confidence=top.score,
                ),
                "Fallback to top semantic match (LLM unavailable).",
            )

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(
        self,
        raw: str,
        candidates: list[UnitCandidate],
    ) -> tuple[ISCOMatch, str]:
        """
        Parse the LLM JSON response.

        Handles markdown fences and partial JSON.  Falls back to the top
        candidate if the response cannot be parsed or contains an
        unrecognised code.
        """
        code_map = {c.code: c for c in candidates}

        # Strip markdown fences
        clean = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL
        ).strip()

        data: dict = {}
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        selected_code = str(data.get("selected_code", "")).strip()
        reasoning     = str(data.get("reasoning", "")).strip()

        if selected_code in code_map:
            c = code_map[selected_code]
            return (
                ISCOMatch(
                    code=c.code,
                    title_en=c.label_en,
                    title_ar=c.label_ar,
                    confidence=c.score,
                ),
                reasoning or "Selected by LLM classifier.",
            )

        # Fallback: top candidate
        top = candidates[0]
        return (
            ISCOMatch(
                code=top.code,
                title_en=top.label_en,
                title_ar=top.label_ar,
                confidence=top.score,
            ),
            "Fallback to top semantic match (LLM response could not be parsed).",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(
        job_title: str = "", lang: str = "other"
    ) -> ISCOClassification:
        """Return a safe sentinel when no candidates are available."""
        placeholder = ISCOMatch(
            code="",
            title_en="Unknown",
            title_ar="غير معروف",
            confidence=0.0,
        )
        return ISCOClassification(
            query=job_title,
            language=lang,
            primary=placeholder,
            alternatives=[],
            method="flat_semantic",
            stage_confidences=None,
            hierarchy_path=None,
            hitl_required=True,
            reasoning="No candidates returned by the vector store.",
        )


# ---------------------------------------------------------------------------
# Script detection  (unchanged from original)
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)
_LATIN_RE = re.compile(r"[A-Za-z]+")


def _detect_script(text: str) -> str:
    """
    Return the dominant script of *text*.

    Returns
    -------
    "ar"    >= 90 % Arabic characters
    "en"    <= 10 % Arabic characters (mostly Latin)
    "mixed" between 10 % and 90 % Arabic
    "other" no alphabetic characters detected
    """
    ar    = sum(len(m.group()) for m in _ARABIC_RE.finditer(text))
    lat   = sum(len(m.group()) for m in _LATIN_RE.finditer(text))
    total = ar + lat
    if total == 0:
        return "other"
    ar_ratio = ar / total
    if ar_ratio >= 0.90:
        return "ar"
    if ar_ratio <= 0.10:
        return "en"
    return "mixed"


# ---------------------------------------------------------------------------
# Convenience: convert OccupationMatch → ISCOMatch
# ---------------------------------------------------------------------------

def _occupation_match_to_isco(m: OccupationMatch) -> ISCOMatch:
    return ISCOMatch(
        code=m.code,
        title_en=m.title_en,
        title_ar=m.title_ar,
        confidence=m.confidence,
    )
