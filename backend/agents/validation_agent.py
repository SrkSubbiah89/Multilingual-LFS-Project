"""
backend/agents/validation_agent.py

Two-stage LFS survey response validator.

Stage 1 — Rule-based checks (pure Python, no LLM)
    Fast, deterministic consistency checks applied in order:

    R01  Required fields — employment_status and industry always required;
         job_title / hours_per_week / employment_type required only when employed.
    R02  hours_per_week must be a parseable number.
    R03  hours_per_week must be in [1, 168].
    R04  hours_per_week > 80 → warning (extreme outlier).
    R05  Unemployed respondent has a non-empty job_title → contradiction.
    R06  Unemployed respondent has hours_per_week > 0 → contradiction.
    R07  Not-in-labour-force respondent has a non-empty job_title → contradiction.
    R08  full_time employment with fewer than 20 hours/week → warning.
    R09  part_time employment with 35 or more hours/week → warning.
    R10  Employed respondent reported 0 hours/week → error.

Stage 2 — Semantic validation (Claude 3.5 Sonnet via CrewAI)
    Runs only when stage 1 produces no ERROR-level violations.
    The LLM checks:
      • Whether the job_title plausibly belongs to the stated industry.
      • Whether the full response set contains hidden contradictions.
      • Whether the answers are realistic for a labour-force respondent.

Output
------
ValidationResult
    is_valid          True when no ERROR rules fail and semantic check passes.
    confidence        Float in [0, 1]; starts at 1.0, penalised per violation.
    rule_violations   List of RuleViolation objects from stage 1.
    semantic_issues   Plain-text list from stage 2 (empty if stage 2 skipped).
    explanation_en    Human-readable verdict in English.
    explanation_ar    Human-readable verdict in Arabic.
    validated_data    Normalised copy of the input (hours cast to int, etc.).

Confidence penalties
    error violation    −0.25 each
    warning violation  −0.10 each
    semantic issue     −0.10 each
    LLM confidence     stage-1 score × llm_confidence (when stage 2 runs)

Usage
-----
from backend.agents.validation_agent import ValidationAgent

agent = ValidationAgent()
result = agent.validate({
    "employment_status": "employed",
    "job_title":         "software engineer",
    "industry":          "technology",
    "hours_per_week":    "40",
    "employment_type":   "full_time",
})
print(result.is_valid)       # True
print(result.confidence)     # ~0.95
print(result.explanation_en)
print(result.explanation_ar)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from crewai import Agent, Crew, Task
from pydantic import BaseModel, Field

from backend.llm import TaskType, get_llm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Always required regardless of employment status
_ALWAYS_REQUIRED: frozenset[str] = frozenset({"employment_status", "industry"})

# Required only when the respondent is employed
_EMPLOYMENT_REQUIRED: frozenset[str] = frozenset({
    "job_title", "hours_per_week", "employment_type"
})

# Statuses that indicate active employment
_EMPLOYED_STATUSES: frozenset[str] = frozenset({"employed"})

# Statuses where job/hours fields should be absent or empty
_INACTIVE_STATUSES: frozenset[str] = frozenset({
    "unemployed", "not_in_labour_force"
})

# Absolute bounds on weekly hours
_HOURS_MIN = 1
_HOURS_MAX = 168
_HOURS_EXTREME = 80          # warning threshold
_FULL_TIME_MIN = 20          # below this → warning for full_time
_PART_TIME_MAX = 35          # at or above this → warning for part_time

# LLM confidence penalties per violation type
_PENALTY_ERROR   = 0.25
_PENALTY_WARNING = 0.10
_PENALTY_SEMANTIC = 0.10


# ---------------------------------------------------------------------------
# Intermediate rule result (internal use only)
# ---------------------------------------------------------------------------

@dataclass
class _RuleResult:
    rule_id: str
    field: str
    severity: str       # "error" | "warning"
    message_en: str
    message_ar: str


# ---------------------------------------------------------------------------
# Public output models
# ---------------------------------------------------------------------------

class RuleViolation(BaseModel):
    """A single rule-based validation finding from stage 1."""

    rule_id: str                    # e.g. "R03"
    field: str                      # which response field triggered the rule
    severity: str                   # "error" | "warning"
    message_en: str                 # human-readable finding (English)
    message_ar: str                 # human-readable finding (Arabic)


class ValidationResult(BaseModel):
    """Structured result returned by ValidationAgent.validate()."""

    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rule_violations: list[RuleViolation]
    semantic_issues: list[str]          # plain-text issues from LLM
    explanation_en: str                 # overall verdict in English
    explanation_ar: str                 # overall verdict in Arabic
    validated_data: dict                # normalised copy of input responses


# ---------------------------------------------------------------------------
# LLM semantic prompt
# ---------------------------------------------------------------------------

_SEMANTIC_INSTRUCTIONS = """\
You are a Labour Force Survey (LFS) data quality expert. Your task is to \
perform semantic consistency checks on a completed survey response.

Check for:
1. Whether the job_title plausibly belongs to the stated industry or sector.
2. Whether the employment_type and hours_per_week are consistent with the \
   job_title (e.g. a surgeon working 8 h/week would be suspicious).
3. Any other hidden contradictions or implausible combinations in the full \
   response set.

Be lenient — flag only genuine inconsistencies, not minor stylistic variations.

Return ONLY a valid JSON object — no markdown fences, no extra text:
{
  "is_semantically_consistent": true | false,
  "confidence": <float 0.0–1.0>,
  "issues": ["<issue 1>", ...],
  "explanation_en": "<one-paragraph verdict in English>",
  "explanation_ar": "<one-paragraph verdict in Arabic>"
}
"""


# ---------------------------------------------------------------------------
# ValidationAgent
# ---------------------------------------------------------------------------

class ValidationAgent:
    """
    Two-stage LFS survey response validator.

    Stage 1 runs fast rule-based checks (no API calls).
    Stage 2 runs a Claude 3.5 Sonnet semantic check only when stage 1 passes.

    Attributes
    ----------
    RULE_IDS : list[str]
        All rule identifiers applied in stage 1, in order.
    """

    RULE_IDS: list[str] = ["R01", "R02", "R03", "R04", "R05",
                            "R06", "R07", "R08", "R09", "R10"]

    def __init__(self) -> None:
        self._llm = get_llm(TaskType.CRITICAL)   # Claude 3.5 Sonnet, temp 0.0
        self._agent = Agent(
            role="LFS Survey Data Quality Specialist",
            goal=(
                "Identify semantic inconsistencies and hidden contradictions in "
                "Labour Force Survey responses, and produce clear bilingual "
                "explanations (English and Arabic) that help interviewers "
                "understand the quality of the collected data."
            ),
            backstory=(
                "You are a senior data quality analyst at a national statistics "
                "office. You have reviewed hundreds of thousands of Labour Force "
                "Survey responses in both English and Arabic. You know the "
                "typical employment patterns of the region, understand common "
                "response errors, and can spot contradictions that rule-based "
                "systems miss — such as a job title that doesn't match the "
                "stated industry or implausible hours for a given occupation."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        responses: dict,
        language: str = "en",
    ) -> ValidationResult:
        """
        Validate a set of LFS survey responses.

        Parameters
        ----------
        responses : dict
            Dictionary of field → value pairs.  Expected keys:
            employment_status, job_title, industry, hours_per_week,
            employment_type.  Extra keys are preserved in validated_data.
        language : str
            "en" or "ar" — used to tailor the semantic explanation.

        Returns
        -------
        ValidationResult
            Always returns a result.  Check ``is_valid`` and
            ``rule_violations`` for actionable details.
        """
        # Normalise input
        normalised = self._normalise(responses)

        # Stage 1: rule-based checks
        raw_violations = self._run_rules(normalised)
        rule_violations = [
            RuleViolation(
                rule_id=r.rule_id,
                field=r.field,
                severity=r.severity,
                message_en=r.message_en,
                message_ar=r.message_ar,
            )
            for r in raw_violations
        ]

        errors   = [v for v in rule_violations if v.severity == "error"]
        warnings = [v for v in rule_violations if v.severity == "warning"]

        # Stage 2: semantic validation — only when stage 1 is error-free
        semantic_issues: list[str] = []
        llm_valid      = True
        llm_confidence = 1.0
        llm_exp_en     = ""
        llm_exp_ar     = ""

        if not errors:
            llm_valid, llm_confidence, semantic_issues, llm_exp_en, llm_exp_ar = (
                self._semantic_validate(normalised, language)
            )

        # Compute overall confidence
        confidence = 1.0
        confidence -= len(errors)   * _PENALTY_ERROR
        confidence -= len(warnings) * _PENALTY_WARNING
        confidence -= len(semantic_issues) * _PENALTY_SEMANTIC
        if not errors:
            confidence *= llm_confidence
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        is_valid = (len(errors) == 0) and llm_valid

        # Build bilingual verdict
        explanation_en, explanation_ar = self._build_explanation(
            is_valid, errors, warnings, semantic_issues, llm_exp_en, llm_exp_ar
        )

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            rule_violations=rule_violations,
            semantic_issues=semantic_issues,
            explanation_en=explanation_en,
            explanation_ar=explanation_ar,
            validated_data=normalised,
        )

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalise(self, responses: dict) -> dict:
        """
        Return a cleaned copy of the input:
        - Strip whitespace from all string values.
        - Lowercase employment_status and employment_type for consistent matching.
        - Cast hours_per_week to int where possible (keep original string on fail).
        """
        out = {}
        for k, v in responses.items():
            sv = str(v).strip() if v is not None else ""
            out[k] = sv

        for key in ("employment_status", "employment_type"):
            if key in out:
                out[key] = out[key].lower()

        if "hours_per_week" in out:
            try:
                out["hours_per_week"] = str(int(float(out["hours_per_week"])))
            except (ValueError, TypeError):
                pass   # leave as-is; R02 will flag it

        return out

    # ------------------------------------------------------------------
    # Stage 1: rule-based checks
    # ------------------------------------------------------------------

    def _run_rules(self, data: dict) -> list[_RuleResult]:
        """Run all rule checks in order and collect violations."""
        violations: list[_RuleResult] = []
        violations.extend(self._r01_required_fields(data))
        violations.extend(self._r02_hours_numeric(data))
        violations.extend(self._r03_hours_range(data))
        violations.extend(self._r04_hours_extreme(data))
        violations.extend(self._r05_r06_unemployed_consistency(data))
        violations.extend(self._r07_nilf_consistency(data))
        violations.extend(self._r08_r09_r10_hours_type_consistency(data))
        return violations

    def _r01_required_fields(self, data: dict) -> list[_RuleResult]:
        """R01 — Required fields must be present and non-empty."""
        violations: list[_RuleResult] = []
        status = data.get("employment_status", "")

        required = set(_ALWAYS_REQUIRED)
        if status in _EMPLOYED_STATUSES:
            required |= _EMPLOYMENT_REQUIRED

        for f in sorted(required):
            if not data.get(f, "").strip():
                violations.append(_RuleResult(
                    rule_id="R01",
                    field=f,
                    severity="error",
                    message_en=(
                        f"Required field '{f}' is missing or empty."
                    ),
                    message_ar=(
                        f"الحقل المطلوب '{f}' مفقود أو فارغ."
                    ),
                ))
        return violations

    def _r02_hours_numeric(self, data: dict) -> list[_RuleResult]:
        """R02 — hours_per_week must be parseable as a number."""
        raw = data.get("hours_per_week", "")
        if not raw:
            return []
        try:
            float(raw)
            return []
        except ValueError:
            return [_RuleResult(
                rule_id="R02",
                field="hours_per_week",
                severity="error",
                message_en=(
                    f"'hours_per_week' value '{raw}' is not a valid number."
                ),
                message_ar=(
                    f"قيمة ساعات العمل الأسبوعية '{raw}' ليست رقمًا صحيحًا."
                ),
            )]

    def _r03_hours_range(self, data: dict) -> list[_RuleResult]:
        """R03 — hours_per_week must be within [1, 168]."""
        raw = data.get("hours_per_week", "")
        if not raw:
            return []
        try:
            hours = float(raw)
        except ValueError:
            return []   # R02 already handles this
        if hours < _HOURS_MIN or hours > _HOURS_MAX:
            return [_RuleResult(
                rule_id="R03",
                field="hours_per_week",
                severity="error",
                message_en=(
                    f"'hours_per_week' ({hours:.0f}) must be between "
                    f"{_HOURS_MIN} and {_HOURS_MAX}."
                ),
                message_ar=(
                    f"ساعات العمل الأسبوعية ({hours:.0f}) يجب أن تكون بين "
                    f"{_HOURS_MIN} و{_HOURS_MAX}."
                ),
            )]
        return []

    def _r04_hours_extreme(self, data: dict) -> list[_RuleResult]:
        """R04 — hours_per_week > 80 is a potential data-entry outlier."""
        raw = data.get("hours_per_week", "")
        if not raw:
            return []
        try:
            hours = float(raw)
        except ValueError:
            return []
        if hours > _HOURS_EXTREME:
            return [_RuleResult(
                rule_id="R04",
                field="hours_per_week",
                severity="warning",
                message_en=(
                    f"'hours_per_week' ({hours:.0f}) is unusually high "
                    f"(> {_HOURS_EXTREME}). Please verify with the respondent."
                ),
                message_ar=(
                    f"ساعات العمل الأسبوعية ({hours:.0f}) مرتفعة بشكل غير معتاد "
                    f"(أكثر من {_HOURS_EXTREME}). يرجى التحقق مع المشارك."
                ),
            )]
        return []

    def _r05_r06_unemployed_consistency(self, data: dict) -> list[_RuleResult]:
        """R05/R06 — Unemployed respondents should not have job_title or hours > 0."""
        violations: list[_RuleResult] = []
        if data.get("employment_status", "") != "unemployed":
            return violations

        if data.get("job_title", "").strip():
            violations.append(_RuleResult(
                rule_id="R05",
                field="job_title",
                severity="error",
                message_en=(
                    "Respondent is unemployed but a 'job_title' was provided. "
                    "This contradicts the stated employment status."
                ),
                message_ar=(
                    "المشارك عاطل عن العمل لكن تم تقديم 'مسمى وظيفي'. "
                    "هذا يتعارض مع حالة التوظيف المذكورة."
                ),
            ))

        raw_hours = data.get("hours_per_week", "")
        if raw_hours:
            try:
                if float(raw_hours) > 0:
                    violations.append(_RuleResult(
                        rule_id="R06",
                        field="hours_per_week",
                        severity="error",
                        message_en=(
                            f"Respondent is unemployed but 'hours_per_week' "
                            f"is {float(raw_hours):.0f}."
                        ),
                        message_ar=(
                            f"المشارك عاطل عن العمل لكن ساعات العمل الأسبوعية "
                            f"هي {float(raw_hours):.0f}."
                        ),
                    ))
            except ValueError:
                pass
        return violations

    def _r07_nilf_consistency(self, data: dict) -> list[_RuleResult]:
        """R07 — Not-in-labour-force respondents should not have a job_title."""
        if data.get("employment_status", "") != "not_in_labour_force":
            return []
        if data.get("job_title", "").strip():
            return [_RuleResult(
                rule_id="R07",
                field="job_title",
                severity="error",
                message_en=(
                    "Respondent is not in the labour force but a 'job_title' "
                    "was provided. This contradicts the stated status."
                ),
                message_ar=(
                    "المشارك خارج سوق العمل لكن تم تقديم 'مسمى وظيفي'. "
                    "هذا يتعارض مع الحالة المذكورة."
                ),
            )]
        return []

    def _r08_r09_r10_hours_type_consistency(
        self, data: dict
    ) -> list[_RuleResult]:
        """R08/R09/R10 — Cross-check hours_per_week with employment_type."""
        violations: list[_RuleResult] = []
        emp_type = data.get("employment_type", "")
        status   = data.get("employment_status", "")
        raw      = data.get("hours_per_week", "")

        if not raw or not emp_type:
            return violations
        try:
            hours = float(raw)
        except ValueError:
            return violations

        # R10 — checked first, before the range guard, because hours==0 is less
        # than _HOURS_MIN (1) and would be swallowed by the range guard otherwise.
        if hours == 0 and status in _EMPLOYED_STATUSES:
            violations.append(_RuleResult(
                rule_id="R10",
                field="hours_per_week",
                severity="error",
                message_en=(
                    "Respondent is employed but reported 0 hours worked per week."
                ),
                message_ar=(
                    "المشارك موظف لكنه أفاد بصفر ساعات عمل أسبوعيًا."
                ),
            ))
            return violations  # R03 already flags 0 as out-of-range; skip R08/R09

        # Skip R08/R09 when hours are already flagged as out-of-range by R03
        if hours < _HOURS_MIN or hours > _HOURS_MAX:
            return violations

        if emp_type == "full_time" and hours < _FULL_TIME_MIN:
            violations.append(_RuleResult(
                rule_id="R08",
                field="hours_per_week",
                severity="warning",
                message_en=(
                    f"Full-time employment with only {hours:.0f} hours/week "
                    f"is unusual (expected ≥ {_FULL_TIME_MIN})."
                ),
                message_ar=(
                    f"العمل بدوام كامل مع {hours:.0f} ساعات أسبوعيًا فقط "
                    f"أمر غير معتاد (المتوقع ≥ {_FULL_TIME_MIN} ساعة)."
                ),
            ))

        if emp_type == "part_time" and hours >= _PART_TIME_MAX:
            violations.append(_RuleResult(
                rule_id="R09",
                field="hours_per_week",
                severity="warning",
                message_en=(
                    f"Part-time employment with {hours:.0f} hours/week may "
                    f"actually be full-time (threshold: {_PART_TIME_MAX} h)."
                ),
                message_ar=(
                    f"العمل بدوام جزئي مع {hours:.0f} ساعات أسبوعيًا قد يكون "
                    f"في الواقع دوامًا كاملًا (الحد: {_PART_TIME_MAX} ساعة)."
                ),
            ))

        return violations

    # ------------------------------------------------------------------
    # Stage 2: semantic validation
    # ------------------------------------------------------------------

    def _semantic_validate(
        self,
        data: dict,
        language: str,
    ) -> tuple[bool, float, list[str], str, str]:
        """
        Run the CrewAI Claude agent for semantic consistency checking.

        Returns
        -------
        (is_valid, llm_confidence, issues, explanation_en, explanation_ar)
        """
        lang_note = {
            "ar": "The survey was conducted in Arabic.",
        }.get(language, "The survey was conducted in English.")

        response_block = "\n".join(
            f"  {k}: {v}" for k, v in sorted(data.items()) if v
        )

        task = Task(
            description=(
                f"{_SEMANTIC_INSTRUCTIONS}\n\n"
                f"Language note: {lang_note}\n\n"
                f"Survey responses:\n{response_block}"
            ),
            expected_output=(
                'JSON: {"is_semantically_consistent": bool, "confidence": float, '
                '"issues": [str, ...], "explanation_en": str, "explanation_ar": str}'
            ),
            agent=self._agent,
        )

        crew    = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw_out = str(crew.kickoff()).strip()

        return self._parse_semantic_response(raw_out)

    def _parse_semantic_response(
        self, raw: str
    ) -> tuple[bool, float, list[str], str, str]:
        """
        Parse the LLM JSON output for semantic validation.

        Returns (is_valid, confidence, issues, explanation_en, explanation_ar).
        Falls back to a safe "valid with low confidence" result on any parse error.
        """
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

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

        is_valid   = bool(data.get("is_semantically_consistent", True))
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        issues     = [str(i) for i in data.get("issues", []) if str(i).strip()]
        exp_en     = str(data.get("explanation_en", "")).strip()
        exp_ar     = str(data.get("explanation_ar", "")).strip()

        return is_valid, confidence, issues, exp_en, exp_ar

    # ------------------------------------------------------------------
    # Verdict construction
    # ------------------------------------------------------------------

    def _build_explanation(
        self,
        is_valid: bool,
        errors: list[RuleViolation],
        warnings: list[RuleViolation],
        semantic_issues: list[str],
        llm_exp_en: str,
        llm_exp_ar: str,
    ) -> tuple[str, str]:
        """Build final human-readable verdict strings in both languages."""
        if is_valid and not warnings and not semantic_issues:
            en = (
                "All responses passed validation. "
                "The employment data is internally consistent and plausible."
            )
            ar = (
                "اجتازت جميع الإجابات عملية التحقق. "
                "بيانات التوظيف متسقة داخليًا ومعقولة."
            )
            if llm_exp_en:
                en = f"{en} {llm_exp_en}"
            if llm_exp_ar:
                ar = f"{ar} {llm_exp_ar}"
            return en, ar

        parts_en: list[str] = []
        parts_ar: list[str] = []

        if errors:
            err_list_en = "; ".join(e.message_en for e in errors)
            err_list_ar = "؛ ".join(e.message_ar for e in errors)
            parts_en.append(
                f"{len(errors)} error(s) found: {err_list_en}"
            )
            parts_ar.append(
                f"تم العثور على {len(errors)} خطأ(أخطاء): {err_list_ar}"
            )

        if warnings:
            warn_list_en = "; ".join(w.message_en for w in warnings)
            warn_list_ar = "؛ ".join(w.message_ar for w in warnings)
            parts_en.append(
                f"{len(warnings)} warning(s): {warn_list_en}"
            )
            parts_ar.append(
                f"{len(warnings)} تحذير(ات): {warn_list_ar}"
            )

        if semantic_issues:
            sem_list_en = "; ".join(semantic_issues)
            parts_en.append(f"Semantic issues: {sem_list_en}")
            parts_ar.append(f"مشكلات دلالية: {sem_list_en}")

        if llm_exp_en:
            parts_en.append(llm_exp_en)
        if llm_exp_ar:
            parts_ar.append(llm_exp_ar)

        verdict_en = "Validation failed. " if not is_valid else "Validation passed with warnings. "
        verdict_ar = "فشل التحقق. " if not is_valid else "نجح التحقق مع تحذيرات. "

        return (
            verdict_en + " ".join(parts_en),
            verdict_ar + " ".join(parts_ar),
        )
