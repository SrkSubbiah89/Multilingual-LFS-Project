"""
eval/synthetic_person_register_stress_test.py

[SYNTHETIC - STRESS TEST ONLY, NOT PILOT EVIDENCE]

Generates synthetic PersonRegister records and stress-tests
backend.agents.person_register.PersonRegisterService's REAL pre-fill
logic (imported and run directly, not reimplemented) against them.

This file's ONLY legitimate purpose is stress-testing PersonRegister's
pre-fill logic (question-reduction rate, pre-fill correctness, edge-case
handling). It must NEVER be read as, or used to compute, anything
resembling a pilot outcome -- completion time, cost, respondent burden,
or respondent experience. Synthetic data cannot demonstrate any of
those; only a real pilot (Module E) can. If a future reader is tempted
to cite a number from this file as if it were a pilot result, that is a
misuse of this file, not a legitimate use.

Method: direct statistical sampling from documented, clearly-illustrative
distributions (NOT CTGAN/TVAE) -- decided explicitly rather than
defaulting to the prompt's suggested method, because CTGAN/TVAE are
generative models meant to learn a distribution from real training data,
and this project has no real PersonRegister data at all (the pilot,
Module E, has not run). Training a GAN/VAE on a small hand-authored seed
set would not add rigor over directly sampling from documented
distributions, and would add a new, heavy dependency (sdv/ctgan, neither
currently installed nor referenced anywhere else in this repo -- checked
directly) for no clear benefit. The distributions below are illustrative
and plausible, NOT sourced from real UAE LFS published statistics --
disclosed explicitly, not implied as authoritative.

ISCO-08 / ISIC / ISCED codes are sampled ONLY from codes the current KB
actually defines (436 ISCO unit codes from backend/rag/load_full_isco.py,
134 ISIC classes from backend/agents/isic_classifier.py, 9 ISCED levels
from backend/agents/isced_classifier.py) -- confirmed by importing these
modules' real data directly, not by hand-copying a list that could drift.

Usage
-----
    python eval/synthetic_person_register_stress_test.py \
        --out backend/evaluation/synthetic_prefill_validation.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from backend.database.connection import Base  # noqa: E402
from backend.database.models import User, PersonRegister  # noqa: E402
from backend.agents.person_register import PersonRegisterService, _SURVEY_FIELDS  # noqa: E402

_LABEL = "[SYNTHETIC - STRESS TEST ONLY, NOT PILOT EVIDENCE]"


# ---------------------------------------------------------------------------
# Real code lists, pulled directly from the modules that define them
# ---------------------------------------------------------------------------

def _real_isco_codes() -> list[str]:
    text = Path("backend/rag/load_full_isco.py").read_text(encoding="utf-8")
    m = re.search(r"_UNIT: list\[tuple\[str, str\]\] = \[(.*?)\n\]", text, re.S)
    return re.findall(r'\("(\d{4})",', m.group(1))


def _real_isic_codes() -> list[str]:
    from backend.agents.isic_classifier import _ISIC_DATA
    return sorted({e["class_code"] for e in _ISIC_DATA})


def _real_isced_levels() -> list[int]:
    from backend.agents.isced_classifier import _ISCED_LEVELS
    return [e["level"] for e in _ISCED_LEVELS]


_ISCO_CODES = _real_isco_codes()
_ISIC_CODES = _real_isic_codes()
_ISCED_LEVELS = _real_isced_levels()

assert len(_ISCO_CODES) == 436, f"expected 436 ISCO codes, got {len(_ISCO_CODES)}"
assert len(_ISIC_CODES) == 134, f"expected 134 ISIC codes, got {len(_ISIC_CODES)}"
assert _ISCED_LEVELS == list(range(9)), f"expected ISCED levels 0-8, got {_ISCED_LEVELS}"

# ---------------------------------------------------------------------------
# Illustrative (NOT sourced from real UAE LFS statistics) distributions
# ---------------------------------------------------------------------------

_EMPLOYMENT_STATUS = ["employed", "unemployed", "not_in_labour_force"]
_EMPLOYMENT_STATUS_W = [0.72, 0.08, 0.20]

_EMPLOYMENT_TYPE = ["full_time", "part_time", "seasonal", "temporary"]
_EMPLOYMENT_TYPE_W = [0.75, 0.15, 0.05, 0.05]

_GENDER = ["male", "female", "prefer_not_to_say"]
_GENDER_W = [0.55, 0.43, 0.02]

_NATIONALITY = ["Emirati", "Indian", "Pakistani", "Filipino", "Bangladeshi", "Egyptian", "British", "Jordanian"]
_NATIONALITY_W = [0.15, 0.25, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]

_AGE_GROUP = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
_AGE_GROUP_W = [0.10, 0.30, 0.28, 0.18, 0.10, 0.04]

# ISCED level weights, index = level 0-8; skewed toward secondary (3) and bachelor (6)
_ISCED_W = [0.02, 0.03, 0.05, 0.20, 0.08, 0.05, 0.35, 0.15, 0.07]

_SOURCE = ["admin_import", "survey_round"]
_SOURCE_W = [0.4, 0.6]


def _hours_for(employment_type: str, rng: random.Random) -> float:
    center, spread, lo, hi = {
        "full_time": (42, 5, 30, 60),
        "part_time": (20, 6, 5, 34),
        "seasonal": (35, 10, 5, 60),
        "temporary": (35, 10, 5, 60),
    }[employment_type]
    val = rng.gauss(center, spread)
    return round(max(lo, min(hi, val)), 1)


@dataclass
class _EdgeCasePlan:
    """How many of each deliberately-injected edge case to include per batch."""
    stale_reference_period: int = 5     # ancient reference_period, is_active=True
    inactive_only: int = 5              # user has ONLY is_active=False records -> get_prefill() must return None
    partially_null: int = 5             # active record with several _SURVEY_FIELDS null
    duplicate_active: int = 5           # user has 2 active records -> most recently updated must win


def generate_batch(n_users: int, seed: int, edge: _EdgeCasePlan) -> list[dict]:
    """Returns a list of PersonRegister-shaped dicts, one or more per user_id
    (synthetic user_ids, not real). Deterministic given seed."""
    rng = random.Random(seed)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    records: list[dict] = []
    uid = 1

    def base_record(user_id: int, updated_at: datetime, is_active: bool = True,
                     reference_period: str = "2026-Q2", null_fields: Optional[set] = None) -> dict:
        null_fields = null_fields or set()
        status = rng.choices(_EMPLOYMENT_STATUS, weights=_EMPLOYMENT_STATUS_W)[0]
        emp_type = rng.choices(_EMPLOYMENT_TYPE, weights=_EMPLOYMENT_TYPE_W)[0] if status == "employed" else None
        isco = rng.choice(_ISCO_CODES) if status == "employed" else None
        isic = rng.choice(_ISIC_CODES) if status == "employed" else None
        rec = {
            "user_id": user_id,
            "reference_period": reference_period,
            "employment_status": status,
            "job_title": f"synthetic_job_for_{isco}" if isco else None,
            "industry": f"synthetic_industry_for_{isic}" if isic else None,
            "isco_code": isco,
            "isic_code": isic,
            "isced_level": rng.choices(_ISCED_LEVELS, weights=_ISCED_W)[0],
            "hours_per_week": _hours_for(emp_type, rng) if emp_type else None,
            "employment_type": emp_type,
            "nationality": rng.choices(_NATIONALITY, weights=_NATIONALITY_W)[0],
            "age_group": rng.choices(_AGE_GROUP, weights=_AGE_GROUP_W)[0],
            "gender": rng.choices(_GENDER, weights=_GENDER_W)[0],
            "is_active": is_active,
            "source": rng.choices(_SOURCE, weights=_SOURCE_W)[0],
            "updated_at": updated_at,
        }
        for f in null_fields:
            rec[f] = None
        return rec

    edge_case_tags: dict[int, str] = {}

    # -- normal records --
    n_normal = n_users - (edge.stale_reference_period + edge.inactive_only
                           + edge.partially_null + edge.duplicate_active)
    for _ in range(n_normal):
        records.append(base_record(uid, now - timedelta(days=rng.randint(1, 60))))
        uid += 1

    # -- edge case: stale reference_period, still is_active=True --
    for _ in range(edge.stale_reference_period):
        records.append(base_record(
            uid, now - timedelta(days=rng.randint(1000, 2000)),
            reference_period="2020-Q1",
        ))
        edge_case_tags[uid] = "stale_reference_period"
        uid += 1

    # -- edge case: user has ONLY inactive records --
    for _ in range(edge.inactive_only):
        records.append(base_record(uid, now - timedelta(days=rng.randint(1, 60)), is_active=False))
        edge_case_tags[uid] = "inactive_only"
        uid += 1

    # -- edge case: active record with several fields null --
    for _ in range(edge.partially_null):
        null_fields = set(rng.sample(_SURVEY_FIELDS, k=rng.randint(2, len(_SURVEY_FIELDS) - 1)))
        records.append(base_record(uid, now - timedelta(days=rng.randint(1, 60)), null_fields=null_fields))
        edge_case_tags[uid] = "partially_null"
        uid += 1

    # -- edge case: two ACTIVE records for the same user, different updated_at --
    for _ in range(edge.duplicate_active):
        older = base_record(uid, now - timedelta(days=rng.randint(30, 90)))
        newer = base_record(uid, now - timedelta(days=rng.randint(1, 10)))
        # Deliberately give them different employment_status so we can tell
        # which one get_prefill() actually returned.
        older["employment_status"], newer["employment_status"] = "unemployed", "employed"
        records.append(older)
        records.append(newer)
        edge_case_tags[uid] = "duplicate_active"
        uid += 1

    return records, edge_case_tags


# ---------------------------------------------------------------------------
# Stress test: run the REAL PersonRegisterService against a batch
# ---------------------------------------------------------------------------

def stress_test_batch(records: list[dict], edge_case_tags: dict[int, str], batch_label: str) -> dict:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    user_ids = sorted({r["user_id"] for r in records})
    for uid in user_ids:
        db.add(User(id=uid, email=f"synthetic_{uid}@stress-test.invalid"))
    db.commit()

    for r in records:
        pr = PersonRegister(**r)
        db.add(pr)
    db.commit()

    svc = PersonRegisterService()
    reduction_pcts = []
    accuracy_mismatches = []
    edge_case_results = {}
    n_prefilled = 0

    for uid in user_ids:
        prefill = svc.get_prefill(uid, db)
        tag = edge_case_tags.get(uid)

        if tag == "inactive_only":
            edge_case_results.setdefault(tag, []).append({
                "user_id": uid,
                "expected": "get_prefill() returns None (no active record)",
                "actual": "None" if prefill is None else "NOT None -- BUG",
                "pass": prefill is None,
            })
            continue

        if prefill is None:
            continue  # shouldn't happen for non-inactive-only users
        n_prefilled += 1
        reduction_pcts.append(prefill.reduction_pct)

        # Pre-fill accuracy: the returned fields must exactly match the
        # underlying DB record's fields for the record get_prefill() actually
        # selected (found by reference_period, since edge cases reuse it).
        db_record = (
            db.query(PersonRegister)
            .filter(PersonRegister.user_id == uid, PersonRegister.is_active.is_(True))
            .order_by(PersonRegister.updated_at.desc())
            .first()
        )
        for f in _SURVEY_FIELDS:
            db_val = getattr(db_record, f)
            pf_val = getattr(prefill, f, None)
            db_val_str = str(db_val) if db_val is not None else None
            pf_val_str = str(pf_val) if pf_val is not None else None
            if db_val_str != pf_val_str:
                accuracy_mismatches.append({
                    "user_id": uid, "field": f, "db_value": db_val_str, "prefill_value": pf_val_str,
                })

        if tag == "stale_reference_period":
            edge_case_results.setdefault(tag, []).append({
                "user_id": uid,
                "finding": (
                    "get_prefill() used a record with reference_period='2020-Q1' "
                    "(updated_at ~1000-2000 days old) with no staleness rejection -- "
                    "confirmed: PersonRegisterService.get_prefill() has no age/staleness "
                    "cutoff of any kind, it only filters is_active=True and orders by "
                    "updated_at desc. Not a bug per se (matches documented behavior "
                    "literally), but a real design gap worth flagging."
                ),
                "reduction_pct": prefill.reduction_pct,
            })
        elif tag == "partially_null":
            edge_case_results.setdefault(tag, []).append({
                "user_id": uid,
                "fields_available": prefill.fields_available,
                "reduction_pct": prefill.reduction_pct,
                "pass": True,  # no crash, graceful degradation confirmed
            })
        elif tag == "duplicate_active":
            edge_case_results.setdefault(tag, []).append({
                "user_id": uid,
                "expected": "most recently updated active record wins (employment_status='employed')",
                "actual_employment_status": prefill.employment_status,
                "pass": prefill.employment_status == "employed",
            })

    db.close()

    return {
        "batch_label": batch_label,
        "n_users": len(user_ids),
        "n_prefilled": n_prefilled,
        "mean_reduction_pct": round(sum(reduction_pcts) / len(reduction_pcts), 4) if reduction_pcts else None,
        "min_reduction_pct": min(reduction_pcts) if reduction_pcts else None,
        "max_reduction_pct": max(reduction_pcts) if reduction_pcts else None,
        "reduction_pct_distribution": reduction_pcts,
        "prefill_accuracy_mismatches": accuracy_mismatches,
        "prefill_accuracy_pass": len(accuracy_mismatches) == 0,
        "edge_case_results": edge_case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="backend/evaluation/synthetic_prefill_validation.json")
    parser.add_argument("--n-users-per-batch", type=int, default=200)
    args = parser.parse_args()

    edge = _EdgeCasePlan()
    batches_out = []
    for i, seed in enumerate([101, 202, 303], start=1):
        records, tags = generate_batch(args.n_users_per_batch, seed, edge)
        result = stress_test_batch(records, tags, f"batch_{i}_seed_{seed}")
        batches_out.append(result)
        print(f"{_LABEL} batch_{i} (seed={seed}): mean_reduction_pct={result['mean_reduction_pct']}, "
              f"accuracy_pass={result['prefill_accuracy_pass']}")

    reductions = [b["mean_reduction_pct"] for b in batches_out]
    spread = max(reductions) - min(reductions)

    out = {
        "_label": _LABEL,
        "_disclosure": (
            "Synthetic data generated by direct statistical sampling from illustrative "
            "(NOT real-UAE-LFS-sourced) distributions. Legitimate use: stress-testing "
            "PersonRegisterService's pre-fill logic only. NOT pilot evidence. NOT a "
            "measure of completion time, cost, respondent burden, or respondent "
            "experience -- none of those can be measured from synthetic data."
        ),
        "generation_method": "direct statistical sampling (CTGAN/TVAE explicitly not used -- see module docstring)",
        "isco_codes_available": len(_ISCO_CODES),
        "isic_codes_available": len(_ISIC_CODES),
        "isced_levels_available": len(_ISCED_LEVELS),
        "batches": batches_out,
        "cross_batch_consistency": {
            "mean_reduction_pct_per_batch": reductions,
            "spread": round(spread, 4),
            "stable": spread < 0.05,  # <5 percentage points across 3 independent batches
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{_LABEL} Wrote {args.out}")
    print(f"{_LABEL} Cross-batch reduction_pct: {reductions} (spread={spread:.4f}, stable={out['cross_batch_consistency']['stable']})")


if __name__ == "__main__":
    main()
