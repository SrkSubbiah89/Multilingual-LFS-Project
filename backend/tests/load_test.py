"""
load_test.py — Comprehensive 100-user concurrent load test for the LFS Survey API.

Simulates real survey sessions across:
  • 5 languages  : en, ar, ar-gulf, ur, hi, tl
  • 3 LF paths   : employed, unemployed, not_in_labour_force
  • 3 difficulty : fast (minimal turns), normal (8-12 turns), full (complete survey)

Run:
    python -m backend.tests.load_test
    python -m backend.tests.load_test --users 50 --base http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL   = "http://localhost:8000"
N_USERS    = 100
TIMEOUT_S  = 30        # per-request timeout
RAMP_UP_S  = 10        # spread user starts over this window
THINK_MS   = (200, 800)  # random think time between turns (ms)

# ── Scenario definitions ───────────────────────────────────────────────────────
# Each scenario: list of user messages that drive the survey forward

SCENARIOS: dict[str, dict] = {

    # ── English scenarios ──────────────────────────────────────────────────────
    "en_employed_engineer": {
        "language": "en",
        "label": "EN · Employed · Software Engineer",
        "messages": [
            "Hello",
            "I am currently employed",
            "I have a bachelor's degree",
            "male",
            "indian",
            "married",
            "dubai",
            "5 years",
            "I work as a software engineer",
            "I develop mobile applications and web platforms",
            "information technology",
            "paid employee",
            "private sector",
            "40",
            "no secondary job",
            "full time",
            "10001 to 20000 AED per month",
            "yes I prefer AI interviewer",
            "very confident",
        ],
    },
    "en_employed_nurse": {
        "language": "en",
        "label": "EN · Employed · Nurse",
        "messages": [
            "Hi there",
            "employed",
            "bachelor degree",
            "female",
            "philippine",
            "single",
            "abu dhabi",
            "3 years",
            "registered nurse",
            "I care for patients in the ICU",
            "healthcare",
            "paid employee",
            "government",
            "48",
            "no",
            "full time",
            "5000 to 10000 AED",
            "no preference",
            "confident",
        ],
    },
    "en_employed_teacher": {
        "language": "en",
        "label": "EN · Employed · Teacher",
        "messages": [
            "Hello",
            "I work as a teacher",
            "master degree",
            "female",
            "egyptian",
            "married",
            "sharjah",
            "7 years",
            "secondary school teacher",
            "teaching mathematics and science",
            "education",
            "paid employee",
            "private sector",
            "35",
            "no",
            "full time",
            "10001 to 20000",
            "prefer AI",
            "very confident",
        ],
    },
    "en_employed_manager": {
        "language": "en",
        "label": "EN · Employed · Manager",
        "messages": [
            "Hello",
            "currently employed",
            "master degree",
            "male",
            "uae national",
            "married",
            "dubai",
            "10 years",
            "operations manager",
            "I manage daily operations across 3 departments",
            "manufacturing",
            "employer",
            "private",
            "50",
            "no",
            "full time",
            "above 20000 AED",
            "prefer AI",
            "very confident",
        ],
    },
    "en_unemployed_recent": {
        "language": "en",
        "label": "EN · Unemployed · Job Seeker",
        "messages": [
            "Hello",
            "I am currently unemployed",
            "bachelor degree",
            "male",
            "pakistani",
            "single",
            "dubai",
            "2 years",
            "yes I am actively searching",
            "online job portals and networking",
            "yes available immediately",
            "3 months",
            "I want a full time position",
            "yes I have worked before",
            "accountant",
            "private sector",
            "company downsizing",
            "under 5000 AED in last job",
            "prefer AI",
            "confident",
        ],
    },
    "en_unemployed_fresh_graduate": {
        "language": "en",
        "label": "EN · Unemployed · Fresh Graduate",
        "messages": [
            "Hi",
            "unemployed",
            "bachelor",
            "female",
            "indian",
            "single",
            "abu dhabi",
            "1 year",
            "yes searching",
            "campus recruitment and LinkedIn",
            "yes available",
            "6 months",
            "full time",
            "never worked before",
            "prefer human",
            "somewhat confident",
        ],
    },
    "en_olf_homemaker": {
        "language": "en",
        "label": "EN · Outside LF · Homemaker",
        "messages": [
            "Hello",
            "not in the labour force",
            "secondary school",
            "female",
            "uae national",
            "married",
            "sharjah",
            "all my life",
            "family responsibilities",
            "yes worked before",
            "housewife before marriage",
            "private",
            "10 years ago",
            "no preference",
            "confident",
        ],
    },
    "en_olf_student": {
        "language": "en",
        "label": "EN · Outside LF · Student",
        "messages": [
            "Hello",
            "I am a student not working",
            "secondary school still studying",
            "male",
            "emirati",
            "single",
            "dubai",
            "18 years",
            "studying full time",
            "never worked",
            "prefer AI",
            "very confident",
        ],
    },

    # ── Arabic scenarios ───────────────────────────────────────────────────────
    "ar_employed_engineer": {
        "language": "ar",
        "label": "AR · Employed · Engineer",
        "messages": [
            "مرحبا",
            "أنا موظف حالياً",
            "بكالوريوس",
            "ذكر",
            "إماراتي",
            "متزوج",
            "دبي",
            "5 سنوات",
            "مهندس برمجيات",
            "أطور تطبيقات الجوال وأنظمة الويب",
            "تكنولوجيا المعلومات",
            "موظف براتب",
            "القطاع الخاص",
            "40",
            "لا",
            "دوام كامل",
            "10001 إلى 20000 درهم",
            "أفضل الذكاء الاصطناعي",
            "واثق جداً",
        ],
    },
    "ar_employed_doctor": {
        "language": "ar",
        "label": "AR · Employed · Doctor",
        "messages": [
            "السلام عليكم",
            "أعمل حالياً",
            "دكتوراه في الطب",
            "ذكر",
            "مصري",
            "متزوج",
            "أبوظبي",
            "8 سنوات",
            "طبيب متخصص في الجراحة",
            "أجري عمليات جراحية وأعالج المرضى",
            "الرعاية الصحية",
            "موظف براتب",
            "حكومي",
            "55",
            "لا",
            "دوام كامل",
            "أكثر من 20000 درهم",
            "لا أهمية",
            "واثق جداً",
        ],
    },
    "ar_unemployed": {
        "language": "ar",
        "label": "AR · Unemployed · Job Seeker",
        "messages": [
            "مرحبا",
            "أنا عاطل عن العمل",
            "بكالوريوس محاسبة",
            "ذكر",
            "أردني",
            "أعزب",
            "دبي",
            "سنتان",
            "نعم أبحث بنشاط",
            "المواقع الإلكترونية والشبكات المهنية",
            "نعم متاح فوراً",
            "4 أشهر",
            "دوام كامل",
            "نعم عملت من قبل",
            "محاسب",
            "خاص",
            "انتهاء العقد",
            "5000 إلى 10000 درهم",
            "لا أهمية",
            "واثق",
        ],
    },

    # ── Gulf Arabic scenarios ──────────────────────────────────────────────────
    "gulf_ar_employed": {
        "language": "ar-gulf",
        "label": "Gulf AR · Employed · Officer",
        "messages": [
            "هلا والله",
            "أبي أقول إني أشتغل",
            "بكالوريوس",
            "ذكر",
            "إماراتي",
            "متزوج",
            "دبي",
            "6 سنوات",
            "ضابط شرطة",
            "أشتغل في حفظ الأمن وإدارة المرور",
            "الأمن والشرطة",
            "موظف حكومي",
            "حكومة",
            "45",
            "ما عندي شغلة ثانية",
            "دوام كامل",
            "10001 إلى 20000",
            "أفضل الذكاء الاصطناعي",
            "واثق جداً",
        ],
    },

    # ── Urdu scenarios ─────────────────────────────────────────────────────────
    "ur_employed": {
        "language": "ur",
        "label": "UR · Employed · Accountant",
        "messages": [
            "السلام علیکم",
            "میں ملازم ہوں",
            "بیچلر ڈگری",
            "مرد",
            "پاکستانی",
            "شادی شدہ",
            "دبئی",
            "4 سال",
            "اکاؤنٹنٹ",
            "میں مالی حسابات اور آڈٹ کرتا ہوں",
            "مالیاتی خدمات",
            "تنخواہ دار ملازم",
            "نجی شعبہ",
            "40",
            "نہیں",
            "فل ٹائم",
            "5000 سے 10000 درہم",
            "AI کو ترجیح دیتا ہوں",
            "پراعتماد",
        ],
    },
    "ur_unemployed": {
        "language": "ur",
        "label": "UR · Unemployed",
        "messages": [
            "ہیلو",
            "میں بے روزگار ہوں",
            "انٹرمیڈیٹ",
            "مرد",
            "پاکستانی",
            "غیر شادی شدہ",
            "ابوظہبی",
            "1 سال",
            "ہاں ڈھونڈ رہا ہوں",
            "آن لائن پورٹل",
            "ہاں دستیاب ہوں",
            "2 مہینے",
            "فل ٹائم",
            "ہاں پہلے کام کیا",
            "کلرک",
            "نجی",
            "ملازمت ختم ہوئی",
            "3000 درہم",
            "کوئی ترجیح نہیں",
            "پراعتماد",
        ],
    },

    # ── Hindi scenarios ────────────────────────────────────────────────────────
    "hi_employed": {
        "language": "hi",
        "label": "HI · Employed · Driver",
        "messages": [
            "नमस्ते",
            "मैं काम करता हूं",
            "दसवीं पास",
            "पुरुष",
            "भारतीय",
            "विवाहित",
            "दुबई",
            "3 साल",
            "ड्राइवर",
            "मैं ट्रक और डिलीवरी वाहन चलाता हूं",
            "परिवहन",
            "वेतनभोगी कर्मचारी",
            "निजी क्षेत्र",
            "60",
            "नहीं",
            "पूर्णकालिक",
            "3000 से 5000 दिरहम",
            "AI पसंद है",
            "आत्मविश्वासी",
        ],
    },
    "hi_unemployed": {
        "language": "hi",
        "label": "HI · Unemployed",
        "messages": [
            "हेलो",
            "मैं बेरोजगार हूं",
            "स्नातक",
            "महिला",
            "भारतीय",
            "अविवाहित",
            "शारजाह",
            "2 साल",
            "हां काम खोज रही हूं",
            "नौकरी पोर्टल",
            "हां उपलब्ध हूं",
            "5 महीने",
            "पूर्णकालिक",
            "हां पहले काम किया",
            "शिक्षक",
            "निजी",
            "अनुबंध समाप्त",
            "5000 दिरहम",
            "कोई प्राथमिकता नहीं",
            "आत्मविश्वासी",
        ],
    },

    # ── Filipino / Tagalog scenarios ───────────────────────────────────────────
    "tl_employed_nars": {
        "language": "tl",
        "label": "TL · Employed · Nurse",
        "messages": [
            "Magandang araw",
            "Nagtatrabaho po ako",
            "Bachelor's degree",
            "Babae",
            "Pilipino",
            "Hindi kasal",
            "Dubai",
            "4 na taon",
            "Nars po ako",
            "Nag-aalaga ng pasyente sa ospital",
            "Pangangalagang pangkalusugan",
            "Empleyado na may suweldo",
            "Pribadong sektor",
            "48",
            "Wala",
            "Buong oras",
            "5000 hanggang 10000 dirham",
            "AI ang mas gusto ko",
            "Tiwala",
        ],
    },
    "tl_employed_driver": {
        "language": "tl",
        "label": "TL · Employed · Driver",
        "messages": [
            "Hello po",
            "May trabaho po ako",
            "Hayskul lang",
            "Lalaki",
            "Pilipino",
            "May asawa",
            "Abu Dhabi",
            "5 taon",
            "Driver po",
            "Nagmamaneho ng delivery truck",
            "Transport",
            "Empleyado",
            "Pribado",
            "60",
            "Wala",
            "Full time",
            "3000 dirham",
            "AI ok lang",
            "Confident",
        ],
    },
    "tl_unemployed": {
        "language": "tl",
        "label": "TL · Unemployed",
        "messages": [
            "Kumusta",
            "Wala akong trabaho ngayon",
            "College graduate",
            "Babae",
            "Pilipino",
            "Hindi kasal",
            "Dubai",
            "1 taon",
            "Oo naghahanap ng trabaho",
            "Online job portals",
            "Oo available na ako",
            "3 buwan",
            "Full time",
            "Oo nagtrabaho na dati",
            "Customer service",
            "Pribado",
            "Nawalan ng trabaho",
            "2500 dirham",
            "Walang preference",
            "Confident",
        ],
    },
}

# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn: int
    latency_ms: float
    status_code: int
    state: str
    detected_lang: str
    error: Optional[str] = None

@dataclass
class UserResult:
    user_id: int
    scenario: str
    label: str
    language: str
    success: bool
    total_ms: float
    turns_completed: int
    turns_total: int
    final_state: str
    turn_latencies: list[float] = field(default_factory=list)
    error: Optional[str] = None
    isco_code: Optional[str] = None
    isco_confidence: Optional[float] = None

# ── HTTP helper ────────────────────────────────────────────────────────────────

def _post(url: str, body: dict, token: Optional[str] = None, timeout: int = TIMEOUT_S) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()) if e.fp else {}
    except Exception as exc:
        return 0, {"error": str(exc)}


def _get(url: str, token: str, timeout: int = TIMEOUT_S) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as exc:
        return 0, {"error": str(exc)}

# ── Single user simulation ─────────────────────────────────────────────────────

_email_lock = threading.Lock()
_email_counter = [0]

def _next_email() -> str:
    with _email_lock:
        _email_counter[0] += 1
        return f"loadtest_user_{_email_counter[0]}@example.com"


def simulate_user(user_id: int, scenario_key: str, base: str) -> UserResult:
    scenario    = SCENARIOS[scenario_key]
    language    = scenario["language"]
    label       = scenario["label"]
    messages    = scenario["messages"]
    email       = _next_email()
    turn_lats: list[float] = []
    t_start     = time.perf_counter()

    # ── Step 1: Request OTP ─────────────────────────────────────────────────
    code, data = _post(f"{base}/auth/request-otp", {"email": email})
    if code != 200:
        return UserResult(user_id, scenario_key, label, language, False,
                          (time.perf_counter()-t_start)*1000, 0, len(messages),
                          "auth_failed", error=f"OTP request {code}: {data}")
    otp = data.get("dev_otp")
    if not otp:
        return UserResult(user_id, scenario_key, label, language, False,
                          (time.perf_counter()-t_start)*1000, 0, len(messages),
                          "auth_failed", error="No dev_otp in response")

    # ── Step 2: Verify OTP → token ──────────────────────────────────────────
    code, data = _post(f"{base}/auth/verify-otp", {"email": email, "code": otp})
    if code != 200:
        return UserResult(user_id, scenario_key, label, language, False,
                          (time.perf_counter()-t_start)*1000, 0, len(messages),
                          "auth_failed", error=f"OTP verify {code}: {data}")
    token = data.get("access_token")
    if not token:
        return UserResult(user_id, scenario_key, label, language, False,
                          (time.perf_counter()-t_start)*1000, 0, len(messages),
                          "auth_failed", error="No access_token")

    # ── Step 3: Create session ──────────────────────────────────────────────
    code, data = _post(f"{base}/survey/sessions",
                       {"language": language if language != "ar-gulf" else "ar"},
                       token=token)
    if code not in (200, 201):
        return UserResult(user_id, scenario_key, label, language, False,
                          (time.perf_counter()-t_start)*1000, 0, len(messages),
                          "session_failed", error=f"Session create {code}: {data}")
    session_id = data.get("id")
    if not session_id:
        return UserResult(user_id, scenario_key, label, language, False,
                          (time.perf_counter()-t_start)*1000, 0, len(messages),
                          "session_failed", error="No session id")

    # ── Step 4: Conversation turns ──────────────────────────────────────────
    current_state = "greeting"
    turns_done    = 0
    last_isco_code = None
    last_isco_conf = None

    for i, msg in enumerate(messages):
        # Random think time (simulates human typing delay)
        think = random.randint(*THINK_MS) / 1000
        time.sleep(think)

        t0 = time.perf_counter()
        code, resp = _post(
            f"{base}/survey/sessions/{session_id}/message",
            {"message": msg, "preferred_language": language if language != "ar-gulf" else "ar"},
            token=token,
        )
        lat = (time.perf_counter() - t0) * 1000
        turn_lats.append(lat)
        turns_done += 1

        if code != 200:
            return UserResult(user_id, scenario_key, label, language, False,
                              (time.perf_counter()-t_start)*1000, turns_done, len(messages),
                              current_state, turn_latencies=turn_lats,
                              error=f"Turn {i+1} HTTP {code}: {resp.get('detail','')}")

        current_state = resp.get("state", current_state)

        # Capture ISCO if available
        isco_list = resp.get("isco_classifications", [])
        if isco_list:
            best = isco_list[0]
            last_isco_code = best.get("primary_code") or best.get("code")
            last_isco_conf = best.get("confidence")

        # Stop early if session completed
        if resp.get("session_completed") or current_state == "completing":
            break

    total_ms = (time.perf_counter() - t_start) * 1000
    return UserResult(
        user_id=user_id,
        scenario=scenario_key,
        label=label,
        language=language,
        success=True,
        total_ms=total_ms,
        turns_completed=turns_done,
        turns_total=len(messages),
        final_state=current_state,
        turn_latencies=turn_lats,
        isco_code=last_isco_code,
        isco_confidence=last_isco_conf,
    )

# ── Result printing ────────────────────────────────────────────────────────────

def _pct(lat_list: list[float], p: int) -> float:
    if not lat_list:
        return 0.0
    s = sorted(lat_list)
    idx = math.ceil(p / 100 * len(s)) - 1
    return s[max(0, idx)]


def print_results(results: list[UserResult], elapsed_s: float) -> None:
    total   = len(results)
    success = [r for r in results if r.success]
    failed  = [r for r in results if not r.success]

    all_turn_lats = [lat for r in success for lat in r.turn_latencies]
    total_lats    = [r.total_ms for r in success]

    print("\n" + "=" * 70)
    print("  LOAD TEST RESULTS")
    print("=" * 70)

    print(f"\n  Total users       : {total}")
    print(f"  Successful        : {len(success)}  ({len(success)/total*100:.1f}%)")
    print(f"  Failed            : {len(failed)}   ({len(failed)/total*100:.1f}%)")
    print(f"  Wall-clock time   : {elapsed_s:.1f}s")
    print(f"  Throughput        : {len(success)/elapsed_s:.2f} users/s")

    if all_turn_lats:
        print(f"\n  --- Per-Turn Latency (n={len(all_turn_lats)} turns) ---")
        print(f"  Mean    : {statistics.mean(all_turn_lats):.0f} ms")
        print(f"  Median  : {statistics.median(all_turn_lats):.0f} ms")
        print(f"  P95     : {_pct(all_turn_lats, 95):.0f} ms")
        print(f"  P99     : {_pct(all_turn_lats, 99):.0f} ms")
        print(f"  Max     : {max(all_turn_lats):.0f} ms")
        print(f"  Min     : {min(all_turn_lats):.0f} ms")

    if total_lats:
        print(f"\n  --- Full Session Duration ---")
        print(f"  Mean    : {statistics.mean(total_lats)/1000:.1f}s")
        print(f"  Median  : {statistics.median(total_lats)/1000:.1f}s")
        print(f"  P95     : {_pct(total_lats, 95)/1000:.1f}s")
        print(f"  Max     : {max(total_lats)/1000:.1f}s")

    # ── Per-language breakdown ──────────────────────────────────────────────
    print(f"\n  --- Per-Language Results ---")
    lang_map: dict[str, list[UserResult]] = {}
    for r in results:
        lang_map.setdefault(r.language, []).append(r)

    lang_order = ["en", "ar", "ar-gulf", "ur", "hi", "tl"]
    print(f"  {'Language':<12} {'Users':>6} {'Success':>8} {'Avg Turn(ms)':>14} {'P95(ms)':>9}")
    print(f"  {'-'*52}")
    for lang in lang_order:
        rs = lang_map.get(lang, [])
        if not rs:
            continue
        ok = [r for r in rs if r.success]
        lats = [lat for r in ok for lat in r.turn_latencies]
        avg  = f"{statistics.mean(lats):.0f}" if lats else "n/a"
        p95  = f"{_pct(lats, 95):.0f}"        if lats else "n/a"
        print(f"  {lang:<12} {len(rs):>6} {len(ok):>7} ({len(ok)/len(rs)*100:.0f}%)"
              f"  {avg:>10} ms  {p95:>6} ms")

    # ── Per-scenario breakdown ──────────────────────────────────────────────
    print(f"\n  --- Per-Scenario Results ---")
    print(f"  {'Scenario':<35} {'Ok':>4} {'Turns':>7} {'State':<16} {'Avg(ms)':>8}")
    print(f"  {'-'*76}")
    seen: dict[str, list[UserResult]] = {}
    for r in results:
        seen.setdefault(r.scenario, []).append(r)
    for scenario_key, rs in sorted(seen.items()):
        ok   = [r for r in rs if r.success]
        lats = [lat for r in ok for lat in r.turn_latencies]
        avg_turns = statistics.mean([r.turns_completed for r in ok]) if ok else 0
        avg_lat   = f"{statistics.mean(lats):.0f}ms" if lats else "n/a"
        states    = [r.final_state for r in ok]
        top_state = max(set(states), key=states.count) if states else "n/a"
        label     = SCENARIOS[scenario_key]["label"]
        print(f"  {label:<35} {len(ok):>2}/{len(rs):<2}  {avg_turns:>5.1f}  {top_state:<16} {avg_lat:>8}")

    # ── FSM state distribution ──────────────────────────────────────────────
    print(f"\n  --- Final FSM State Distribution ---")
    states: dict[str, int] = {}
    for r in success:
        states[r.final_state] = states.get(r.final_state, 0) + 1
    for state, count in sorted(states.items(), key=lambda x: -x[1]):
        bar = "#" * int(count / len(success) * 30)
        print(f"  {state:<20} {count:>3}  {bar}")

    # ── Failure summary ─────────────────────────────────────────────────────
    if failed:
        print(f"\n  --- Failures ({len(failed)}) ---")
        err_counts: dict[str, int] = {}
        for r in failed:
            key = (r.error or "unknown")[:80]
            err_counts[key] = err_counts.get(key, 0) + 1
        for msg, cnt in sorted(err_counts.items(), key=lambda x: -x[1]):
            print(f"  [{cnt}x] {msg}")

    print("\n" + "=" * 70)


# ── Main entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LFS 100-user load test")
    parser.add_argument("--users",    type=int, default=N_USERS)
    parser.add_argument("--base",     type=str, default=BASE_URL)
    parser.add_argument("--ramp",     type=float, default=RAMP_UP_S)
    parser.add_argument("--workers",  type=int, default=20,
                        help="Max parallel threads (default 20)")
    args = parser.parse_args()

    # Check backend is up
    try:
        with urllib.request.urlopen(f"{args.base}/health", timeout=5) as r:
            health = json.loads(r.read())
        print(f"  Backend: {args.base} — {health}")
    except Exception as e:
        print(f"  ERROR: Cannot reach backend at {args.base}: {e}")
        sys.exit(1)

    # Assign scenarios round-robin across users
    scenario_keys = list(SCENARIOS.keys())
    assignments   = [(i, scenario_keys[i % len(scenario_keys)]) for i in range(args.users)]

    print(f"\n  Starting load test:")
    print(f"    Users      : {args.users}")
    print(f"    Scenarios  : {len(scenario_keys)} ({', '.join(s[:20] for s in scenario_keys[:5])}...)")
    print(f"    Languages  : en, ar, ar-gulf, ur, hi, tl")
    print(f"    Ramp-up    : {args.ramp}s")
    print(f"    Workers    : {args.workers}")
    print(f"    Think time : {THINK_MS[0]}-{THINK_MS[1]}ms between turns")

    results:  list[UserResult] = []
    results_lock = threading.Lock()
    done_count   = [0]

    def run_user(user_id: int, scenario_key: str) -> None:
        # Staggered start — spread over ramp-up window
        delay = random.uniform(0, args.ramp)
        time.sleep(delay)
        r = simulate_user(user_id, scenario_key, args.base)
        with results_lock:
            results.append(r)
            done_count[0] += 1
            pct = done_count[0] / args.users * 100
            status = "OK " if r.success else "ERR"
            print(f"  [{status}] User {user_id:03d} | {r.label[:35]:<35} | "
                  f"{r.turns_completed}/{r.turns_total} turns | "
                  f"{r.total_ms/1000:.1f}s | state={r.final_state} "
                  f"[{done_count[0]}/{args.users} {pct:.0f}%]",
                  flush=True)

    import concurrent.futures
    t_wall_start = time.perf_counter()
    print(f"\n  Progress:")
    print(f"  {'-'*70}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_user, uid, sc) for uid, sc in assignments]
        concurrent.futures.wait(futures)

    elapsed = time.perf_counter() - t_wall_start
    print_results(results, elapsed)

    # Save JSON results
    out_path = "backend/evaluation/load_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{
            "user_id":         r.user_id,
            "scenario":        r.scenario,
            "label":           r.label,
            "language":        r.language,
            "success":         r.success,
            "total_ms":        round(r.total_ms, 1),
            "turns_completed": r.turns_completed,
            "turns_total":     r.turns_total,
            "final_state":     r.final_state,
            "avg_turn_ms":     round(statistics.mean(r.turn_latencies), 1) if r.turn_latencies else None,
            "p95_turn_ms":     round(_pct(r.turn_latencies, 95), 1) if r.turn_latencies else None,
            "isco_code":       r.isco_code,
            "isco_confidence": r.isco_confidence,
            "error":           r.error,
        } for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved: {out_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# pytest integration  — run with: pytest -m slow
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.slow
def test_load_smoke(tmp_path):
    """
    Smoke load test: 5 users, 2 concurrent workers, no ramp-up.
    Requires a running backend at http://localhost:8000.
    Skip gracefully when backend is not reachable.

    Run:  pytest -m slow --no-header -rN backend/tests/load_test.py
    """
    import urllib.request
    import urllib.error

    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3):
            pass
    except Exception:
        pytest.skip("Backend not reachable at http://localhost:8000 — skipping load test.")

    scenario_keys = list(SCENARIOS.keys())
    assignments = [(i, scenario_keys[i % len(scenario_keys)]) for i in range(5)]
    results: list[UserResult] = []
    lock = threading.Lock()

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(simulate_user, uid, sc, BASE_URL) for uid, sc in assignments]
        for f in concurrent.futures.as_completed(futures):
            with lock:
                results.append(f.result())

    success_rate = sum(1 for r in results if r.success) / len(results)
    assert success_rate >= 0.8, (
        f"Load smoke: only {success_rate:.0%} of users succeeded. "
        f"Errors: {[r.error for r in results if not r.success]}"
    )
