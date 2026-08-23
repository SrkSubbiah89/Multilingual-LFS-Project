"""
backend/agents/isced_classifier.py

ISCED 2011 + ISCED-F 2013 classifier for the LFS survey education question.

Two complementary dimensions:
  1. ISCED 2011 Level (0–8)  — how far they studied
  2. ISCED-F 2013 Field      — what they specialised in (4-digit detailed field)

ISCED-F 2013 hierarchy
-----------------------
  Broad field (2-digit)  →  Narrow field (3-digit)  →  Detailed field (4-digit)

  00  Generic programmes
  01  Education
  02  Arts and humanities
  03  Social sciences, journalism and information
  04  Business, administration and law
  05  Natural sciences, mathematics and statistics
  06  Information and communication technologies
  07  Engineering, manufacturing and construction
  08  Agriculture, forestry, fisheries and veterinary
  09  Health and welfare
  10  Services

Output
------
ISCEDClassification
  .level           – 0–8 (ISCED 2011 level)
  .level_title     – "Bachelor's or equivalent level"
  .broad_code      – "06"
  .broad_title     – "Information and Communication Technologies"
  .narrow_code     – "061"
  .narrow_title    – "Information and communication technologies"
  .detailed_code   – "0613"                      ← 4-digit specialisation
  .detailed_title  – "Software and applications development and analysis"
  .confidence      – float 0–1
  .method          – "keyword" | "rule" | "llm" (only when a reranker_model
                     is configured -- see ISCEDClassifier.__init__)

Usage
-----
from backend.agents.isced_classifier import ISCEDClassifier

clf = ISCEDClassifier()
r = clf.classify("Bachelor of Engineering in Computer Science")
print(r.level)          # 6
print(r.detailed_code)  # "0613"
print(r.detailed_title) # "Software and applications development and analysis"

r = clf.classify("دكتوراه في الطب")
print(r.level)          # 8
print(r.detailed_code)  # "0912"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from crewai import Agent, Crew, Task

from backend.agents.classifier_methods import (
    ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD,
    ISCEDF_HIERARCHICAL_RETRIEVAL,
)
from backend.llm.llm_client import get_llm_strict

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISCED 2011 — level definitions (level-only scoring)
# ---------------------------------------------------------------------------

_ISCED_LEVELS: list[dict] = [
    {"level": 0, "level_title": "Early childhood education",
     "keywords": "no education illiterate nursery kindergarten preschool KG روضة حضانة أمي بدون تعليم"},
    {"level": 1, "level_title": "Primary education",
     "keywords": "primary elementary grade 1 2 3 4 5 6 basic school ابتدائية ابتدائي"},
    {"level": 2, "level_title": "Lower secondary education",
     "keywords": "middle junior secondary intermediate grade 7 8 9 متوسطة إعدادية"},
    {"level": 3, "level_title": "Upper secondary education",
     "keywords": "high school secondary grade 10 11 12 A-levels baccalaureate ثانوية ثانوي توجيهي intermediate matric mataas na paaralan hayskul"},
    {"level": 4, "level_title": "Post-secondary non-tertiary education",
     "keywords": "vocational certificate post-secondary diploma technical institute دبلوم فني معهد تقني"},
    {"level": 5, "level_title": "Short-cycle tertiary education",
     "keywords": "associate degree two-year HND higher diploma community college دبلوم عالي"},
    {"level": 6, "level_title": "Bachelor's or equivalent level",
     "keywords": "bachelor BSc BA BEng BTech BBA undergraduate university degree بكالوريوس ليسانس جامعي graduation snatak kolehiyo antas lisensyado"},
    {"level": 7, "level_title": "Master's or equivalent level",
     "keywords": "master masters MSc MA MBA MEng postgraduate graduate ماجستير دراسات عليا masters degree masteral masters program"},
    {"level": 8, "level_title": "Doctoral or equivalent level",
     "keywords": "PhD doctorate doctoral DPhil thesis professor دكتوراه دكتور باحث doktori shodh doktorado phd research"},
]

_LEVEL_TOKEN_INDEX: dict[str, list[dict]] = {}
for _entry in _ISCED_LEVELS:
    for _tok in re.findall(r"[a-z\u0600-\u06ff]{2,}", _entry["keywords"].lower()):
        _LEVEL_TOKEN_INDEX.setdefault(_tok, []).append(_entry)


# ---------------------------------------------------------------------------
# ISCED-F 2013 — Fields of Education and Training
# Each entry: broad (2-digit) → narrow (3-digit) → detailed (4-digit) + keywords
# Source: UNESCO Institute for Statistics, ISCED Fields of Education 2013
# ---------------------------------------------------------------------------

_ISCED_FIELDS: list[dict] = [

    # ── 00 Generic programmes and qualifications ──────────────────────────────
    {"broad_code": "00", "broad_title": "Generic programmes and qualifications",
     "narrow_code": "001", "narrow_title": "Basic programmes and qualifications",
     "detailed_code": "0011", "detailed_title": "Basic programmes (no specific field)",
     "keywords": "general education no specific field undeclared unspecified تعليم عام"},
    {"broad_code": "00", "broad_title": "Generic programmes and qualifications",
     "narrow_code": "002", "narrow_title": "Literacy and numeracy",
     "detailed_code": "0021", "detailed_title": "Literacy and numeracy skills",
     "keywords": "literacy numeracy reading writing arithmetic adult learning محو الأمية"},

    # ── 01 Education ──────────────────────────────────────────────────────────
    {"broad_code": "01", "broad_title": "Education",
     "narrow_code": "011", "narrow_title": "Education",
     "detailed_code": "0111", "detailed_title": "Education science",
     "keywords": "education science pedagogy curriculum educational studies علوم التربية التعليم"},
    {"broad_code": "01", "broad_title": "Education",
     "narrow_code": "011", "narrow_title": "Education",
     "detailed_code": "0112", "detailed_title": "Training for pre-school teachers",
     "keywords": "pre-school kindergarten early childhood teacher training معلمة روضة تأهيل تربوي"},
    {"broad_code": "01", "broad_title": "Education",
     "narrow_code": "011", "narrow_title": "Education",
     "detailed_code": "0113", "detailed_title": "Teacher training without subject specialisation",
     "keywords": "teacher training general education pedagogy PGCE تدريب المعلمين إعداد المعلم"},
    {"broad_code": "01", "broad_title": "Education",
     "narrow_code": "011", "narrow_title": "Education",
     "detailed_code": "0114", "detailed_title": "Teacher training with subject specialisation",
     "keywords": "subject teacher maths physics chemistry biology english arabic teacher تدريس تخصص معلم مادة"},

    # ── 02 Arts and Humanities ────────────────────────────────────────────────
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "021", "narrow_title": "Arts",
     "detailed_code": "0211", "detailed_title": "Audio-visual techniques and media production",
     "keywords": "media production film television photography video audio visual broadcast إنتاج إعلامي تصوير"},
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "021", "narrow_title": "Arts",
     "detailed_code": "0212", "detailed_title": "Fashion, interior and industrial design",
     "keywords": "design fashion interior industrial graphic product designer تصميم أزياء ديكور صناعي"},
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "021", "narrow_title": "Arts",
     "detailed_code": "0213", "detailed_title": "Fine arts",
     "keywords": "fine arts painting sculpture drawing ceramics artwork gallery فنون جميلة رسم نحت"},
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "021", "narrow_title": "Arts",
     "detailed_code": "0215", "detailed_title": "Music and performing arts",
     "keywords": "music performing arts theatre drama dance singing instrument موسيقى مسرح فنون أدائية"},
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "022", "narrow_title": "Humanities",
     "detailed_code": "0221", "detailed_title": "Religion and theology",
     "keywords": "religion theology islamic studies sharia fiqh divinity الشريعة الإسلامية الدراسات الدينية فقه"},
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "022", "narrow_title": "Humanities",
     "detailed_code": "0222", "detailed_title": "Foreign languages and cultures",
     "keywords": "foreign language english french german spanish chinese language culture اللغة الإنجليزية لغة أجنبية"},
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "022", "narrow_title": "Humanities",
     "detailed_code": "0223", "detailed_title": "Mother tongue",
     "keywords": "arabic language mother tongue literature linguistics اللغة العربية أدب عربي لغويات"},
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "022", "narrow_title": "Humanities",
     "detailed_code": "0224", "detailed_title": "History, philosophy and related subjects",
     "keywords": "history philosophy archaeology classics heritage تاريخ فلسفة آثار"},
    {"broad_code": "02", "broad_title": "Arts and humanities",
     "narrow_code": "023", "narrow_title": "Languages",
     "detailed_code": "0231", "detailed_title": "Language acquisition",
     "keywords": "language learning IELTS TOEFL translation interpretation bilingual تعلم اللغة ترجمة"},

    # ── 03 Social sciences, journalism and information ─────────────────────
    {"broad_code": "03", "broad_title": "Social sciences, journalism and information",
     "narrow_code": "031", "narrow_title": "Social and behavioural sciences",
     "detailed_code": "0311", "detailed_title": "Economics",
     "keywords": "economics economic theory macro micro finance monetary اقتصاد علوم اقتصادية"},
    {"broad_code": "03", "broad_title": "Social sciences, journalism and information",
     "narrow_code": "031", "narrow_title": "Social and behavioural sciences",
     "detailed_code": "0312", "detailed_title": "Political sciences and civics",
     "keywords": "political science politics government policy international relations علوم سياسية سياسة"},
    {"broad_code": "03", "broad_title": "Social sciences, journalism and information",
     "narrow_code": "031", "narrow_title": "Social and behavioural sciences",
     "detailed_code": "0313", "detailed_title": "Psychology",
     "keywords": "psychology psychiatry mental health counselling behavioral علم النفس نفسية"},
    {"broad_code": "03", "broad_title": "Social sciences, journalism and information",
     "narrow_code": "031", "narrow_title": "Social and behavioural sciences",
     "detailed_code": "0314", "detailed_title": "Sociology and cultural studies",
     "keywords": "sociology anthropology cultural studies social science علم الاجتماع دراسات اجتماعية"},
    {"broad_code": "03", "broad_title": "Social sciences, journalism and information",
     "narrow_code": "032", "narrow_title": "Journalism and information",
     "detailed_code": "0321", "detailed_title": "Journalism and reporting",
     "keywords": "journalism media reporter news broadcasting PR صحافة إعلام تلفزيون"},
    {"broad_code": "03", "broad_title": "Social sciences, journalism and information",
     "narrow_code": "032", "narrow_title": "Journalism and information",
     "detailed_code": "0322", "detailed_title": "Library, information and archival sciences",
     "keywords": "library information science archives documentation records علم المعلومات المكتبات"},

    # ── 04 Business, administration and law ───────────────────────────────────
    {"broad_code": "04", "broad_title": "Business, administration and law",
     "narrow_code": "041", "narrow_title": "Business and administration",
     "detailed_code": "0411", "detailed_title": "Accounting and taxation",
     "keywords": "accounting accountant CPA taxation tax audit finance محاسبة محاسب ضريبة مراجعة"},
    {"broad_code": "04", "broad_title": "Business, administration and law",
     "narrow_code": "041", "narrow_title": "Business and administration",
     "detailed_code": "0412", "detailed_title": "Finance, banking and insurance",
     "keywords": "finance banking insurance investment financial markets مالية بنوك تأمين تمويل"},
    {"broad_code": "04", "broad_title": "Business, administration and law",
     "narrow_code": "041", "narrow_title": "Business and administration",
     "detailed_code": "0413", "detailed_title": "Management and administration",
     "keywords": "management business administration MBA HR human resources إدارة الأعمال إدارة موارد بشرية"},
    {"broad_code": "04", "broad_title": "Business, administration and law",
     "narrow_code": "041", "narrow_title": "Business and administration",
     "detailed_code": "0414", "detailed_title": "Marketing and advertising",
     "keywords": "marketing advertising brand digital marketing sales تسويق إعلانات مبيعات"},
    {"broad_code": "04", "broad_title": "Business, administration and law",
     "narrow_code": "041", "narrow_title": "Business and administration",
     "detailed_code": "0415", "detailed_title": "Secretarial and office work",
     "keywords": "secretarial office administration receptionist clerical سكرتارية إدارية مكتبية"},
    {"broad_code": "04", "broad_title": "Business, administration and law",
     "narrow_code": "042", "narrow_title": "Law",
     "detailed_code": "0421", "detailed_title": "Law",
     "keywords": "law LLB LLM lawyer attorney legal studies شريعة وقانون حقوق قانون"},

    # ── 05 Natural sciences, mathematics and statistics ───────────────────────
    {"broad_code": "05", "broad_title": "Natural sciences, mathematics and statistics",
     "narrow_code": "051", "narrow_title": "Biological and related sciences",
     "detailed_code": "0511", "detailed_title": "Biology",
     "keywords": "biology biological sciences microbiology genetics molecular احياء بيولوجيا"},
    {"broad_code": "05", "broad_title": "Natural sciences, mathematics and statistics",
     "narrow_code": "051", "narrow_title": "Biological and related sciences",
     "detailed_code": "0512", "detailed_title": "Biochemistry",
     "keywords": "biochemistry molecular biology biotechnology biomedical كيمياء حيوية بيوكيمياء"},
    {"broad_code": "05", "broad_title": "Natural sciences, mathematics and statistics",
     "narrow_code": "052", "narrow_title": "Environment",
     "detailed_code": "0521", "detailed_title": "Environmental sciences",
     "keywords": "environmental science ecology sustainability climate change علوم البيئة بيئة"},
    {"broad_code": "05", "broad_title": "Natural sciences, mathematics and statistics",
     "narrow_code": "053", "narrow_title": "Physical sciences",
     "detailed_code": "0531", "detailed_title": "Chemistry",
     "keywords": "chemistry chemical organic inorganic analytical كيمياء"},
    {"broad_code": "05", "broad_title": "Natural sciences, mathematics and statistics",
     "narrow_code": "053", "narrow_title": "Physical sciences",
     "detailed_code": "0533", "detailed_title": "Physics",
     "keywords": "physics applied physics nuclear optics فيزياء"},
    {"broad_code": "05", "broad_title": "Natural sciences, mathematics and statistics",
     "narrow_code": "054", "narrow_title": "Mathematics and statistics",
     "detailed_code": "0541", "detailed_title": "Mathematics",
     "keywords": "mathematics maths applied mathematics رياضيات"},
    {"broad_code": "05", "broad_title": "Natural sciences, mathematics and statistics",
     "narrow_code": "054", "narrow_title": "Mathematics and statistics",
     "detailed_code": "0542", "detailed_title": "Statistics",
     "keywords": "statistics data analysis actuarial statistics probability إحصاء"},

    # ── 06 Information and Communication Technologies (ICTs) ──────────────────
    {"broad_code": "06", "broad_title": "Information and Communication Technologies",
     "narrow_code": "061", "narrow_title": "Information and communication technologies",
     "detailed_code": "0611", "detailed_title": "Computer use",
     "keywords": "computer use IT basic computing ICDL information technology تقنية معلومات حاسوب استخدام"},
    {"broad_code": "06", "broad_title": "Information and Communication Technologies",
     "narrow_code": "061", "narrow_title": "Information and communication technologies",
     "detailed_code": "0612", "detailed_title": "Database and network design and administration",
     "keywords": "database network administration DBA sysadmin networking CCNA قواعد بيانات شبكات"},
    {"broad_code": "06", "broad_title": "Information and Communication Technologies",
     "narrow_code": "061", "narrow_title": "Information and communication technologies",
     "detailed_code": "0613", "detailed_title": "Software and applications development and analysis",
     "keywords": "software development programming coding computer science CS engineering web mobile app developer برمجيات تطوير praudyogiki suchana IT computer teknolohiya programador"},
    {"broad_code": "06", "broad_title": "Information and Communication Technologies",
     "narrow_code": "061", "narrow_title": "Information and communication technologies",
     "detailed_code": "0619", "detailed_title": "Information and communication technologies (not elsewhere classified)",
     "keywords": "cybersecurity artificial intelligence AI machine learning data science cloud computing أمن معلومات ذكاء اصطناعي"},

    # ── 07 Engineering, manufacturing and construction ────────────────────────
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "071", "narrow_title": "Engineering and engineering trades",
     "detailed_code": "0711", "detailed_title": "Chemical engineering and processes",
     "keywords": "chemical engineering process engineering petroleum refining كيمياء هندسية هندسة كيميائية"},
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "071", "narrow_title": "Engineering and engineering trades",
     "detailed_code": "0712", "detailed_title": "Environmental protection technology",
     "keywords": "environmental engineering green technology waste treatment هندسة بيئية"},
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "071", "narrow_title": "Engineering and engineering trades",
     "detailed_code": "0713", "detailed_title": "Electricity and energy",
     "keywords": "electrical engineering power energy renewable هندسة كهربائية طاقة"},
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "071", "narrow_title": "Engineering and engineering trades",
     "detailed_code": "0714", "detailed_title": "Electronics and automation",
     "keywords": "electronics engineering automation robotics embedded systems هندسة إلكترونية"},
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "071", "narrow_title": "Engineering and engineering trades",
     "detailed_code": "0715", "detailed_title": "Mechanics and metal trades",
     "keywords": "mechanical engineering mechanics manufacturing production هندسة ميكانيكية"},
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "071", "narrow_title": "Engineering and engineering trades",
     "detailed_code": "0716", "detailed_title": "Motor vehicles, ships and aircraft",
     "keywords": "automotive aerospace naval aeronautical marine engineering طيران سفن سيارات"},
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "073", "narrow_title": "Architecture and construction",
     "detailed_code": "0731", "detailed_title": "Architecture and town planning",
     "keywords": "architecture urban planning town planning design architect هندسة معمارية تخطيط عمراني"},
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "073", "narrow_title": "Architecture and construction",
     "detailed_code": "0732", "detailed_title": "Building and civil engineering",
     "keywords": "civil engineering structural construction building infrastructure هندسة مدنية إنشاءات"},
    {"broad_code": "07", "broad_title": "Engineering, manufacturing and construction",
     "narrow_code": "071", "narrow_title": "Engineering and engineering trades",
     "detailed_code": "0719", "detailed_title": "Engineering and engineering trades (not elsewhere classified)",
     "keywords": "industrial engineering systems engineering biomedical engineering petroleum oil gas هندسة صناعية بترول"},

    # ── 08 Agriculture, forestry, fisheries and veterinary ────────────────────
    {"broad_code": "08", "broad_title": "Agriculture, forestry, fisheries and veterinary",
     "narrow_code": "081", "narrow_title": "Agriculture",
     "detailed_code": "0811", "detailed_title": "Crop and livestock production",
     "keywords": "agriculture agronomy crop livestock farming زراعة إنتاج حيواني"},
    {"broad_code": "08", "broad_title": "Agriculture, forestry, fisheries and veterinary",
     "narrow_code": "084", "narrow_title": "Veterinary",
     "detailed_code": "0841", "detailed_title": "Veterinary",
     "keywords": "veterinary vet animal science animal health طب بيطري بيطرة"},

    # ── 09 Health and welfare ──────────────────────────────────────────────────
    {"broad_code": "09", "broad_title": "Health and welfare",
     "narrow_code": "091", "narrow_title": "Health",
     "detailed_code": "0911", "detailed_title": "Dental studies",
     "keywords": "dentistry dental surgery oral health orthodontics طب أسنان"},
    {"broad_code": "09", "broad_title": "Health and welfare",
     "narrow_code": "091", "narrow_title": "Health",
     "detailed_code": "0912", "detailed_title": "Medicine",
     "keywords": "medicine MBBS medical doctor physician surgery clinical طب الطب tibb dawakhana sehat chikitsa aspatal kalusugan ospital nars"},
    {"broad_code": "09", "broad_title": "Health and welfare",
     "narrow_code": "091", "narrow_title": "Health",
     "detailed_code": "0913", "detailed_title": "Nursing and midwifery",
     "keywords": "nursing nurse midwifery BSN registered nurse تمريض ممرضة"},
    {"broad_code": "09", "broad_title": "Health and welfare",
     "narrow_code": "091", "narrow_title": "Health",
     "detailed_code": "0914", "detailed_title": "Medical diagnostic and treatment technology",
     "keywords": "radiology laboratory biomedical technology diagnostic imaging تشخيص طبي مختبر"},
    {"broad_code": "09", "broad_title": "Health and welfare",
     "narrow_code": "091", "narrow_title": "Health",
     "detailed_code": "0915", "detailed_title": "Therapy and rehabilitation",
     "keywords": "physiotherapy occupational therapy speech therapy rehabilitation علاج طبيعي"},
    {"broad_code": "09", "broad_title": "Health and welfare",
     "narrow_code": "091", "narrow_title": "Health",
     "detailed_code": "0916", "detailed_title": "Pharmacy",
     "keywords": "pharmacy pharmaceutical sciences pharmacology صيدلة أدوية"},
    {"broad_code": "09", "broad_title": "Health and welfare",
     "narrow_code": "091", "narrow_title": "Health",
     "detailed_code": "0919", "detailed_title": "Health (not elsewhere classified)",
     "keywords": "public health epidemiology nutrition dietitian health management صحة عامة تغذية"},
    {"broad_code": "09", "broad_title": "Health and welfare",
     "narrow_code": "092", "narrow_title": "Welfare",
     "detailed_code": "0923", "detailed_title": "Social work and counselling",
     "keywords": "social work welfare counselling psychology community social خدمة اجتماعية"},

    # ── 10 Services ───────────────────────────────────────────────────────────
    {"broad_code": "10", "broad_title": "Services",
     "narrow_code": "101", "narrow_title": "Personal services",
     "detailed_code": "1013", "detailed_title": "Hotel, restaurant and catering",
     "keywords": "hospitality hotel management tourism catering food service restaurant ضيافة فنادق سياحة"},
    {"broad_code": "10", "broad_title": "Services",
     "narrow_code": "101", "narrow_title": "Personal services",
     "detailed_code": "1014", "detailed_title": "Sports",
     "keywords": "sports science physical education fitness coaching PE رياضة تربية بدنية"},
    {"broad_code": "10", "broad_title": "Services",
     "narrow_code": "101", "narrow_title": "Personal services",
     "detailed_code": "1015", "detailed_title": "Travel, tourism and leisure",
     "keywords": "tourism travel management events سياحة سفر"},
    {"broad_code": "10", "broad_title": "Services",
     "narrow_code": "102", "narrow_title": "Hygiene and occupational health services",
     "detailed_code": "1022", "detailed_title": "Occupational health and safety",
     "keywords": "health safety HSE NEBOSH occupational safety سلامة مهنية"},
    {"broad_code": "10", "broad_title": "Services",
     "narrow_code": "103", "narrow_title": "Security services",
     "detailed_code": "1031", "detailed_title": "Military and defence",
     "keywords": "military defence army police security service دفاع عسكري أمن"},
    {"broad_code": "10", "broad_title": "Services",
     "narrow_code": "104", "narrow_title": "Transport services",
     "detailed_code": "1041", "detailed_title": "Transport services",
     "keywords": "transport logistics aviation maritime supply chain نقل لوجستيات طيران"},
]

# Build level token index
_LEVEL_TOKEN_INDEX: dict[str, list[dict]] = {}
for _lvl_entry in _ISCED_LEVELS:
    for _tok in re.findall(r"[a-z\u0600-\u06ff]{2,}", _lvl_entry["keywords"].lower()):
        _LEVEL_TOKEN_INDEX.setdefault(_tok, []).append(_lvl_entry)

# Build field token index
_FIELD_TOKEN_INDEX: dict[str, list[dict]] = {}
for _fld_entry in _ISCED_FIELDS:
    for _tok in re.findall(r"[a-z\u0600-\u06ff]{2,}", _fld_entry["keywords"].lower()):
        _FIELD_TOKEN_INDEX.setdefault(_tok, []).append(_fld_entry)

# Per-level title lookups, used only by the hierarchical-retrieval path
# (_from_hierarchical_result) to attach broad/narrow titles to a
# hierarchy_path that only carries codes for the non-final stages. Built
# from the same _ISCED_FIELDS table the keyword pipeline already uses.
_BROAD_TITLES: dict[str, str] = {}
_NARROW_TITLES: dict[str, str] = {}
for _fld_entry in _ISCED_FIELDS:
    _BROAD_TITLES.setdefault(_fld_entry["broad_code"], _fld_entry["broad_title"])
    _NARROW_TITLES.setdefault(_fld_entry["narrow_code"], _fld_entry["narrow_title"])

# Default field for generic/pre-tertiary education
_DEFAULT_FIELD = {
    "broad_code": "00", "broad_title": "Generic programmes and qualifications",
    "narrow_code": "001", "narrow_title": "Basic programmes and qualifications",
    "detailed_code": "0011", "detailed_title": "Basic / general education (no specific field)",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ISCEDClassification:
    # ISCED 2011 Level
    level:          int               # 0–8
    level_title:    str               # "Bachelor's or equivalent level"
    # ISCED-F 2013 Field of Specialisation
    broad_code:     str               # 2-digit, e.g. "06"
    broad_title:    str               # "Information and Communication Technologies"
    narrow_code:    str               # 3-digit, e.g. "061"
    narrow_title:   str               # "Information and communication technologies"
    detailed_code:  str               # 4-digit, e.g. "0613"
    detailed_title: str               # "Software and applications development and analysis"
    # Confidence / meta
    confidence:     float
    method:         str               # "keyword" | "rule" | see classifier_methods.py for hierarchical/fallback labels
    raw_text:       Optional[str] = None
    # Populated only by the hierarchical-retrieval path (method=iscedf_hierarchical_retrieval);
    # left at these defaults for every other path (legacy keyword/rule and fallback results).
    # ISCED 2011 level/level_title above are ALWAYS independently classified regardless of path.
    hierarchy_path:      list = field(default_factory=list)      # e.g. ["06", "061", "0613"]
    stage_confidences:   dict = field(default_factory=dict)      # {"stage1": .., "stage2": .., "stage3": ..}
    top_candidates:       list = field(default_factory=list)
    hitl_required:        Optional[bool] = None
    fallback_used:        bool = False
    fallback_reason:      Optional[str] = None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ISCEDClassifier:
    """
    Classify free-text education descriptions to:
      - ISCED 2011 Level (0–8)
      - ISCED-F 2013 Field of Specialisation (4-digit detailed code)

    Fully offline by default — no LLM required unless reranker_model is set.
    """

    _FIELD_KEYWORD_THRESHOLD = 0.85   # skip LLM when field keyword score ≥ this
    _MIN_CANDIDATE_GAP       = 0.15   # skip LLM when top1-top2 score gap ≥ this (provisional -- see ISICClassifier._classify_legacy)
    _TOP_K                   = 3

    def __init__(self, reranker_model: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        reranker_model : str, optional
            Default ``None`` preserves this classifier's original, fully
            offline, no-LLM behaviour exactly — every existing caller that
            omits this parameter sees identical output to before. ISCED
            2011 attainment LEVEL is never affected by this parameter
            either way; it always comes from the deterministic
            ``_score_level()`` scorer. When set to a provider-qualified
            model string (e.g. "gemini/gemini-3.6-flash",
            "groq/openai/gpt-oss-120b"), the ISCED-F 2013 FIELD dimension
            gains an optional LLM re-ranking step mirroring
            ISICClassifier's keyword+LLM pattern: when the keyword field
            match is ambiguous (score < _FIELD_KEYWORD_THRESHOLD), an LLM
            picks the best of the top-3 keyword field candidates and
            ``method`` becomes ``"llm"``. Uses ``get_llm_strict()`` — the
            same fail-closed contract as ISCOClassifier/ISICClassifier's
            ``reranker_model``: a missing or unreachable pinned model
            raises RuntimeError out of ``__init__`` immediately, never a
            silent substitution.
        """
        self._reranker_model_pin = reranker_model
        self._llm = None
        self.reranker_model_resolved = "none (no reranker configured)"
        if reranker_model:
            self._llm = get_llm_strict(reranker_model, temperature=0.3)
            self.reranker_model_resolved = getattr(self._llm, "model", reranker_model)

    def classify(self, text: str, *, method: Optional[str] = None) -> ISCEDClassification:
        """
        Parameters
        ----------
        method : str, optional
            Default ``None`` runs today's unchanged keyword/rule pipeline
            (identical to calling ``classify(text)`` before this parameter
            existed). Passing
            ``backend.agents.classifier_methods.ISCEDF_HIERARCHICAL_RETRIEVAL``
            runs the real parent-filtered hierarchical retrieval engine
            (backend/rag/standard_hierarchical_store.py) for the ISCED-F
            broad/narrow/detailed FIELD dimension only -- the independent
            ISCED 2011 attainment LEVEL is always computed by
            ``_score_level()`` regardless of which path runs; it is never
            part of the hierarchical retrieval. If the required Qdrant
            collections are unavailable or the search returns no usable
            result, this falls back to the legacy keyword/rule pipeline --
            but the returned result's ``method`` is set to
            ``ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD`` (never silently
            reported as ``ISCEDF_HIERARCHICAL_RETRIEVAL``), with
            ``fallback_used=True`` and a non-empty ``fallback_reason``. See
            Documentation/Conference_I_Reviewer_2/
            ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md.
        """
        if method == ISCEDF_HIERARCHICAL_RETRIEVAL:
            return self._classify_hierarchical(text)
        return self._classify_legacy(text)

    # ── Legacy keyword/rule pipeline (unchanged behaviour) ──────────────────────

    def _classify_legacy(self, text: str) -> ISCEDClassification:
        text = (text or "").strip()
        if not text:
            return self._fallback(text)

        level_entry, level_conf = self._score_level(text)
        scored_fields = self._score_field_candidates(text)
        field_conf, field_entry = (scored_fields[0] if scored_fields else (0.0, _DEFAULT_FIELD))
        method = "keyword"

        second_field_conf = scored_fields[1][0] if len(scored_fields) > 1 else 0.0
        field_gap = field_conf - second_field_conf
        ambiguous = field_conf < self._FIELD_KEYWORD_THRESHOLD and field_gap < self._MIN_CANDIDATE_GAP
        if self._llm is not None and scored_fields and ambiguous:
            top_candidates = [e for _, e in scored_fields[:self._TOP_K]]
            llm_result = self._llm_rerank_field(text, top_candidates)
            if llm_result is not None:
                field_entry, field_conf = llm_result
                method = "llm"

        # Use field confidence if there's a strong signal, else default to generic
        combined_conf = round(min((level_conf * 0.4 + field_conf * 0.6), 1.0), 4)

        return ISCEDClassification(
            level=level_entry["level"],
            level_title=level_entry["level_title"],
            broad_code=field_entry["broad_code"],
            broad_title=field_entry["broad_title"],
            narrow_code=field_entry["narrow_code"],
            narrow_title=field_entry["narrow_title"],
            detailed_code=field_entry["detailed_code"],
            detailed_title=field_entry["detailed_title"],
            confidence=combined_conf,
            method=method,
            raw_text=text,
        )

    # ── Hierarchical field retrieval, with explicit fallback labelling ──────────

    def _classify_hierarchical(self, text: str) -> ISCEDClassification:
        from backend.rag.standard_hierarchical_store import get_iscedf_hierarchical_store

        stripped = (text or "").strip()
        if not stripped:
            return self._fallback(text)

        # ISCED 2011 level is independently classified regardless of path.
        level_entry, level_conf = self._score_level(stripped)

        store = get_iscedf_hierarchical_store()
        result = store.search(stripped)

        if result.ready and not result.unavailable_reason and result.code:
            return self._from_hierarchical_result(result, level_entry, level_conf, text)

        field_entry, field_conf = self._score_field(stripped)
        combined_conf = round(min((level_conf * 0.4 + field_conf * 0.6), 1.0), 4)
        return ISCEDClassification(
            level=level_entry["level"],
            level_title=level_entry["level_title"],
            broad_code=field_entry["broad_code"],
            broad_title=field_entry["broad_title"],
            narrow_code=field_entry["narrow_code"],
            narrow_title=field_entry["narrow_title"],
            detailed_code=field_entry["detailed_code"],
            detailed_title=field_entry["detailed_title"],
            confidence=combined_conf,
            method=ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD,
            raw_text=text,
            fallback_used=True,
            fallback_reason=(
                result.unavailable_reason or "ISCED-F hierarchical store returned no usable result"
            ),
        )

    def _from_hierarchical_result(self, result, level_entry: dict, level_conf: float, text: str) -> ISCEDClassification:
        path = list(result.hierarchy_path)
        broad = path[0] if len(path) > 0 else ""
        narrow = path[1] if len(path) > 1 else ""
        detailed = path[2] if len(path) > 2 else result.code

        combined_conf = round(min((level_conf * 0.4 + result.confidence * 0.6), 1.0), 4)

        return ISCEDClassification(
            level=level_entry["level"],
            level_title=level_entry["level_title"],
            broad_code=broad,
            broad_title=_BROAD_TITLES.get(broad, ""),
            narrow_code=narrow,
            narrow_title=_NARROW_TITLES.get(narrow, ""),
            detailed_code=detailed,
            detailed_title=result.label_en,
            confidence=combined_conf,
            method=ISCEDF_HIERARCHICAL_RETRIEVAL,
            raw_text=text,
            hierarchy_path=path,
            stage_confidences=dict(result.stage_confidences),
            top_candidates=list(result.top_candidates),
            hitl_required=result.hitl_required,
            fallback_used=False,
            fallback_reason=None,
        )

    # ── Level scoring ──────────────────────────────────────────────────────────

    def _score_level(self, text: str) -> tuple[dict, float]:
        tokens = set(re.findall(r"[a-z\u0600-\u06ff]{2,}", text.lower()))
        hit_counts: dict[int, int] = {}
        hit_entries: dict[int, dict] = {}
        for tok in tokens:
            for entry in _LEVEL_TOKEN_INDEX.get(tok, []):
                lvl = entry["level"]
                hit_counts[lvl] = hit_counts.get(lvl, 0) + 1
                hit_entries[lvl] = entry
        if not hit_counts:
            return _ISCED_LEVELS[3], 0.3   # default: upper secondary
        max_h = max(hit_counts.values())
        best_lvl = max(hit_counts, key=hit_counts.__getitem__)
        return hit_entries[best_lvl], round(hit_counts[best_lvl] / max_h, 4)

    # ── Field scoring ──────────────────────────────────────────────────────────

    def _score_field_candidates(self, text: str) -> list[tuple[float, dict]]:
        """Return (score, entry) pairs sorted descending by score.

        score = (tokens matching this entry) / (distinct query tokens that
        matched *something*) -- a genuine match-coverage fraction, not a
        normalisation by the winning entry's own hit count (which would make
        the top-1 score mathematically always 1.0 -- see the identical bug
        found and fixed in ISICClassifier._keyword_score, 2026-08-23).
        """
        tokens = set(re.findall(r"[a-z\u0600-\u06ff]{2,}", text.lower()))
        hit_counts: dict[str, int] = {}
        hit_entries: dict[str, dict] = {}
        matched_tokens: set[str] = set()
        for tok in tokens:
            entries = _FIELD_TOKEN_INDEX.get(tok, [])
            if entries:
                matched_tokens.add(tok)
            for entry in entries:
                code = entry["detailed_code"]
                hit_counts[code] = hit_counts.get(code, 0) + 1
                hit_entries[code] = entry
        if not hit_counts:
            return []
        denom = len(matched_tokens)
        scored = [
            (round(count / denom, 4), hit_entries[code])
            for code, count in hit_counts.items()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _score_field(self, text: str) -> tuple[dict, float]:
        scored = self._score_field_candidates(text)
        if not scored:
            return _DEFAULT_FIELD, 0.0
        return scored[0][1], scored[0][0]

    # \u2500\u2500 LLM re-ranking (field dimension only) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _llm_rerank_field(
        self, text: str, candidates: list[dict]
    ) -> Optional[tuple[dict, float]]:
        candidates_text = "\n".join(
            f"Field {c['detailed_code']}: {c['detailed_title']} "
            f"[Narrow {c['narrow_code']}: {c['narrow_title']}, "
            f"Broad {c['broad_code']}: {c['broad_title']}]"
            for c in candidates
        )
        task_desc = (
            f"A survey respondent described their education / field of study as:\n\"{text}\"\n\n"
            f"Choose the single most appropriate ISCED-F 2013 detailed field (4-digit) from:\n"
            f"{candidates_text}\n\n"
            "Respond with JSON only (no markdown):\n"
            '{"detailed_code": "XXXX", "confidence": 0.0-1.0, "reasoning": "..."}'
        )

        try:
            agent = Agent(
                role="ISCED-F 2013 Field Classifier",
                goal="Select the best ISCED-F 2013 detailed field for the education description.",
                backstory="You are a UNESCO Institute for Statistics expert in ISCED-F 2013 field-of-education classification.",
                llm=self._llm,
                verbose=False,
                allow_delegation=False,
            )
            task = Task(
                description=task_desc,
                expected_output='JSON: {"detailed_code": "XXXX", "confidence": 0.0, "reasoning": "..."}',
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            raw = str(crew.kickoff()).strip()
        except Exception as exc:
            log.warning("ISCED-F LLM re-ranking failed: %s", exc)
            return None

        return self._parse_llm_field_response(raw, candidates)

    @staticmethod
    def _parse_llm_field_response(
        raw: str, candidates: list[dict]
    ) -> Optional[tuple[dict, float]]:
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        data: Optional[dict] = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if not data:
            return None

        code = str(data.get("detailed_code", "")).strip()
        conf = float(data.get("confidence", 0.7))

        entry = next((c for c in candidates if c["detailed_code"] == code), None)
        if entry is None:
            entry = next((c for c in candidates if c["narrow_code"] == code), None)
        if entry is None:
            entry = candidates[0]
            conf *= 0.7

        return entry, round(min(conf, 1.0), 4)

    # ── Fallback ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(text: str) -> ISCEDClassification:
        lvl = _ISCED_LEVELS[3]   # upper secondary
        return ISCEDClassification(
            level=lvl["level"],
            level_title=lvl["level_title"],
            broad_code=_DEFAULT_FIELD["broad_code"],
            broad_title=_DEFAULT_FIELD["broad_title"],
            narrow_code=_DEFAULT_FIELD["narrow_code"],
            narrow_title=_DEFAULT_FIELD["narrow_title"],
            detailed_code=_DEFAULT_FIELD["detailed_code"],
            detailed_title=_DEFAULT_FIELD["detailed_title"],
            confidence=0.0,
            method="rule",
            raw_text=text,
        )
