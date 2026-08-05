"""
backend/agents/nationality_classifier.py

UN M49 Nationality / Country-of-Origin Classifier for the LFS survey.

Maps free-text nationality mentions (in English, Arabic, Urdu, Hindi,
or Tagalog) to standardised UN M49 country codes and ISO 3166-1 alpha-3
codes, following the UNSD standard used by the UAE Statistics Centre.

Pipeline
--------
Stage 1 – Keyword/alias lookup (offline, O(1))
    Checks the respondent's text against a comprehensive alias table that
    covers common adjective forms, demonym variants, and Arabic equivalents
    for all nationalities commonly found in the UAE labour force.

Stage 2 – Fuzzy token match (offline, O(n))
    Tokenises the text and checks each token against the alias index.
    Selects the entry with the most token hits.

Output
------
NationalityClassification
    .m49_code       – 3-digit UN M49 numeric string (e.g. "784" for UAE)
    .iso_alpha3     – ISO 3166-1 alpha-3 code (e.g. "ARE")
    .country_en     – English country name
    .country_ar     – Arabic country name
    .region_en      – UN M49 sub-region (e.g. "Western Asia")
    .nationality_en – Common English demonym (e.g. "Emirati")
    .nationality_ar – Arabic demonym (e.g. "إماراتي")
    .confidence     – float 0–1
    .method         – "exact" | "alias" | "token" | "unknown"

Usage
-----
from backend.agents.nationality_classifier import NationalityClassifier

clf = NationalityClassifier()
r = clf.classify("I am from the Philippines")
print(r.m49_code)      # "608"
print(r.iso_alpha3)    # "PHL"
print(r.country_en)    # "Philippines"
print(r.nationality_en)# "Filipino"
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NationalityClassification:
    m49_code:       str
    iso_alpha3:     str
    country_en:     str
    country_ar:     str
    region_en:      str
    nationality_en: str
    nationality_ar: str
    confidence:     float
    method:         str            # "exact" | "alias" | "token" | "unknown"
    raw_text:       Optional[str] = None


# ---------------------------------------------------------------------------
# UN M49 country table
# ---------------------------------------------------------------------------
# Each entry:
#   (m49_code, iso_alpha3, country_en, country_ar, region_en,
#    nationality_en, nationality_ar, aliases)
#
# aliases: space-separated lowercase strings (EN + AR demonyms, adjective
#          forms, common misspellings, transliterations from UR/HI/TL).
# ---------------------------------------------------------------------------

_COUNTRY_DATA: list[dict] = [
    # ── United Arab Emirates ──────────────────────────────────────────────
    {
        "m49": "784", "iso3": "ARE",
        "country_en": "United Arab Emirates", "country_ar": "الإمارات العربية المتحدة",
        "region_en": "Western Asia",
        "nat_en": "Emirati", "nat_ar": "إماراتي",
        "aliases": "uae emirati emiraati emirate emirates إماراتي إمارات اماراتي اتحاد امارات",
    },
    # ── India ─────────────────────────────────────────────────────────────
    {
        "m49": "356", "iso3": "IND",
        "country_en": "India", "country_ar": "الهند",
        "region_en": "Southern Asia",
        "nat_en": "Indian", "nat_ar": "هندي",
        "aliases": "india indian هندي الهند bharat indian subcontinent hindi hindustan",
    },
    # ── Pakistan ──────────────────────────────────────────────────────────
    {
        "m49": "586", "iso3": "PAK",
        "country_en": "Pakistan", "country_ar": "باكستان",
        "region_en": "Southern Asia",
        "nat_en": "Pakistani", "nat_ar": "باكستاني",
        "aliases": "pakistan pakistani باكستاني باكستان پاکستانی پاکستان",
    },
    # ── Bangladesh ────────────────────────────────────────────────────────
    {
        "m49": "050", "iso3": "BGD",
        "country_en": "Bangladesh", "country_ar": "بنغلاديش",
        "region_en": "Southern Asia",
        "nat_en": "Bangladeshi", "nat_ar": "بنغلاديشي",
        "aliases": "bangladesh bangladeshi بنغلاديش بنغلاديشي bangla",
    },
    # ── Philippines ───────────────────────────────────────────────────────
    {
        "m49": "608", "iso3": "PHL",
        "country_en": "Philippines", "country_ar": "الفلبين",
        "region_en": "South-eastern Asia",
        "nat_en": "Filipino", "nat_ar": "فلبيني",
        "aliases": "philippines filipino filipina pilipinas الفلبين فلبيني pinoy pinay",
    },
    # ── Egypt ─────────────────────────────────────────────────────────────
    {
        "m49": "818", "iso3": "EGY",
        "country_en": "Egypt", "country_ar": "مصر",
        "region_en": "Northern Africa",
        "nat_en": "Egyptian", "nat_ar": "مصري",
        "aliases": "egypt egyptian مصري مصر masri",
    },
    # ── Saudi Arabia ──────────────────────────────────────────────────────
    {
        "m49": "682", "iso3": "SAU",
        "country_en": "Saudi Arabia", "country_ar": "المملكة العربية السعودية",
        "region_en": "Western Asia",
        "nat_en": "Saudi", "nat_ar": "سعودي",
        "aliases": "saudi saudi arabia سعودي السعودية سعودية ksa",
    },
    # ── Jordan ────────────────────────────────────────────────────────────
    {
        "m49": "400", "iso3": "JOR",
        "country_en": "Jordan", "country_ar": "الأردن",
        "region_en": "Western Asia",
        "nat_en": "Jordanian", "nat_ar": "أردني",
        "aliases": "jordan jordanian أردني الأردن urduni",
    },
    # ── Lebanon ───────────────────────────────────────────────────────────
    {
        "m49": "422", "iso3": "LBN",
        "country_en": "Lebanon", "country_ar": "لبنان",
        "region_en": "Western Asia",
        "nat_en": "Lebanese", "nat_ar": "لبناني",
        "aliases": "lebanon lebanese لبناني لبنان",
    },
    # ── Syria ─────────────────────────────────────────────────────────────
    {
        "m49": "760", "iso3": "SYR",
        "country_en": "Syria", "country_ar": "سوريا",
        "region_en": "Western Asia",
        "nat_en": "Syrian", "nat_ar": "سوري",
        "aliases": "syria syrian سوري سوريا",
    },
    # ── Oman ─────────────────────────────────────────────────────────────
    {
        "m49": "512", "iso3": "OMN",
        "country_en": "Oman", "country_ar": "عُمان",
        "region_en": "Western Asia",
        "nat_en": "Omani", "nat_ar": "عُماني",
        "aliases": "oman omani عماني عمان",
    },
    # ── Kuwait ────────────────────────────────────────────────────────────
    {
        "m49": "414", "iso3": "KWT",
        "country_en": "Kuwait", "country_ar": "الكويت",
        "region_en": "Western Asia",
        "nat_en": "Kuwaiti", "nat_ar": "كويتي",
        "aliases": "kuwait kuwaiti كويتي الكويت",
    },
    # ── Bahrain ───────────────────────────────────────────────────────────
    {
        "m49": "048", "iso3": "BHR",
        "country_en": "Bahrain", "country_ar": "البحرين",
        "region_en": "Western Asia",
        "nat_en": "Bahraini", "nat_ar": "بحريني",
        "aliases": "bahrain bahraini بحريني البحرين",
    },
    # ── Qatar ─────────────────────────────────────────────────────────────
    {
        "m49": "634", "iso3": "QAT",
        "country_en": "Qatar", "country_ar": "قطر",
        "region_en": "Western Asia",
        "nat_en": "Qatari", "nat_ar": "قطري",
        "aliases": "qatar qatari قطري",
    },
    # ── Yemen ─────────────────────────────────────────────────────────────
    {
        "m49": "887", "iso3": "YEM",
        "country_en": "Yemen", "country_ar": "اليمن",
        "region_en": "Western Asia",
        "nat_en": "Yemeni", "nat_ar": "يمني",
        "aliases": "yemen yemeni يمني اليمن",
    },
    # ── Iraq ──────────────────────────────────────────────────────────────
    {
        "m49": "368", "iso3": "IRQ",
        "country_en": "Iraq", "country_ar": "العراق",
        "region_en": "Western Asia",
        "nat_en": "Iraqi", "nat_ar": "عراقي",
        "aliases": "iraq iraqi عراقي العراق",
    },
    # ── Sri Lanka ─────────────────────────────────────────────────────────
    {
        "m49": "144", "iso3": "LKA",
        "country_en": "Sri Lanka", "country_ar": "سريلانكا",
        "region_en": "Southern Asia",
        "nat_en": "Sri Lankan", "nat_ar": "سيريلانكي",
        "aliases": "sri lanka sri lankan سريلانكي ceylon",
    },
    # ── Nepal ─────────────────────────────────────────────────────────────
    {
        "m49": "524", "iso3": "NPL",
        "country_en": "Nepal", "country_ar": "نيبال",
        "region_en": "Southern Asia",
        "nat_en": "Nepali", "nat_ar": "نيبالي",
        "aliases": "nepal nepali نيبالي nepalese",
    },
    # ── Indonesia ─────────────────────────────────────────────────────────
    {
        "m49": "360", "iso3": "IDN",
        "country_en": "Indonesia", "country_ar": "إندونيسيا",
        "region_en": "South-eastern Asia",
        "nat_en": "Indonesian", "nat_ar": "إندونيسي",
        "aliases": "indonesia indonesian إندونيسي",
    },
    # ── Ethiopia ──────────────────────────────────────────────────────────
    {
        "m49": "231", "iso3": "ETH",
        "country_en": "Ethiopia", "country_ar": "إثيوبيا",
        "region_en": "Eastern Africa",
        "nat_en": "Ethiopian", "nat_ar": "إثيوبي",
        "aliases": "ethiopia ethiopian إثيوبي",
    },
    # ── Kenya ─────────────────────────────────────────────────────────────
    {
        "m49": "404", "iso3": "KEN",
        "country_en": "Kenya", "country_ar": "كينيا",
        "region_en": "Eastern Africa",
        "nat_en": "Kenyan", "nat_ar": "كيني",
        "aliases": "kenya kenyan كيني",
    },
    # ── Nigeria ───────────────────────────────────────────────────────────
    {
        "m49": "566", "iso3": "NGA",
        "country_en": "Nigeria", "country_ar": "نيجيريا",
        "region_en": "Western Africa",
        "nat_en": "Nigerian", "nat_ar": "نيجيري",
        "aliases": "nigeria nigerian نيجيري",
    },
    # ── Sudan ─────────────────────────────────────────────────────────────
    {
        "m49": "729", "iso3": "SDN",
        "country_en": "Sudan", "country_ar": "السودان",
        "region_en": "Northern Africa",
        "nat_en": "Sudanese", "nat_ar": "سوداني",
        "aliases": "sudan sudanese سوداني السودان",
    },
    # ── Morocco ───────────────────────────────────────────────────────────
    {
        "m49": "504", "iso3": "MAR",
        "country_en": "Morocco", "country_ar": "المغرب",
        "region_en": "Northern Africa",
        "nat_en": "Moroccan", "nat_ar": "مغربي",
        "aliases": "morocco moroccan مغربي المغرب",
    },
    # ── Tunisia ───────────────────────────────────────────────────────────
    {
        "m49": "788", "iso3": "TUN",
        "country_en": "Tunisia", "country_ar": "تونس",
        "region_en": "Northern Africa",
        "nat_en": "Tunisian", "nat_ar": "تونسي",
        "aliases": "tunisia tunisian تونسي تونس",
    },
    # ── Iran ──────────────────────────────────────────────────────────────
    {
        "m49": "364", "iso3": "IRN",
        "country_en": "Iran", "country_ar": "إيران",
        "region_en": "Southern Asia",
        "nat_en": "Iranian", "nat_ar": "إيراني",
        "aliases": "iran iranian إيراني persia persian",
    },
    # ── Afghanistan ───────────────────────────────────────────────────────
    {
        "m49": "004", "iso3": "AFG",
        "country_en": "Afghanistan", "country_ar": "أفغانستان",
        "region_en": "Southern Asia",
        "nat_en": "Afghan", "nat_ar": "أفغاني",
        "aliases": "afghanistan afghan أفغاني افغان",
    },
    # ── United Kingdom ────────────────────────────────────────────────────
    {
        "m49": "826", "iso3": "GBR",
        "country_en": "United Kingdom", "country_ar": "المملكة المتحدة",
        "region_en": "Northern Europe",
        "nat_en": "British", "nat_ar": "بريطاني",
        "aliases": "uk britain british england english scotland scottish wales welsh kingdom بريطاني المملكة المتحدة",
    },
    # ── United States ─────────────────────────────────────────────────────
    {
        "m49": "840", "iso3": "USA",
        "country_en": "United States of America", "country_ar": "الولايات المتحدة الأمريكية",
        "region_en": "Northern America",
        "nat_en": "American", "nat_ar": "أمريكي",
        "aliases": "usa us american america states أمريكي الولايات المتحدة",
    },
    # ── Canada ────────────────────────────────────────────────────────────
    {
        "m49": "124", "iso3": "CAN",
        "country_en": "Canada", "country_ar": "كندا",
        "region_en": "Northern America",
        "nat_en": "Canadian", "nat_ar": "كندي",
        "aliases": "canada canadian كندي كندا",
    },
    # ── Australia ────────────────────────────────────────────────────────
    {
        "m49": "036", "iso3": "AUS",
        "country_en": "Australia", "country_ar": "أستراليا",
        "region_en": "Australia and New Zealand",
        "nat_en": "Australian", "nat_ar": "أسترالي",
        "aliases": "australia australian أسترالي",
    },
    # ── France ───────────────────────────────────────────────────────────
    {
        "m49": "250", "iso3": "FRA",
        "country_en": "France", "country_ar": "فرنسا",
        "region_en": "Western Europe",
        "nat_en": "French", "nat_ar": "فرنسي",
        "aliases": "france french فرنسي فرنسا",
    },
    # ── Germany ───────────────────────────────────────────────────────────
    {
        "m49": "276", "iso3": "DEU",
        "country_en": "Germany", "country_ar": "ألمانيا",
        "region_en": "Western Europe",
        "nat_en": "German", "nat_ar": "ألماني",
        "aliases": "germany german ألماني ألمانيا deutsch",
    },
    # ── China ─────────────────────────────────────────────────────────────
    {
        "m49": "156", "iso3": "CHN",
        "country_en": "China", "country_ar": "الصين",
        "region_en": "Eastern Asia",
        "nat_en": "Chinese", "nat_ar": "صيني",
        "aliases": "china chinese صيني الصين",
    },
    # ── South Korea ───────────────────────────────────────────────────────
    {
        "m49": "410", "iso3": "KOR",
        "country_en": "Republic of Korea", "country_ar": "كوريا الجنوبية",
        "region_en": "Eastern Asia",
        "nat_en": "Korean", "nat_ar": "كوري",
        "aliases": "korea korean southkorea كوري كوريا",
    },
    # ── Japan ─────────────────────────────────────────────────────────────
    {
        "m49": "392", "iso3": "JPN",
        "country_en": "Japan", "country_ar": "اليابان",
        "region_en": "Eastern Asia",
        "nat_en": "Japanese", "nat_ar": "ياباني",
        "aliases": "japan japanese ياباني اليابان",
    },
    # ── Russia ────────────────────────────────────────────────────────────
    {
        "m49": "643", "iso3": "RUS",
        "country_en": "Russian Federation", "country_ar": "روسيا",
        "region_en": "Eastern Europe",
        "nat_en": "Russian", "nat_ar": "روسي",
        "aliases": "russia russian روسي روسيا",
    },
    # ── Turkey ────────────────────────────────────────────────────────────
    {
        "m49": "792", "iso3": "TUR",
        "country_en": "Turkey", "country_ar": "تركيا",
        "region_en": "Western Asia",
        "nat_en": "Turkish", "nat_ar": "تركي",
        "aliases": "turkey turkish تركي تركيا turkiye",
    },
    # ── South Africa ──────────────────────────────────────────────────────
    {
        "m49": "710", "iso3": "ZAF",
        "country_en": "South Africa", "country_ar": "جنوب أفريقيا",
        "region_en": "Southern Africa",
        "nat_en": "South African", "nat_ar": "جنوب أفريقي",
        "aliases": "africa african southafrica southafrican جنوب أفريقيا",
    },
    # ── Tanzania ──────────────────────────────────────────────────────────
    {
        "m49": "834", "iso3": "TZA",
        "country_en": "Tanzania", "country_ar": "تنزانيا",
        "region_en": "Eastern Africa",
        "nat_en": "Tanzanian", "nat_ar": "تنزاني",
        "aliases": "tanzania tanzanian تنزاني",
    },
    # ── Ghana ─────────────────────────────────────────────────────────────
    {
        "m49": "288", "iso3": "GHA",
        "country_en": "Ghana", "country_ar": "غانا",
        "region_en": "Western Africa",
        "nat_en": "Ghanaian", "nat_ar": "غاني",
        "aliases": "ghana ghanaian غاني غانا",
    },
    # ── Uganda ────────────────────────────────────────────────────────────
    {
        "m49": "800", "iso3": "UGA",
        "country_en": "Uganda", "country_ar": "أوغندا",
        "region_en": "Eastern Africa",
        "nat_en": "Ugandan", "nat_ar": "أوغندي",
        "aliases": "uganda ugandan أوغندا أوغندي",
    },
    # ── Algeria ───────────────────────────────────────────────────────────
    {
        "m49": "012", "iso3": "DZA",
        "country_en": "Algeria", "country_ar": "الجزائر",
        "region_en": "Northern Africa",
        "nat_en": "Algerian", "nat_ar": "جزائري",
        "aliases": "algeria algerian جزائري الجزائر",
    },
    # ── Libya ────────────────────────────────────────────────────────────
    {
        "m49": "434", "iso3": "LBY",
        "country_en": "Libya", "country_ar": "ليبيا",
        "region_en": "Northern Africa",
        "nat_en": "Libyan", "nat_ar": "ليبي",
        "aliases": "libya libyan ليبي ليبيا",
    },
    # ── Myanmar ───────────────────────────────────────────────────────────
    {
        "m49": "104", "iso3": "MMR",
        "country_en": "Myanmar", "country_ar": "ميانمار",
        "region_en": "South-eastern Asia",
        "nat_en": "Myanmar", "nat_ar": "ميانماري",
        "aliases": "myanmar burma burmese ميانمار",
    },
    # ── Vietnam ───────────────────────────────────────────────────────────
    {
        "m49": "704", "iso3": "VNM",
        "country_en": "Viet Nam", "country_ar": "فيتنام",
        "region_en": "South-eastern Asia",
        "nat_en": "Vietnamese", "nat_ar": "فيتنامي",
        "aliases": "vietnam vietnamese فيتنام فيتنامي",
    },
    # ── Malaysia ──────────────────────────────────────────────────────────
    {
        "m49": "458", "iso3": "MYS",
        "country_en": "Malaysia", "country_ar": "ماليزيا",
        "region_en": "South-eastern Asia",
        "nat_en": "Malaysian", "nat_ar": "ماليزي",
        "aliases": "malaysia malaysian ماليزيا ماليزي",
    },
    # ── Thailand ──────────────────────────────────────────────────────────
    {
        "m49": "764", "iso3": "THA",
        "country_en": "Thailand", "country_ar": "تايلاند",
        "region_en": "South-eastern Asia",
        "nat_en": "Thai", "nat_ar": "تايلاندي",
        "aliases": "thailand thai تايلاند تايلاندي",
    },
    # ── Italy ─────────────────────────────────────────────────────────────
    {
        "m49": "380", "iso3": "ITA",
        "country_en": "Italy", "country_ar": "إيطاليا",
        "region_en": "Southern Europe",
        "nat_en": "Italian", "nat_ar": "إيطالي",
        "aliases": "italy italian إيطاليا إيطالي",
    },
    # ── Spain ─────────────────────────────────────────────────────────────
    {
        "m49": "724", "iso3": "ESP",
        "country_en": "Spain", "country_ar": "إسبانيا",
        "region_en": "Southern Europe",
        "nat_en": "Spanish", "nat_ar": "إسباني",
        "aliases": "spain spanish إسبانيا إسباني",
    },
]


# ---------------------------------------------------------------------------
# Build lookup index
# ---------------------------------------------------------------------------

# alias → entry index
_ALIAS_INDEX: dict[str, int] = {}

for _idx, _entry in enumerate(_COUNTRY_DATA):
    for _alias in re.findall(r"[a-z\u0600-\u06ff]{2,}", _entry["aliases"].lower()):
        if _alias not in _ALIAS_INDEX:
            _ALIAS_INDEX[_alias] = _idx

# ISO-3 → index and M49 → index for direct lookups
_ISO3_INDEX: dict[str, int]  = {e["iso3"].lower(): i for i, e in enumerate(_COUNTRY_DATA)}
_M49_INDEX:  dict[str, int]  = {e["m49"]: i       for i, e in enumerate(_COUNTRY_DATA)}

# Unknown sentinel
_UNKNOWN = NationalityClassification(
    m49_code="000", iso_alpha3="XXX",
    country_en="Unknown", country_ar="غير محدد",
    region_en="Unknown", nationality_en="Unknown", nationality_ar="غير محدد",
    confidence=0.0, method="unknown",
)


def _build_result(idx: int, confidence: float, method: str, raw: str) -> NationalityClassification:
    e = _COUNTRY_DATA[idx]
    return NationalityClassification(
        m49_code=e["m49"],
        iso_alpha3=e["iso3"],
        country_en=e["country_en"],
        country_ar=e["country_ar"],
        region_en=e["region_en"],
        nationality_en=e["nat_en"],
        nationality_ar=e["nat_ar"],
        confidence=confidence,
        method=method,
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class NationalityClassifier:
    """
    Classify a free-text nationality mention to a UN M49 country code.

    Stateless — no models loaded, no network calls.  Instantiation is free.
    """

    def classify(self, text: str) -> NationalityClassification:
        """
        Return the best UN M49 match for *text*.

        Parameters
        ----------
        text : str
            Raw text from the respondent (nationality field or free text).
            Can be a country name, demonym, or short phrase in any supported
            language.

        Returns
        -------
        NationalityClassification
            Always returns a result.  Check ``.method == "unknown"`` or
            ``.confidence == 0.0`` to detect no-match.
        """
        if not text or not text.strip():
            return _UNKNOWN

        raw   = text.strip()
        lower = raw.lower()

        # ── Stage 0: ISO-3 direct lookup (e.g. "IND", "ARE") ─────────────
        iso3_candidate = re.search(r"\b([a-z]{3})\b", lower)
        if iso3_candidate:
            token = iso3_candidate.group(1)
            if token in _ISO3_INDEX:
                return _build_result(_ISO3_INDEX[token], 1.0, "exact", raw)

        # ── Stage 0b: M49 numeric code (e.g. "784", "356") ───────────────
        m49_candidate = re.search(r"\b(\d{3})\b", lower)
        if m49_candidate:
            token = m49_candidate.group(1)
            if token in _M49_INDEX:
                return _build_result(_M49_INDEX[token], 1.0, "exact", raw)

        # ── Stage 1: multi-word alias phrases ────────────────────────────
        # Check full phrase first (e.g. "south africa", "sri lanka")
        for phrase_len in (4, 3, 2):
            tokens = re.findall(r"[a-z\u0600-\u06ff]{2,}", lower)
            for start in range(len(tokens) - phrase_len + 1):
                phrase = " ".join(tokens[start : start + phrase_len])
                if phrase in _ALIAS_INDEX:
                    return _build_result(_ALIAS_INDEX[phrase], 0.95, "alias", raw)

        # ── Stage 2: single-token alias lookup ───────────────────────────
        scores: dict[int, int] = {}
        tokens = re.findall(r"[a-z\u0600-\u06ff]{2,}", lower)
        for token in tokens:
            if token in _ALIAS_INDEX:
                idx = _ALIAS_INDEX[token]
                scores[idx] = scores.get(idx, 0) + 1

        if scores:
            best_idx   = max(scores, key=scores.__getitem__)
            best_score = scores[best_idx]
            total      = len(tokens) or 1
            confidence = min(0.90, 0.50 + 0.40 * (best_score / total))
            method     = "alias" if confidence >= 0.80 else "token"
            return _build_result(best_idx, confidence, method, raw)

        # ── No match ──────────────────────────────────────────────────────
        _logger.debug("NationalityClassifier: no match for %r", raw)
        result = _UNKNOWN
        result.raw_text = raw
        return result

    def classify_by_m49(self, m49_code: str) -> NationalityClassification:
        """Direct lookup by M49 numeric code string (e.g. '784')."""
        idx = _M49_INDEX.get(m49_code.strip().zfill(3))
        if idx is not None:
            return _build_result(idx, 1.0, "exact", m49_code)
        return _UNKNOWN

    def classify_by_iso3(self, iso3: str) -> NationalityClassification:
        """Direct lookup by ISO 3166-1 alpha-3 code (e.g. 'ARE')."""
        idx = _ISO3_INDEX.get(iso3.strip().lower())
        if idx is not None:
            return _build_result(idx, 1.0, "exact", iso3)
        return _UNKNOWN
