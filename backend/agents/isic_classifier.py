"""
backend/agents/isic_classifier.py

ISIC Rev.4 (International Standard Industrial Classification) industry
classifier for the LFS survey.

Pipeline
--------
Stage 1 – Keyword lookup
    Fast exact / partial match against an embedded ISIC Rev.4 class table
    covering all major sections down to 4-digit class codes.
    Returns the best matching class and its full ancestry.

Stage 2 – LLM re-ranking (Ollama / GPT-4o-mini, TaskType.GENERAL)
    A CrewAI agent selects the single best ISIC class (4-digit) from the
    top-3 keyword candidates and returns structured JSON.
    Skipped when keyword similarity is unambiguous (≥ 0.85).

Output
------
ISICClassification
    .section        – "A"–"U"  (one-letter section)
    .section_title  – e.g. "Information and Communication"
    .division_code  – two-digit string e.g. "62"
    .division_title – e.g. "Computer programming, consultancy and related activities"
    .group_code     – three-digit string e.g. "620"
    .group_title    – e.g. "Computer programming, consultancy and related activities"
    .class_code     – four-digit string e.g. "6201"
    .class_title    – e.g. "Computer programming activities"
    .confidence     – float 0–1
    .method         – "keyword" | "llm"

Usage
-----
from backend.agents.isic_classifier import ISICClassifier

clf    = ISICClassifier()
result = clf.classify("I work at a software company building mobile apps")
print(result.class_code)    # "6201"
print(result.class_title)   # "Computer programming activities"
print(result.section)       # "J"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from crewai import Agent, Crew, Task

from backend.agents.classifier_methods import NOT_IMPLEMENTED_METHODS, NOT_IMPLEMENTED_REASON
from backend.llm.llm_client import TaskType, get_llm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISIC Rev.4 — full 4-level hierarchy entries
# Each entry: section → division (2-digit) → group (3-digit) → class (4-digit)
# Source: UN Statistics Division ISIC Rev.4 official publication
# ---------------------------------------------------------------------------
_ISIC_DATA: list[dict] = [

    # ── A – Agriculture, Forestry and Fishing ────────────────────────────────
    {"section": "A", "section_title": "Agriculture, Forestry and Fishing",
     "division_code": "01", "division_title": "Crop and animal production, hunting and related service activities",
     "group_code": "011", "group_title": "Growing of non-perennial crops",
     "class_code": "0111", "class_title": "Growing of cereals, leguminous crops and oil seeds",
     "keywords": "farm farming wheat rice cereal grain crop field harvest agricultural زراعة محصول حبوب قمح أرز"},
    {"section": "A", "section_title": "Agriculture, Forestry and Fishing",
     "division_code": "01", "division_title": "Crop and animal production, hunting and related service activities",
     "group_code": "013", "group_title": "Growing of vegetables and melons, roots and tubers",
     "class_code": "0130", "class_title": "Growing of vegetables and melons, roots and tubers",
     "keywords": "vegetable fruit garden horticulture greenhouse خضروات فاكهة بستنة"},
    {"section": "A", "section_title": "Agriculture, Forestry and Fishing",
     "division_code": "01", "division_title": "Crop and animal production, hunting and related service activities",
     "group_code": "014", "group_title": "Animal production",
     "class_code": "0141", "class_title": "Raising of cattle and buffaloes",
     "keywords": "livestock cattle sheep poultry animal farm dairy milk حيوانات ماشية ألبان دجاج"},
    {"section": "A", "section_title": "Agriculture, Forestry and Fishing",
     "division_code": "02", "division_title": "Forestry and logging",
     "group_code": "021", "group_title": "Silviculture and other forestry activities",
     "class_code": "0210", "class_title": "Silviculture and other forestry activities",
     "keywords": "forest forestry logging timber wood lumber tree غابة أخشاب تحريج"},
    {"section": "A", "section_title": "Agriculture, Forestry and Fishing",
     "division_code": "03", "division_title": "Fishing and aquaculture",
     "group_code": "031", "group_title": "Fishing",
     "class_code": "0311", "class_title": "Marine fishing",
     "keywords": "fish fishing fisherman seafood marine ocean sea صيد سمك بحر أسماك"},
    {"section": "A", "section_title": "Agriculture, Forestry and Fishing",
     "division_code": "03", "division_title": "Fishing and aquaculture",
     "group_code": "032", "group_title": "Aquaculture",
     "class_code": "0321", "class_title": "Marine aquaculture",
     "keywords": "aquaculture fish farm shrimp prawn seafood breeding استزراع سمكي"},

    # ── B – Mining and Quarrying ───────────────────────────────────────────────
    {"section": "B", "section_title": "Mining and Quarrying",
     "division_code": "05", "division_title": "Mining of coal and lignite",
     "group_code": "051", "group_title": "Mining of hard coal",
     "class_code": "0510", "class_title": "Mining of hard coal",
     "keywords": "coal mine mining lignite تعدين فحم"},
    {"section": "B", "section_title": "Mining and Quarrying",
     "division_code": "06", "division_title": "Extraction of crude petroleum and natural gas",
     "group_code": "061", "group_title": "Extraction of crude petroleum",
     "class_code": "0610", "class_title": "Extraction of crude petroleum",
     "keywords": "oil petroleum extraction drilling well oilfield refinery crude ADNOC ARAMCO نفط بترول استخراج حفر بئر نفطي"},
    {"section": "B", "section_title": "Mining and Quarrying",
     "division_code": "06", "division_title": "Extraction of crude petroleum and natural gas",
     "group_code": "062", "group_title": "Extraction of natural gas",
     "class_code": "0620", "class_title": "Extraction of natural gas",
     "keywords": "natural gas extraction gas field LNG pipeline غاز طبيعي استخراج أنابيب"},
    {"section": "B", "section_title": "Mining and Quarrying",
     "division_code": "08", "division_title": "Other mining and quarrying",
     "group_code": "081", "group_title": "Quarrying of stone, sand and clay",
     "class_code": "0810", "class_title": "Quarrying of stone, sand and clay",
     "keywords": "quarry stone sand gravel mineral ore mining تعدين حجارة رمل معادن"},

    # ── C – Manufacturing ─────────────────────────────────────────────────────
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "10", "division_title": "Manufacture of food products",
     "group_code": "101", "group_title": "Processing and preserving of meat",
     "class_code": "1010", "class_title": "Processing and preserving of meat and meat products",
     "keywords": "meat processing slaughterhouse butcher food factory لحوم معالجة karkhana sanat utpadan pabrika pagmamanupaktura manupaktura"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "10", "division_title": "Manufacture of food products",
     "group_code": "107", "group_title": "Manufacture of bakery and farinaceous products",
     "class_code": "1071", "class_title": "Manufacture of bread; manufacture of fresh pastry goods and cakes",
     "keywords": "bakery bread biscuit pastry food factory cake خبز مخبز حلويات تصنيع غذاء"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "10", "division_title": "Manufacture of food products",
     "group_code": "105", "group_title": "Manufacture of dairy products",
     "class_code": "1050", "class_title": "Manufacture of dairy products",
     "keywords": "dairy milk cheese butter yogurt cream food factory ألبان حليب جبن زبدة"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "11", "division_title": "Manufacture of beverages",
     "group_code": "110", "group_title": "Manufacture of beverages",
     "class_code": "1104", "class_title": "Manufacture of soft drinks; production of mineral waters and other bottled waters",
     "keywords": "beverage drink soft drink water factory bottling مشروبات مياه مصنع"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "13", "division_title": "Manufacture of textiles",
     "group_code": "139", "group_title": "Manufacture of other textiles",
     "class_code": "1392", "class_title": "Manufacture of made-up textile articles, except apparel",
     "keywords": "textile fabric weaving spinning garment clothing سجاد نسيج قماش ملابس تصنيع"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "14", "division_title": "Manufacture of wearing apparel",
     "group_code": "141", "group_title": "Manufacture of wearing apparel, except fur apparel",
     "class_code": "1410", "class_title": "Manufacture of wearing apparel, except fur apparel",
     "keywords": "clothing apparel fashion garment sewing tailoring ملابس خياطة أزياء"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "20", "division_title": "Manufacture of chemicals and chemical products",
     "group_code": "201", "group_title": "Manufacture of basic chemicals, fertilizers and nitrogen compounds",
     "class_code": "2011", "class_title": "Manufacture of basic chemicals",
     "keywords": "chemical factory laboratory compound industrial كيماويات مصنع تصنيع مختبر"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "21", "division_title": "Manufacture of basic pharmaceutical products and preparations",
     "group_code": "210", "group_title": "Manufacture of basic pharmaceutical products and preparations",
     "class_code": "2100", "class_title": "Manufacture of basic pharmaceutical products and preparations",
     "keywords": "pharmaceutical drug medicine factory laboratory R&D pharma أدوية مستحضرات صيدلانية"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "26", "division_title": "Manufacture of computer, electronic and optical products",
     "group_code": "261", "group_title": "Manufacture of electronic components and boards",
     "class_code": "2610", "class_title": "Manufacture of electronic components and boards",
     "keywords": "electronics semiconductor chip circuit board manufacture hardware تصنيع الكترونيات رقائق"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "26", "division_title": "Manufacture of computer, electronic and optical products",
     "group_code": "262", "group_title": "Manufacture of computers and peripheral equipment",
     "class_code": "2620", "class_title": "Manufacture of computers and peripheral equipment",
     "keywords": "computer hardware PC laptop server manufacture assembly تصنيع حاسوب أجهزة"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "27", "division_title": "Manufacture of electrical equipment",
     "group_code": "271", "group_title": "Manufacture of electric motors, generators, transformers and electricity distribution and control apparatus",
     "class_code": "2710", "class_title": "Manufacture of electric motors, generators, transformers",
     "keywords": "electrical equipment motor generator transformer manufacturer تصنيع كهربائي محولات"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "28", "division_title": "Manufacture of machinery and equipment n.e.c.",
     "group_code": "282", "group_title": "Manufacture of other general-purpose machinery",
     "class_code": "2829", "class_title": "Manufacture of other general-purpose machinery n.e.c.",
     "keywords": "machinery machine equipment industrial factory engineer تصنيع آلات معدات مصنع"},
    {"section": "C", "section_title": "Manufacturing",
     "division_code": "29", "division_title": "Manufacture of motor vehicles, trailers and semi-trailers",
     "group_code": "291", "group_title": "Manufacture of motor vehicles",
     "class_code": "2910", "class_title": "Manufacture of motor vehicles",
     "keywords": "car automobile vehicle motor assembly manufacturer تصنيع سيارات مركبات"},

    # ── D – Electricity, Gas, Steam ───────────────────────────────────────────
    {"section": "D", "section_title": "Electricity, Gas, Steam and Air Conditioning Supply",
     "division_code": "35", "division_title": "Electricity, gas, steam and air conditioning supply",
     "group_code": "351", "group_title": "Electric power generation, transmission and distribution",
     "class_code": "3510", "class_title": "Electric power generation, transmission and distribution",
     "keywords": "electricity power plant utility energy grid generation DEWA ADDC كهرباء طاقة محطة"},
    {"section": "D", "section_title": "Electricity, Gas, Steam and Air Conditioning Supply",
     "division_code": "35", "division_title": "Electricity, gas, steam and air conditioning supply",
     "group_code": "352", "group_title": "Manufacture of gas; distribution of gaseous fuels through mains",
     "class_code": "3520", "class_title": "Manufacture of gas; distribution of gaseous fuels through mains",
     "keywords": "gas distribution utility pipeline steam cooling غاز توزيع تبريد بخار"},

    # ── E – Water Supply, Sewerage, Waste ─────────────────────────────────────
    {"section": "E", "section_title": "Water Supply; Sewerage, Waste Management and Remediation Activities",
     "division_code": "36", "division_title": "Water collection, treatment and supply",
     "group_code": "360", "group_title": "Water collection, treatment and supply",
     "class_code": "3600", "class_title": "Water collection, treatment and supply",
     "keywords": "water supply treatment purification desalination DEWA مياه تنقية تحلية"},
    {"section": "E", "section_title": "Water Supply; Sewerage, Waste Management and Remediation Activities",
     "division_code": "38", "division_title": "Waste collection, treatment and disposal activities; materials recovery",
     "group_code": "381", "group_title": "Waste collection",
     "class_code": "3811", "class_title": "Collection of non-hazardous waste",
     "keywords": "waste garbage recycling disposal environment sanitation collection نفايات مخلفات بيئة تجميع"},
    {"section": "E", "section_title": "Water Supply; Sewerage, Waste Management and Remediation Activities",
     "division_code": "37", "division_title": "Sewerage",
     "group_code": "370", "group_title": "Sewerage",
     "class_code": "3700", "class_title": "Sewerage",
     "keywords": "sewage wastewater treatment sewerage صرف صحي مياه عادمة معالجة"},

    # ── F – Construction ──────────────────────────────────────────────────────
    {"section": "F", "section_title": "Construction",
     "division_code": "41", "division_title": "Construction of buildings",
     "group_code": "410", "group_title": "Construction of buildings",
     "class_code": "4100", "class_title": "Construction of buildings",
     "keywords": "construction building contractor builder site real estate residential commercial بناء مقاول إنشاء عمارة مشاريع tameer mazdoor nirman thekedar konstruksiyon gusali manggagawa"},
    {"section": "F", "section_title": "Construction",
     "division_code": "42", "division_title": "Civil engineering",
     "group_code": "421", "group_title": "Construction of roads and railways",
     "class_code": "4210", "class_title": "Construction of roads and motorways",
     "keywords": "civil engineering road bridge highway infrastructure مدني هندسة طرق جسور بنية تحتية"},
    {"section": "F", "section_title": "Construction",
     "division_code": "42", "division_title": "Civil engineering",
     "group_code": "422", "group_title": "Construction of utility projects",
     "class_code": "4220", "class_title": "Construction of utility projects",
     "keywords": "pipeline utility infrastructure water power project مشروع بنية تحتية"},
    {"section": "F", "section_title": "Construction",
     "division_code": "43", "division_title": "Specialised construction activities",
     "group_code": "431", "group_title": "Demolition and site preparation",
     "class_code": "4311", "class_title": "Demolition",
     "keywords": "demolition site preparation excavation هدم تحضير موقع حفر"},
    {"section": "F", "section_title": "Construction",
     "division_code": "43", "division_title": "Specialised construction activities",
     "group_code": "432", "group_title": "Electrical, plumbing and other construction installation activities",
     "class_code": "4321", "class_title": "Electrical installation",
     "keywords": "electrical installation wiring fit-out contractor كهرباء تمديد كابلات تركيب"},
    {"section": "F", "section_title": "Construction",
     "division_code": "43", "division_title": "Specialised construction activities",
     "group_code": "432", "group_title": "Electrical, plumbing and other construction installation activities",
     "class_code": "4322", "class_title": "Plumbing, heat and air-conditioning installation",
     "keywords": "plumbing HVAC air conditioning installation pipes سباكة تكييف أنابيب"},
    {"section": "F", "section_title": "Construction",
     "division_code": "43", "division_title": "Specialised construction activities",
     "group_code": "433", "group_title": "Building completion and finishing",
     "class_code": "4330", "class_title": "Building completion and finishing",
     "keywords": "finishing renovation fit-out interior painting تشطيب ديكور دهان داخلي"},

    # ── G – Wholesale and Retail Trade ────────────────────────────────────────
    {"section": "G", "section_title": "Wholesale and Retail Trade; Repair of Motor Vehicles",
     "division_code": "45", "division_title": "Wholesale and retail trade and repair of motor vehicles",
     "group_code": "451", "group_title": "Sale of motor vehicles",
     "class_code": "4510", "class_title": "Sale of motor vehicles",
     "keywords": "car dealership auto sales vehicle showroom dealer تجارة سيارات معرض وكيل"},
    {"section": "G", "section_title": "Wholesale and Retail Trade; Repair of Motor Vehicles",
     "division_code": "45", "division_title": "Wholesale and retail trade and repair of motor vehicles",
     "group_code": "452", "group_title": "Maintenance and repair of motor vehicles",
     "class_code": "4520", "class_title": "Maintenance and repair of motor vehicles",
     "keywords": "car repair garage mechanic service workshop صيانة سيارات ورشة ميكانيكي"},
    {"section": "G", "section_title": "Wholesale and Retail Trade; Repair of Motor Vehicles",
     "division_code": "46", "division_title": "Wholesale trade, except of motor vehicles and motorcycles",
     "group_code": "461", "group_title": "Wholesale on a fee or contract basis",
     "class_code": "4610", "class_title": "Wholesale on a fee or contract basis",
     "keywords": "wholesale trade distributor agent broker importer exporter merchant تجارة جملة توزيع وكيل"},
    {"section": "G", "section_title": "Wholesale and Retail Trade; Repair of Motor Vehicles",
     "division_code": "46", "division_title": "Wholesale trade, except of motor vehicles and motorcycles",
     "group_code": "464", "group_title": "Wholesale of household goods",
     "class_code": "4649", "class_title": "Wholesale of other household goods",
     "keywords": "wholesale household goods electronics furniture FMCG تجارة جملة"},
    {"section": "G", "section_title": "Wholesale and Retail Trade; Repair of Motor Vehicles",
     "division_code": "47", "division_title": "Retail trade, except of motor vehicles and motorcycles",
     "group_code": "471", "group_title": "Retail sale in non-specialised stores",
     "class_code": "4711", "class_title": "Retail sale in non-specialised stores with food, beverages or tobacco predominating",
     "keywords": "supermarket hypermarket grocery store mall retail food Carrefour LuLu تجزئة سوبرماركت بقالة dukan khareed tijaarat tindahan palengke bilhin"},
    {"section": "G", "section_title": "Wholesale and Retail Trade; Repair of Motor Vehicles",
     "division_code": "47", "division_title": "Retail trade, except of motor vehicles and motorcycles",
     "group_code": "471", "group_title": "Retail sale in non-specialised stores",
     "class_code": "4719", "class_title": "Other retail sale in non-specialised stores",
     "keywords": "retail store shop department store mall sales cashier clerk بيع تجزئة متجر مركز تسوق"},
    {"section": "G", "section_title": "Wholesale and Retail Trade; Repair of Motor Vehicles",
     "division_code": "47", "division_title": "Retail trade, except of motor vehicles and motorcycles",
     "group_code": "477", "group_title": "Retail sale of other goods in specialised stores",
     "class_code": "4771", "class_title": "Retail sale of clothing, footwear and leather articles in specialised stores",
     "keywords": "clothing fashion retail boutique shoes apparel store تجزئة ملابس أزياء بوتيك"},

    # ── H – Transportation and Storage ───────────────────────────────────────
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "49", "division_title": "Land transport and transport via pipelines",
     "group_code": "492", "group_title": "Other land transport",
     "class_code": "4921", "class_title": "Urban and suburban passenger land transport",
     "keywords": "bus taxi transport urban passenger RTA نقل حضري ركاب حافلة driver gaadi safar parivahan sasakyan transportasyon"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "49", "division_title": "Land transport and transport via pipelines",
     "group_code": "492", "group_title": "Other land transport",
     "class_code": "4923", "class_title": "Freight transport by road",
     "keywords": "truck freight transport road logistics delivery driver شاحنة نقل بري توصيل لوجستيات"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "50", "division_title": "Water transport",
     "group_code": "501", "group_title": "Sea and coastal water transport",
     "class_code": "5011", "class_title": "Sea and coastal passenger water transport",
     "keywords": "shipping maritime vessel ship port sailor cruise نقل بحري ملاحة سفينة ميناء"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "50", "division_title": "Water transport",
     "group_code": "502", "group_title": "Inland water transport",
     "class_code": "5022", "class_title": "Inland freight water transport",
     "keywords": "cargo ship port freight maritime شحن بحري ميناء"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "51", "division_title": "Air transport",
     "group_code": "511", "group_title": "Passenger air transport",
     "class_code": "5110", "class_title": "Passenger air transport",
     "keywords": "airline aviation pilot flight crew cabin attendant Emirates Etihad طيران مطار طيار مضيف"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "51", "division_title": "Air transport",
     "group_code": "512", "group_title": "Freight air transport",
     "class_code": "5120", "class_title": "Freight air transport",
     "keywords": "cargo air freight logistics airport DHL FedEx شحن جوي بضائع طيران"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "52", "division_title": "Warehousing and support activities for transportation",
     "group_code": "521", "group_title": "Warehousing and storage",
     "class_code": "5210", "class_title": "Warehousing and storage",
     "keywords": "warehouse storage logistics cold chain distribution centre مستودع تخزين لوجستيات"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "52", "division_title": "Warehousing and support activities for transportation",
     "group_code": "522", "group_title": "Support activities for transportation",
     "class_code": "5229", "class_title": "Other service activities incidental to transportation",
     "keywords": "freight forwarding courier delivery express shipping agent وكيل شحن توصيل سريع"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "53", "division_title": "Postal and courier activities",
     "group_code": "531", "group_title": "Postal activities",
     "class_code": "5310", "class_title": "Postal activities",
     "keywords": "post office mail postal delivery courier بريد بريد سريع توصيل طرود"},
    {"section": "H", "section_title": "Transportation and Storage",
     "division_code": "53", "division_title": "Postal and courier activities",
     "group_code": "532", "group_title": "Courier activities",
     "class_code": "5320", "class_title": "Courier activities, other than national post activities",
     "keywords": "courier express delivery last mile Aramex FedEx DHL شحن سريع توصيل"},

    # ── I – Accommodation and Food Service ────────────────────────────────────
    {"section": "I", "section_title": "Accommodation and Food Service Activities",
     "division_code": "55", "division_title": "Accommodation",
     "group_code": "551", "group_title": "Short term accommodation activities",
     "class_code": "5510", "class_title": "Short term accommodation activities",
     "keywords": "hotel motel resort accommodation hospitality front desk reception manager فندق ضيافة استقبال إدارة paryatan mehman siyaahat hotel bisita manlalakbay"},
    {"section": "I", "section_title": "Accommodation and Food Service Activities",
     "division_code": "56", "division_title": "Food and beverage service activities",
     "group_code": "561", "group_title": "Restaurants and mobile food service activities",
     "class_code": "5610", "class_title": "Restaurants and mobile food service activities",
     "keywords": "restaurant cafe dining food service waiter cook chef kitchen مطعم مقهى خدمة طعام طاه نادل"},
    {"section": "I", "section_title": "Accommodation and Food Service Activities",
     "division_code": "56", "division_title": "Food and beverage service activities",
     "group_code": "562", "group_title": "Event catering and other food service activities",
     "class_code": "5621", "class_title": "Event catering activities",
     "keywords": "catering event corporate buffet banquet ضيافة تقديم طعام فعاليات"},
    {"section": "I", "section_title": "Accommodation and Food Service Activities",
     "division_code": "56", "division_title": "Food and beverage service activities",
     "group_code": "563", "group_title": "Beverage serving activities",
     "class_code": "5630", "class_title": "Beverage serving activities",
     "keywords": "coffee shop bar barista beverage drinks café قهوة مقهى مشروبات"},

    # ── J – Information and Communication ─────────────────────────────────────
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "58", "division_title": "Publishing activities",
     "group_code": "581", "group_title": "Publishing of books, periodicals and other publishing activities",
     "class_code": "5813", "class_title": "Publishing of newspapers",
     "keywords": "newspaper publishing journalism media print reporter نشر صحافة إعلام جريدة"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "58", "division_title": "Publishing activities",
     "group_code": "582", "group_title": "Software publishing",
     "class_code": "5820", "class_title": "Software publishing",
     "keywords": "software publisher game application platform SaaS نشر برمجيات"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "60", "division_title": "Programming and broadcasting activities",
     "group_code": "601", "group_title": "Radio broadcasting",
     "class_code": "6010", "class_title": "Radio broadcasting",
     "keywords": "radio broadcasting media station راديو إذاعة بث"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "60", "division_title": "Programming and broadcasting activities",
     "group_code": "602", "group_title": "Television programming and broadcasting activities",
     "class_code": "6020", "class_title": "Television programming and broadcasting activities",
     "keywords": "television TV broadcast media production channel studio presenter تلفزيون بث قناة إعلام"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "61", "division_title": "Telecommunications",
     "group_code": "611", "group_title": "Wired telecommunications activities",
     "class_code": "6110", "class_title": "Wired telecommunications activities",
     "keywords": "telecom wired internet broadband fibre cable network provider اتصالات سلكية شبكة إنترنت"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "61", "division_title": "Telecommunications",
     "group_code": "612", "group_title": "Wireless telecommunications activities",
     "class_code": "6120", "class_title": "Wireless telecommunications activities",
     "keywords": "telecom mobile wireless network Etisalat du 5G cellular اتصالات لاسلكية جوال خلوي شبكة"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "62", "division_title": "Computer programming, consultancy and related activities",
     "group_code": "620", "group_title": "Computer programming, consultancy and related activities",
     "class_code": "6201", "class_title": "Computer programming activities",
     "keywords": "software developer programmer coding app mobile web development engineer frontend backend fullstack تقنية برمجيات مطور كود تطبيق software programmer IT praudyogiki suchana teknolohiya programador"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "62", "division_title": "Computer programming, consultancy and related activities",
     "group_code": "620", "group_title": "Computer programming, consultancy and related activities",
     "class_code": "6202", "class_title": "Computer consultancy and computer facilities management activities",
     "keywords": "IT consultant technology infrastructure cloud systems admin network DevOps استشارات تقنية بنية تحتية"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "62", "division_title": "Computer programming, consultancy and related activities",
     "group_code": "620", "group_title": "Computer programming, consultancy and related activities",
     "class_code": "6209", "class_title": "Other information technology and computer service activities",
     "keywords": "IT support helpdesk technical support technology company tech startup دعم تقني خدمات تقنية"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "63", "division_title": "Information service activities",
     "group_code": "631", "group_title": "Data processing, hosting and related activities; web portals",
     "class_code": "6311", "class_title": "Data processing, hosting and related activities",
     "keywords": "data analytics cloud computing hosting AWS Azure big data AI machine learning خدمات بيانات سحابة ذكاء اصطناعي"},
    {"section": "J", "section_title": "Information and Communication",
     "division_code": "63", "division_title": "Information service activities",
     "group_code": "631", "group_title": "Data processing, hosting and related activities; web portals",
     "class_code": "6312", "class_title": "Web portals",
     "keywords": "web portal platform marketplace e-commerce online website بوابة إلكترونية تجارة إلكترونية"},

    # ── K – Financial and Insurance Activities ────────────────────────────────
    {"section": "K", "section_title": "Financial and Insurance Activities",
     "division_code": "64", "division_title": "Financial service activities, except insurance and pension funding",
     "group_code": "641", "group_title": "Monetary intermediation",
     "class_code": "6411", "class_title": "Central banking",
     "keywords": "central bank monetary policy CBUAE reserve بنك مركزي سياسة نقدية bank maaliyat bima paisa bangko pera seguro pananalapi"},
    {"section": "K", "section_title": "Financial and Insurance Activities",
     "division_code": "64", "division_title": "Financial service activities, except insurance and pension funding",
     "group_code": "641", "group_title": "Monetary intermediation",
     "class_code": "6419", "class_title": "Other monetary intermediation",
     "keywords": "bank banking commercial retail investment fund capital broker ADCB Emirates NBD ENBD بنك تجاري استثمار مصرف"},
    {"section": "K", "section_title": "Financial and Insurance Activities",
     "division_code": "64", "division_title": "Financial service activities, except insurance and pension funding",
     "group_code": "649", "group_title": "Other financial service activities",
     "class_code": "6491", "class_title": "Financial leasing",
     "keywords": "leasing finance credit lending microfinance تمويل إيجار مالي قرض"},
    {"section": "K", "section_title": "Financial and Insurance Activities",
     "division_code": "64", "division_title": "Financial service activities, except insurance and pension funding",
     "group_code": "649", "group_title": "Other financial service activities",
     "class_code": "6499", "class_title": "Other financial service activities n.e.c.",
     "keywords": "fintech payment transfer money exchange forex remittance تحويل مالي صرف عملات"},
    {"section": "K", "section_title": "Financial and Insurance Activities",
     "division_code": "65", "division_title": "Insurance, reinsurance and pension funding",
     "group_code": "651", "group_title": "Insurance",
     "class_code": "6512", "class_title": "Non-life insurance",
     "keywords": "insurance health life motor car property casualty actuary تأمين صحي سيارة ممتلكات"},
    {"section": "K", "section_title": "Financial and Insurance Activities",
     "division_code": "65", "division_title": "Insurance, reinsurance and pension funding",
     "group_code": "653", "group_title": "Pension funding",
     "class_code": "6530", "class_title": "Pension funding",
     "keywords": "pension retirement fund gratuity معاشات تقاعد صندوق"},
    {"section": "K", "section_title": "Financial and Insurance Activities",
     "division_code": "66", "division_title": "Activities auxiliary to financial services and insurance activities",
     "group_code": "661", "group_title": "Activities auxiliary to financial services",
     "class_code": "6612", "class_title": "Security and commodity contracts brokerage",
     "keywords": "stock exchange brokerage forex securities trading DFM ADX تداول وساطة مالية بورصة"},
    {"section": "K", "section_title": "Financial and Insurance Activities",
     "division_code": "66", "division_title": "Activities auxiliary to financial services and insurance activities",
     "group_code": "661", "group_title": "Activities auxiliary to financial services",
     "class_code": "6619", "class_title": "Other activities auxiliary to financial services",
     "keywords": "financial advisory wealth management private equity consulting مالي استشارات إدارة ثروات"},

    # ── L – Real Estate ───────────────────────────────────────────────────────
    {"section": "L", "section_title": "Real Estate Activities",
     "division_code": "68", "division_title": "Real estate activities",
     "group_code": "681", "group_title": "Real estate activities with own or leased property",
     "class_code": "6810", "class_title": "Real estate activities with own or leased property",
     "keywords": "real estate property developer landlord investment Emaar Aldar عقارات تطوير مطور"},
    {"section": "L", "section_title": "Real Estate Activities",
     "division_code": "68", "division_title": "Real estate activities",
     "group_code": "682", "group_title": "Real estate activities on a fee or contract basis",
     "class_code": "6820", "class_title": "Real estate activities on a fee or contract basis",
     "keywords": "real estate agent broker rental leasing property management وسيط عقاري إيجار إدارة عقارية"},

    # ── M – Professional, Scientific and Technical Activities ─────────────────
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "69", "division_title": "Legal and accounting activities",
     "group_code": "691", "group_title": "Legal activities",
     "class_code": "6910", "class_title": "Legal activities",
     "keywords": "law lawyer attorney legal advocate counsel solicitor notary محامي قانون مستشار قانوني"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "69", "division_title": "Legal and accounting activities",
     "group_code": "692", "group_title": "Accounting, bookkeeping and auditing activities; tax consultancy",
     "class_code": "6920", "class_title": "Accounting, bookkeeping and auditing activities; tax consultancy",
     "keywords": "accountant accounting audit CPA tax bookkeeping finance KPMG Deloitte محاسب مراجع حسابات ضريبة"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "70", "division_title": "Activities of head offices; management consultancy activities",
     "group_code": "701", "group_title": "Activities of head offices",
     "class_code": "7010", "class_title": "Activities of head offices",
     "keywords": "headquarters head office holding company group corporate HQ مقر رئيسي"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "70", "division_title": "Activities of head offices; management consultancy activities",
     "group_code": "702", "group_title": "Management consultancy activities",
     "class_code": "7020", "class_title": "Management consultancy activities",
     "keywords": "consulting management advisory strategy McKinsey BCG Accenture استشارات إدارة استراتيجية"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "71", "division_title": "Architectural and engineering activities",
     "group_code": "711", "group_title": "Architectural and engineering activities and related technical consultancy",
     "class_code": "7110", "class_title": "Architectural and engineering activities and related technical consultancy",
     "keywords": "architect architecture engineering design structural civil consultant هندسة معمارية تصميم مستشار هندسي"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "71", "division_title": "Architectural and engineering activities",
     "group_code": "712", "group_title": "Technical testing and analysis",
     "class_code": "7120", "class_title": "Technical testing and analysis",
     "keywords": "testing laboratory inspection quality control certification اختبار مختبر فحص جودة"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "72", "division_title": "Scientific research and development",
     "group_code": "721", "group_title": "Research and experimental development on natural sciences and engineering",
     "class_code": "7210", "class_title": "Research and experimental development on natural sciences and engineering",
     "keywords": "research laboratory scientist R&D development university innovation STEM بحث علمي مختبر جامعة ابتكار"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "72", "division_title": "Scientific research and development",
     "group_code": "722", "group_title": "Research and experimental development on social sciences and humanities",
     "class_code": "7220", "class_title": "Research and experimental development on social sciences and humanities",
     "keywords": "social research survey statistics economics think tank بحث اجتماعي إحصاء اقتصاد"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "73", "division_title": "Advertising and market research",
     "group_code": "731", "group_title": "Advertising",
     "class_code": "7311", "class_title": "Advertising agencies",
     "keywords": "advertising marketing agency brand campaign creative digital media إعلانات تسويق وكالة إعلامية"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "73", "division_title": "Advertising and market research",
     "group_code": "732", "group_title": "Market research and public opinion polling",
     "class_code": "7320", "class_title": "Market research and public opinion polling",
     "keywords": "market research survey data analysis polling consumer insights بحث سوقي استطلاع بيانات"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "74", "division_title": "Other professional, scientific and technical activities",
     "group_code": "741", "group_title": "Specialized design activities",
     "class_code": "7410", "class_title": "Specialised design activities",
     "keywords": "design graphic interior fashion product UX UI designer تصميم جرافيك ديكور داخلي"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "74", "division_title": "Other professional, scientific and technical activities",
     "group_code": "742", "group_title": "Photographic activities",
     "class_code": "7420", "class_title": "Photographic activities",
     "keywords": "photography photographer videography media production تصوير فوتوغرافي إنتاج إعلامي"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "74", "division_title": "Other professional, scientific and technical activities",
     "group_code": "743", "group_title": "Translation and interpretation activities",
     "class_code": "7430", "class_title": "Translation and interpretation activities",
     "keywords": "translation interpreter language translator Arabic English ترجمة مترجم لغة"},
    {"section": "M", "section_title": "Professional, Scientific and Technical Activities",
     "division_code": "75", "division_title": "Veterinary activities",
     "group_code": "750", "group_title": "Veterinary activities",
     "class_code": "7500", "class_title": "Veterinary activities",
     "keywords": "veterinary vet animal clinic pet طبيب بيطري حيوانات عيادة بيطرية"},

    # ── N – Administrative and Support Service Activities ─────────────────────
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "77", "division_title": "Rental and leasing activities",
     "group_code": "771", "group_title": "Renting and leasing of motor vehicles",
     "class_code": "7710", "class_title": "Renting and leasing of motor vehicles",
     "keywords": "car rental leasing vehicle hire Hertz Avis تأجير سيارات إيجار"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "77", "division_title": "Rental and leasing activities",
     "group_code": "773", "group_title": "Renting and leasing of other machinery, equipment and tangible goods",
     "class_code": "7739", "class_title": "Renting and leasing of other machinery, equipment and tangible goods n.e.c.",
     "keywords": "equipment rental leasing machinery hire تأجير معدات إيجار"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "78", "division_title": "Employment activities",
     "group_code": "781", "group_title": "Activities of employment placement agencies",
     "class_code": "7810", "class_title": "Activities of employment placement agencies",
     "keywords": "recruitment HR human resources staffing agency placement headhunter توظيف استقطاب موارد بشرية وكالة"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "78", "division_title": "Employment activities",
     "group_code": "782", "group_title": "Temporary employment agency activities",
     "class_code": "7820", "class_title": "Temporary employment agency activities",
     "keywords": "temporary staffing manpower outsourcing contract workers عمالة مؤقتة استعانة بمصادر خارجية"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "78", "division_title": "Employment activities",
     "group_code": "783", "group_title": "Human resources provision and management of human resources functions",
     "class_code": "7830", "class_title": "Human resources provision and management",
     "keywords": "HR payroll benefits compensation talent management موارد بشرية رواتب مزايا"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "79", "division_title": "Travel agency, tour operator and other reservation service activities",
     "group_code": "791", "group_title": "Travel agency and tour operator activities",
     "class_code": "7911", "class_title": "Travel agency activities",
     "keywords": "travel agency tour operator tourism package holiday visa وكالة سفر سياحة رحلات تأشيرة"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "80", "division_title": "Security and investigation activities",
     "group_code": "801", "group_title": "Private security activities",
     "class_code": "8010", "class_title": "Private security activities",
     "keywords": "security guard protection patrol surveillance investigation أمن حراسة حماية"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "81", "division_title": "Services to buildings and landscape activities",
     "group_code": "811", "group_title": "Combined facilities support activities",
     "class_code": "8110", "class_title": "Combined facilities support activities",
     "keywords": "facilities management FM property maintenance operations خدمات منشآت صيانة"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "81", "division_title": "Services to buildings and landscape activities",
     "group_code": "812", "group_title": "Cleaning activities",
     "class_code": "8121", "class_title": "General cleaning of buildings",
     "keywords": "cleaning janitor housekeeping maid service building نظافة تنظيف حارس"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "82", "division_title": "Office administrative, office support and other business support activities",
     "group_code": "821", "group_title": "Office administrative and support activities",
     "class_code": "8211", "class_title": "Combined office administrative service activities",
     "keywords": "admin administrative office secretary receptionist clerical data entry إداري سكرتارية استقبال إدخال بيانات"},
    {"section": "N", "section_title": "Administrative and Support Service Activities",
     "division_code": "82", "division_title": "Office administrative, office support and other business support activities",
     "group_code": "822", "group_title": "Activities of call centres",
     "class_code": "8220", "class_title": "Activities of call centres",
     "keywords": "call centre customer service helpdesk BPO contact centre خدمة عملاء مركز اتصال"},

    # ── O – Public Administration and Defence ─────────────────────────────────
    {"section": "O", "section_title": "Public Administration and Defence; Compulsory Social Security",
     "division_code": "84", "division_title": "Public administration and defence; compulsory social security",
     "group_code": "841", "group_title": "Administration of the State and the economic and social policy of the community",
     "class_code": "8411", "class_title": "General public administration activities",
     "keywords": "government public sector ministry civil servant administration إدارة حكومية وزارة موظف حكومي خدمة مدنية sarkar hukumat daftar gobyerno opisyal serbisyo publiko"},
    {"section": "O", "section_title": "Public Administration and Defence; Compulsory Social Security",
     "division_code": "84", "division_title": "Public administration and defence; compulsory social security",
     "group_code": "842", "group_title": "Provision of services to the community as a whole",
     "class_code": "8422", "class_title": "Defence activities",
     "keywords": "military army defence armed forces soldier officer دفاع جيش عسكري ضابط"},
    {"section": "O", "section_title": "Public Administration and Defence; Compulsory Social Security",
     "division_code": "84", "division_title": "Public administration and defence; compulsory social security",
     "group_code": "842", "group_title": "Provision of services to the community as a whole",
     "class_code": "8424", "class_title": "Public order and safety activities",
     "keywords": "police law enforcement security officer patrol شرطة أمن عام حارس"},
    {"section": "O", "section_title": "Public Administration and Defence; Compulsory Social Security",
     "division_code": "84", "division_title": "Public administration and defence; compulsory social security",
     "group_code": "843", "group_title": "Compulsory social security activities",
     "class_code": "8430", "class_title": "Compulsory social security activities",
     "keywords": "social security pension GPSSA PIFSS تأمين اجتماعي معاشات ضمان"},

    # ── P – Education ──────────────────────────────────────────────────────────
    {"section": "P", "section_title": "Education",
     "division_code": "85", "division_title": "Education",
     "group_code": "851", "group_title": "Pre-primary education",
     "class_code": "8510", "class_title": "Pre-primary education",
     "keywords": "kindergarten nursery preschool early childhood روضة أطفال حضانة taleem madrasa shiksha vidyalaya ustaz paaralan guro edukasyon"},
    {"section": "P", "section_title": "Education",
     "division_code": "85", "division_title": "Education",
     "group_code": "852", "group_title": "Primary education",
     "class_code": "8521", "class_title": "General primary education",
     "keywords": "primary school teacher elementary grade school مدرسة ابتدائي معلم"},
    {"section": "P", "section_title": "Education",
     "division_code": "85", "division_title": "Education",
     "group_code": "853", "group_title": "Secondary education",
     "class_code": "8531", "class_title": "General secondary education",
     "keywords": "secondary school high school teacher subject تعليم ثانوي مدرسة ثانوية معلم"},
    {"section": "P", "section_title": "Education",
     "division_code": "85", "division_title": "Education",
     "group_code": "854", "group_title": "Higher education",
     "class_code": "8542", "class_title": "Tertiary education",
     "keywords": "university college professor lecturer academic faculty higher education جامعة كلية أستاذ أكاديمي تعليم عالٍ"},
    {"section": "P", "section_title": "Education",
     "division_code": "85", "division_title": "Education",
     "group_code": "855", "group_title": "Other education",
     "class_code": "8559", "class_title": "Other education n.e.c.",
     "keywords": "training vocational institute language school driving corporate training تدريب مهني معهد لغات"},
    {"section": "P", "section_title": "Education",
     "division_code": "85", "division_title": "Education",
     "group_code": "856", "group_title": "Educational support activities",
     "class_code": "8560", "class_title": "Educational support activities",
     "keywords": "tutoring coaching educational support curriculum counselling دعم تعليمي إرشاد"},

    # ── Q – Human Health and Social Work ─────────────────────────────────────
    {"section": "Q", "section_title": "Human Health and Social Work Activities",
     "division_code": "86", "division_title": "Human health activities",
     "group_code": "861", "group_title": "Hospital activities",
     "class_code": "8610", "class_title": "Hospital activities",
     "keywords": "hospital doctor nurse physician surgeon specialist emergency ward inpatient government hospital مستشفى طبيب ممرض جراح قسم طوارئ مستشفى حكومي"},
    {"section": "Q", "section_title": "Human Health and Social Work Activities",
     "division_code": "86", "division_title": "Human health activities",
     "group_code": "862", "group_title": "Medical and dental practice activities",
     "class_code": "8621", "class_title": "General medical practice activities",
     "keywords": "clinic general practitioner GP family doctor primary care عيادة طبيب عام رعاية صحية أولية"},
    {"section": "Q", "section_title": "Human Health and Social Work Activities",
     "division_code": "86", "division_title": "Human health activities",
     "group_code": "862", "group_title": "Medical and dental practice activities",
     "class_code": "8622", "class_title": "Specialist medical practice activities",
     "keywords": "specialist consultant cardiologist dermatologist paediatrician specialist clinic طبيب متخصص استشاري عيادة تخصصية"},
    {"section": "Q", "section_title": "Human Health and Social Work Activities",
     "division_code": "86", "division_title": "Human health activities",
     "group_code": "862", "group_title": "Medical and dental practice activities",
     "class_code": "8623", "class_title": "Dental practice activities",
     "keywords": "dentist dental clinic teeth orthodontist طبيب أسنان عيادة أسنان"},
    {"section": "Q", "section_title": "Human Health and Social Work Activities",
     "division_code": "86", "division_title": "Human health activities",
     "group_code": "869", "group_title": "Other human health activities",
     "class_code": "8690", "class_title": "Other human health activities",
     "keywords": "pharmacy pharmacist nurse paramedic physiotherapy radiology lab صيدلة صيدلاني ممرض علاج طبيعي مختبر"},
    {"section": "Q", "section_title": "Human Health and Social Work Activities",
     "division_code": "87", "division_title": "Residential care activities",
     "group_code": "871", "group_title": "Residential nursing care facilities",
     "class_code": "8710", "class_title": "Residential nursing care facilities",
     "keywords": "nursing home care elderly residential رعاية منزلية مسنين مقيم"},
    {"section": "Q", "section_title": "Human Health and Social Work Activities",
     "division_code": "88", "division_title": "Social work activities without accommodation",
     "group_code": "881", "group_title": "Social work activities without accommodation for the elderly and disabled",
     "class_code": "8810", "class_title": "Social work activities without accommodation for the elderly and disabled",
     "keywords": "social work welfare NGO charity community elderly care رعاية اجتماعية خيرية مجتمع"},
    {"section": "Q", "section_title": "Human Health and Social Work Activities",
     "division_code": "88", "division_title": "Social work activities without accommodation",
     "group_code": "889", "group_title": "Other social work activities without accommodation",
     "class_code": "8899", "class_title": "Other social work activities without accommodation n.e.c.",
     "keywords": "social services community development women children youth خدمات اجتماعية تنمية مجتمعية"},

    # ── R – Arts, Entertainment and Recreation ────────────────────────────────
    {"section": "R", "section_title": "Arts, Entertainment and Recreation",
     "division_code": "90", "division_title": "Creative, arts and entertainment activities",
     "group_code": "900", "group_title": "Creative, arts and entertainment activities",
     "class_code": "9001", "class_title": "Performing arts",
     "keywords": "arts music theatre performance entertainment actor singer فنون موسيقى مسرح ترفيه فنان"},
    {"section": "R", "section_title": "Arts, Entertainment and Recreation",
     "division_code": "90", "division_title": "Creative, arts and entertainment activities",
     "group_code": "900", "group_title": "Creative, arts and entertainment activities",
     "class_code": "9003", "class_title": "Artistic creation",
     "keywords": "artist creative content creator social media influencer فنان محتوى إبداعي"},
    {"section": "R", "section_title": "Arts, Entertainment and Recreation",
     "division_code": "92", "division_title": "Gambling and betting activities",
     "group_code": "920", "group_title": "Gambling and betting activities",
     "class_code": "9200", "class_title": "Gambling and betting activities",
     "keywords": "gaming esports sports recreation نادي رياضي ترفيه"},
    {"section": "R", "section_title": "Arts, Entertainment and Recreation",
     "division_code": "93", "division_title": "Sports activities and amusement and recreation activities",
     "group_code": "931", "group_title": "Sports activities",
     "class_code": "9311", "class_title": "Operation of sports facilities",
     "keywords": "sports gym fitness club athlete coach trainer stadium رياضة ملعب صالة لياقة مدرب"},
    {"section": "R", "section_title": "Arts, Entertainment and Recreation",
     "division_code": "93", "division_title": "Sports activities and amusement and recreation activities",
     "group_code": "932", "group_title": "Amusement and recreation activities",
     "class_code": "9329", "class_title": "Other amusement and recreation activities",
     "keywords": "theme park amusement entertainment recreation leisure ترفيه ملاهي استجمام"},

    # ── S – Other Service Activities ──────────────────────────────────────────
    {"section": "S", "section_title": "Other Service Activities",
     "division_code": "94", "division_title": "Activities of membership organisations",
     "group_code": "941", "group_title": "Activities of business and employers membership organisations",
     "class_code": "9411", "class_title": "Activities of business and employers membership organisations",
     "keywords": "chamber commerce business association federation نقابة غرفة تجارة جمعية مهنية"},
    {"section": "S", "section_title": "Other Service Activities",
     "division_code": "94", "division_title": "Activities of membership organisations",
     "group_code": "949", "group_title": "Activities of other membership organisations",
     "class_code": "9491", "class_title": "Activities of religious organisations",
     "keywords": "mosque church religious organisation faith إمام مسجد دينية"},
    {"section": "S", "section_title": "Other Service Activities",
     "division_code": "95", "division_title": "Repair of computers and personal and household goods",
     "group_code": "951", "group_title": "Repair of computers and communication equipment",
     "class_code": "9511", "class_title": "Repair of computers and peripheral equipment",
     "keywords": "computer repair IT support technician laptop phone fix صيانة حاسوب تصليح"},
    {"section": "S", "section_title": "Other Service Activities",
     "division_code": "95", "division_title": "Repair of computers and personal and household goods",
     "group_code": "951", "group_title": "Repair of computers and communication equipment",
     "class_code": "9512", "class_title": "Repair of communication equipment",
     "keywords": "mobile phone repair smartphone electronics fix تصليح هاتف إصلاح إلكترونيات"},
    {"section": "S", "section_title": "Other Service Activities",
     "division_code": "96", "division_title": "Other personal service activities",
     "group_code": "960", "group_title": "Other personal service activities",
     "class_code": "9601", "class_title": "Washing and dry-cleaning of textile and fur products",
     "keywords": "laundry dry cleaning washing textile مغسلة غسيل ملابس"},
    {"section": "S", "section_title": "Other Service Activities",
     "division_code": "96", "division_title": "Other personal service activities",
     "group_code": "960", "group_title": "Other personal service activities",
     "class_code": "9602", "class_title": "Hairdressing and other beauty treatment",
     "keywords": "beauty salon barber hairdresser spa grooming nail تجميل حلاقة صالون تجميل"},
    {"section": "S", "section_title": "Other Service Activities",
     "division_code": "96", "division_title": "Other personal service activities",
     "group_code": "960", "group_title": "Other personal service activities",
     "class_code": "9609", "class_title": "Other personal service activities n.e.c.",
     "keywords": "personal service tailor alterations pet grooming خدمات شخصية خياط"},

    # ── T – Households as Employers ───────────────────────────────────────────
    {"section": "T", "section_title": "Activities of Households as Employers of Domestic Personnel",
     "division_code": "97", "division_title": "Activities of households as employers of domestic personnel",
     "group_code": "970", "group_title": "Activities of households as employers of domestic personnel",
     "class_code": "9700", "class_title": "Activities of households as employers of domestic personnel",
     "keywords": "domestic worker maid housekeeper nanny chauffeur driver gardener cook home خادم عامل منزلي سائق طباخ"},

    # ── U – Extraterritorial ──────────────────────────────────────────────────
    {"section": "U", "section_title": "Activities of Extraterritorial Organisations and Bodies",
     "division_code": "99", "division_title": "Activities of extraterritorial organisations and bodies",
     "group_code": "990", "group_title": "Activities of extraterritorial organisations and bodies",
     "class_code": "9900", "class_title": "Activities of extraterritorial organisations and bodies",
     "keywords": "UN UNESCO WHO embassy consulate NGO international organisation diplomat أمم متحدة سفارة دبلوماسي منظمة دولية"},
]

# Build fast-lookup index: tokens → list of class entries
_TOKEN_INDEX: dict[str, list[dict]] = {}
for _entry in _ISIC_DATA:
    for _tok in re.findall(r"[a-z\u0600-\u06ff]{3,}", _entry["keywords"].lower()):
        _TOKEN_INDEX.setdefault(_tok, []).append(_entry)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ISICClassification:
    section:         str
    section_title:   str
    division_code:   str
    division_title:  str
    group_code:      str
    group_title:     str
    class_code:      str
    class_title:     str
    confidence:      float
    method:          str                    # "keyword" | "llm"
    alternatives:    list[dict] = field(default_factory=list)
    raw_text:        Optional[str] = None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ISICClassifier:
    """Classify free-text industry descriptions to ISIC Rev.4 4-digit classes."""

    _KEYWORD_THRESHOLD = 0.85   # skip LLM when keyword score ≥ this
    _TOP_K             = 3

    def __init__(self) -> None:
        self._llm = get_llm(TaskType.GENERAL)

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str, *, method: Optional[str] = None) -> ISICClassification:
        """
        Classify *text* to an ISIC Rev.4 4-digit class with full hierarchy.

        Returns ``ISICClassification`` with section → division → group → class.

        Parameters
        ----------
        method : str, optional
            Default ``None`` runs today's unchanged keyword+LLM pipeline
            (identical to calling ``classify(text)`` before this parameter
            existed). Passing a value from
            ``backend.agents.classifier_methods.NOT_IMPLEMENTED_METHODS``
            (currently just ``"isic_hierarchical_retrieval"``) returns a
            structured not-implemented result instead of running any
            classification -- hierarchical retrieval for ISIC is deferred
            scope, not yet built. See
            Documentation/Conference_I_Reviewer_2/CLASSIFIER_METHOD_REGISTRY.md.
        """
        if method is not None and method in NOT_IMPLEMENTED_METHODS:
            return self._not_implemented(method, text)

        text = (text or "").strip()
        if not text:
            return self._fallback(text)

        scored = self._keyword_score(text)
        if not scored:
            return self._fallback(text)

        best_score, best_entry = scored[0]
        if best_score >= self._KEYWORD_THRESHOLD:
            return self._make_result(best_entry, best_score, "keyword", scored, text)

        # LLM re-ranking
        top_candidates = [e for _, e in scored[:self._TOP_K]]
        llm_result = self._llm_rerank(text, top_candidates)
        if llm_result:
            return llm_result

        # LLM failed — fall back to best keyword match with deflated confidence
        return self._make_result(best_entry, best_score * 0.8, "keyword", [], text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_result(
        self,
        entry: dict,
        score: float,
        method: str,
        scored: list[tuple[float, dict]],
        text: str,
    ) -> ISICClassification:
        return ISICClassification(
            section=entry["section"],
            section_title=entry["section_title"],
            division_code=entry["division_code"],
            division_title=entry["division_title"],
            group_code=entry["group_code"],
            group_title=entry["group_title"],
            class_code=entry["class_code"],
            class_title=entry["class_title"],
            confidence=round(min(score, 1.0), 4),
            method=method,
            alternatives=[
                {
                    "class_code": e["class_code"],
                    "class_title": e["class_title"],
                    "division_code": e["division_code"],
                    "section": e["section"],
                    "confidence": round(s, 4),
                }
                for s, e in scored[1:self._TOP_K]
            ],
            raw_text=text,
        )

    # ── Keyword scoring ───────────────────────────────────────────────────────

    def _keyword_score(self, text: str) -> list[tuple[float, dict]]:
        """Return (score, entry) pairs sorted descending by score."""
        tokens = set(re.findall(r"[a-z\u0600-\u06ff]{3,}", text.lower()))
        if not tokens:
            return []

        hit_counts: dict[int, int] = {}
        hit_entries: dict[int, dict] = {}

        for tok in tokens:
            for entry in _TOKEN_INDEX.get(tok, []):
                eid = id(entry)
                hit_counts[eid] = hit_counts.get(eid, 0) + 1
                hit_entries[eid] = entry

        if not hit_counts:
            return []

        max_hits = max(hit_counts.values())
        scored = [
            (count / max_hits, hit_entries[eid])
            for eid, count in hit_counts.items()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ── LLM re-ranking ────────────────────────────────────────────────────────

    def _llm_rerank(
        self, text: str, candidates: list[dict]
    ) -> Optional[ISICClassification]:
        candidates_text = "\n".join(
            f"Class {c['class_code']}: {c['class_title']} "
            f"[Division {c['division_code']}: {c['division_title']}, "
            f"Section {c['section']}: {c['section_title']}]"
            for c in candidates
        )
        task_desc = (
            f"A survey respondent described their industry as:\n\"{text}\"\n\n"
            f"Choose the single most appropriate ISIC Rev.4 class (4-digit) from:\n"
            f"{candidates_text}\n\n"
            "Respond with JSON only (no markdown):\n"
            '{"class_code": "XXXX", "confidence": 0.0-1.0, "reasoning": "..."}'
        )

        try:
            agent = Agent(
                role="ISIC Industry Classifier",
                goal="Select the best ISIC Rev.4 4-digit class for the industry description.",
                backstory="You are an ILO statistician expert in ISIC Rev.4 industrial classification.",
                llm=self._llm,
                verbose=False,
                allow_delegation=False,
            )
            task = Task(
                description=task_desc,
                expected_output='JSON: {"class_code": "XXXX", "confidence": 0.0, "reasoning": "..."}',
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            raw = str(crew.kickoff()).strip()
        except Exception as exc:
            log.warning("ISIC LLM re-ranking failed: %s", exc)
            return None

        return self._parse_llm_response(raw, candidates, text)

    def _parse_llm_response(
        self, raw: str, candidates: list[dict], original_text: str
    ) -> Optional[ISICClassification]:
        # Strip markdown fences
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

        code = str(data.get("class_code", "")).strip()
        conf = float(data.get("confidence", 0.7))

        # Resolve to entry — match by class_code first, then division_code as fallback
        entry = next((c for c in candidates if c["class_code"] == code), None)
        if entry is None:
            # Try matching by division (LLM may have returned 2-digit code)
            entry = next((c for c in candidates if c["division_code"] == code), None)
        if entry is None:
            entry = candidates[0]
            conf *= 0.7

        return ISICClassification(
            section=entry["section"],
            section_title=entry["section_title"],
            division_code=entry["division_code"],
            division_title=entry["division_title"],
            group_code=entry["group_code"],
            group_title=entry["group_title"],
            class_code=entry["class_code"],
            class_title=entry["class_title"],
            confidence=round(min(conf, 1.0), 4),
            method="llm",
            raw_text=original_text,
        )

    # ── Not-implemented stub (deferred scope, see classifier_methods.py) ───────

    @staticmethod
    def _not_implemented(method: str, text: str) -> ISICClassification:
        """
        Structured "not implemented" result for a ``method`` value in
        ``NOT_IMPLEMENTED_METHODS`` -- returned instead of raising so
        CLI/eval callers that don't expect an exception stay safe, and
        instead of silently running the default keyword/LLM pipeline (which
        would misreport which method actually produced the result). All
        hierarchy fields are deliberately empty rather than a fabricated or
        borrowed classification.
        """
        return ISICClassification(
            section="", section_title="", division_code="", division_title="",
            group_code="", group_title="", class_code="", class_title="",
            confidence=0.0, method=method, alternatives=[],
            raw_text=f"{NOT_IMPLEMENTED_REASON} (requested method={method!r}, input={text!r})",
        )

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(text: str) -> ISICClassification:
        return ISICClassification(
            section="G",
            section_title="Wholesale and Retail Trade; Repair of Motor Vehicles",
            division_code="47",
            division_title="Retail trade, except of motor vehicles and motorcycles",
            group_code="471",
            group_title="Retail sale in non-specialised stores",
            class_code="4719",
            class_title="Other retail sale in non-specialised stores",
            confidence=0.0,
            method="keyword",
            raw_text=text,
        )
