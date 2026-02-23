"""
backend/rag/vector_store.py

Qdrant-backed vector store for ISCO-08 occupation classification.

Responsibilities
----------------
1. Connects to Qdrant (host/port read from .env).
2. Holds a curated ISCO-08 dataset (~110 entries: major groups, sub-major
   groups, and the most common unit groups, each with English + Arabic titles).
3. On first run, embeds the dataset with intfloat/multilingual-e5-large and
   upserts all vectors into a "isco_occupations" Qdrant collection.
4. Exposes `search(query, top_k)` → list[OccupationMatch] sorted by cosine
   similarity, with confidence scores in [0, 1].

Embedding notes
---------------
- Model  : intfloat/multilingual-e5-large  (1024-dim, multilingual)
- Prefix : "passage: " when indexing, "query: " when searching  (E5 convention)
- Vectors are L2-normalised so cosine similarity == dot product.

Usage
-----
from backend.rag.vector_store import get_vector_store

vs = get_vector_store()                          # lazy singleton
matches = vs.search("software engineer")
matches = vs.search("مهندس برمجيات")             # Arabic
matches = vs.search("software engineer بالرياض") # code-switched
for m in matches:
    print(m.code, m.title_en, m.confidence)
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "isco_occupations"
MODEL_NAME      = "intfloat/multilingual-e5-large"
VECTOR_DIM      = 1024
TOP_K_DEFAULT   = 5
_BATCH_SIZE     = 64   # embedding batch size


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class OccupationMatch(BaseModel):
    """A single ISCO-08 occupation returned by semantic search."""

    code: str
    title_en: str
    title_ar: str
    level: int                          # 1 = major group … 4 = unit group
    description: str
    confidence: float = Field(ge=0.0, le=1.0)  # cosine similarity [0, 1]


# ---------------------------------------------------------------------------
# ISCO-08 dataset
# ---------------------------------------------------------------------------

_ISCO_DATA: list[dict] = [

    # ── Major Groups (level 1) ───────────────────────────────────────────────
    {"code": "1", "level": 1,
     "title_en": "Managers",
     "title_ar": "المديرون",
     "description": "Plan, direct, coordinate and evaluate activities of enterprises, governments and organisations"},
    {"code": "2", "level": 1,
     "title_en": "Professionals",
     "title_ar": "المهنيون",
     "description": "Increase existing knowledge, apply scientific and artistic concepts and theories to solve problems"},
    {"code": "3", "level": 1,
     "title_en": "Technicians and Associate Professionals",
     "title_ar": "الفنيون والمهنيون المساعدون",
     "description": "Perform technical and related tasks connected with research and the application of scientific concepts"},
    {"code": "4", "level": 1,
     "title_en": "Clerical Support Workers",
     "title_ar": "موظفو الدعم الكتابي",
     "description": "Record, organise, store, compute and retrieve information and perform a range of clerical duties"},
    {"code": "5", "level": 1,
     "title_en": "Service and Sales Workers",
     "title_ar": "عمال الخدمات والمبيعات",
     "description": "Perform personal and protective services and sell goods in shops and markets"},
    {"code": "6", "level": 1,
     "title_en": "Skilled Agricultural, Forestry and Fishery Workers",
     "title_ar": "العمال المهرة في الزراعة والغابات وصيد الأسماك",
     "description": "Grow and harvest field crops, tend and hunt animals, catch fish and gather natural resources"},
    {"code": "7", "level": 1,
     "title_en": "Craft and Related Trades Workers",
     "title_ar": "عمال الحرف والمهن ذات الصلة",
     "description": "Apply specific technical knowledge and skills to construct and maintain buildings and related structures"},
    {"code": "8", "level": 1,
     "title_en": "Plant and Machine Operators and Assemblers",
     "title_ar": "مشغلو المصانع والآلات والمجمّعون",
     "description": "Operate and monitor industrial machinery, equipment, and assemble products"},
    {"code": "9", "level": 1,
     "title_en": "Elementary Occupations",
     "title_ar": "المهن الأولية",
     "description": "Perform simple and routine tasks requiring mainly the use of hand-held tools and physical effort"},
    {"code": "0", "level": 1,
     "title_en": "Armed Forces Occupations",
     "title_ar": "مهن القوات المسلحة",
     "description": "Military, naval and air force occupations requiring the use of arms and military training"},

    # ── Sub-Major Groups (level 2) ───────────────────────────────────────────
    {"code": "11", "level": 2,
     "title_en": "Chief Executives, Senior Officials and Legislators",
     "title_ar": "الرؤساء التنفيذيون وكبار المسؤولين والمشرّعون",
     "description": "Determine and formulate policies and plan, direct and coordinate activities of enterprises and governments"},
    {"code": "12", "level": 2,
     "title_en": "Administrative and Commercial Managers",
     "title_ar": "المديرون الإداريون والتجاريون",
     "description": "Plan, direct and coordinate administrative and commercial activities of enterprises"},
    {"code": "13", "level": 2,
     "title_en": "Production and Specialised Services Managers",
     "title_ar": "مديرو الإنتاج والخدمات المتخصصة",
     "description": "Plan, direct and coordinate production, mining, construction and distribution activities"},
    {"code": "14", "level": 2,
     "title_en": "Hospitality, Retail and Other Services Managers",
     "title_ar": "مديرو الضيافة والتجزئة والخدمات الأخرى",
     "description": "Plan, organise and direct operations of hotels, restaurants, shops and other services"},
    {"code": "21", "level": 2,
     "title_en": "Science and Engineering Professionals",
     "title_ar": "مهنيو العلوم والهندسة",
     "description": "Research, develop and apply scientific knowledge in engineering, mathematics and natural sciences"},
    {"code": "22", "level": 2,
     "title_en": "Health Professionals",
     "title_ar": "المهنيون الصحيون",
     "description": "Conduct research, improve or develop concepts and operational methods of medicine and surgery"},
    {"code": "23", "level": 2,
     "title_en": "Teaching Professionals",
     "title_ar": "مهنيو التدريس",
     "description": "Provide instruction and education to students from early childhood through to higher education"},
    {"code": "24", "level": 2,
     "title_en": "Business and Administration Professionals",
     "title_ar": "مهنيو الأعمال والإدارة",
     "description": "Develop and apply business theories and operational methods to achieve organisational goals"},
    {"code": "25", "level": 2,
     "title_en": "Information and Communications Technology Professionals",
     "title_ar": "مهنيو تكنولوجيا المعلومات والاتصالات",
     "description": "Research, design, develop and maintain IT systems, software and communications infrastructure"},
    {"code": "26", "level": 2,
     "title_en": "Legal, Social and Cultural Professionals",
     "title_ar": "المهنيون القانونيون والاجتماعيون والثقافيون",
     "description": "Apply legal, social science and cultural concepts to interpret and implement laws and provide services"},
    {"code": "31", "level": 2,
     "title_en": "Science and Engineering Associate Professionals",
     "title_ar": "المهنيون المساعدون في العلوم والهندسة",
     "description": "Perform technical tasks connected with research and the application of engineering concepts"},
    {"code": "32", "level": 2,
     "title_en": "Health Associate Professionals",
     "title_ar": "المهنيون المساعدون في الصحة",
     "description": "Carry out diagnostic, preventive and curative health work under the guidance of physicians"},
    {"code": "33", "level": 2,
     "title_en": "Business and Administration Associate Professionals",
     "title_ar": "المهنيون المساعدون في الأعمال والإدارة",
     "description": "Perform technical and operational tasks in business administration and financial services"},
    {"code": "34", "level": 2,
     "title_en": "Legal, Social, Cultural and Related Associate Professionals",
     "title_ar": "المهنيون المساعدون في المجالات القانونية والاجتماعية والثقافية",
     "description": "Assist legal, social and cultural professionals and perform related tasks"},
    {"code": "35", "level": 2,
     "title_en": "Information and Communications Technicians",
     "title_ar": "فنيو المعلومات والاتصالات",
     "description": "Install, operate and maintain information and communications technology equipment and systems"},
    {"code": "41", "level": 2,
     "title_en": "General and Keyboard Clerks",
     "title_ar": "الكتبة العامون وكتبة لوحة المفاتيح",
     "description": "Perform general clerical work and keyboard data entry in offices"},
    {"code": "42", "level": 2,
     "title_en": "Customer Services Clerks",
     "title_ar": "كتبة خدمة العملاء",
     "description": "Deal directly with customers and provide information and assistance with services"},
    {"code": "43", "level": 2,
     "title_en": "Numerical and Material Recording Clerks",
     "title_ar": "كتبة التسجيل العددي والمادي",
     "description": "Compute, classify and record numerical data and handle materials in commercial settings"},
    {"code": "44", "level": 2,
     "title_en": "Other Clerical Support Workers",
     "title_ar": "موظفو الدعم الكتابي الآخرون",
     "description": "Perform miscellaneous clerical and administrative support tasks"},
    {"code": "51", "level": 2,
     "title_en": "Personal Service Workers",
     "title_ar": "عمال الخدمة الشخصية",
     "description": "Provide personal and household services including travel, food preparation and personal care"},
    {"code": "52", "level": 2,
     "title_en": "Sales Workers",
     "title_ar": "عمال المبيعات",
     "description": "Sell merchandise and goods in retail establishments, markets and door-to-door"},
    {"code": "53", "level": 2,
     "title_en": "Personal Care Workers",
     "title_ar": "عمال الرعاية الشخصية",
     "description": "Provide personal care, assistance and support to children, the elderly and persons with disabilities"},
    {"code": "54", "level": 2,
     "title_en": "Protective Services Workers",
     "title_ar": "عمال الخدمات الحمائية",
     "description": "Guard persons and property against crime, fire, accidents and other hazards"},
    {"code": "61", "level": 2,
     "title_en": "Market-Oriented Skilled Agricultural Workers",
     "title_ar": "العمال الزراعيون المهرة الموجهون للسوق",
     "description": "Grow crops and raise animals primarily for market and commercial purposes"},
    {"code": "71", "level": 2,
     "title_en": "Building and Related Trades Workers",
     "title_ar": "عمال البناء والمهن ذات الصلة",
     "description": "Construct, maintain and repair buildings, structures and other civil engineering works"},
    {"code": "72", "level": 2,
     "title_en": "Metal, Machinery and Related Trades Workers",
     "title_ar": "عمال المعادن والآلات والمهن ذات الصلة",
     "description": "Shape, treat and join metals and construct metal structures and machinery"},
    {"code": "74", "level": 2,
     "title_en": "Electrical and Electronic Trades Workers",
     "title_ar": "عمال المهن الكهربائية والإلكترونية",
     "description": "Install, maintain and repair electrical systems, electronic equipment and ICT infrastructure"},
    {"code": "75", "level": 2,
     "title_en": "Food Processing, Woodworking and Other Craft Workers",
     "title_ar": "عمال معالجة الأغذية والنجارة والحرف الأخرى",
     "description": "Process food and produce goods from wood, textiles, leather and other craft materials"},
    {"code": "81", "level": 2,
     "title_en": "Stationary Plant and Machine Operators",
     "title_ar": "مشغلو المصانع والآلات الثابتة",
     "description": "Operate and monitor stationary plant and machinery in industrial and processing environments"},
    {"code": "83", "level": 2,
     "title_en": "Drivers and Mobile Plant Operators",
     "title_ar": "السائقون ومشغلو الآلات المتنقلة",
     "description": "Drive vehicles and operate mobile machinery to transport people and goods"},
    {"code": "91", "level": 2,
     "title_en": "Cleaners and Helpers",
     "title_ar": "عمال النظافة والمساعدون",
     "description": "Clean the interior of buildings and perform related domestic helper tasks"},
    {"code": "93", "level": 2,
     "title_en": "Labourers in Mining, Construction, Manufacturing and Transport",
     "title_ar": "العمال في التعدين والبناء والتصنيع والنقل",
     "description": "Perform simple and routine physical tasks in mining, construction, manufacturing and transport"},

    # ── Unit Groups (level 4) ────────────────────────────────────────────────

    # Managers
    {"code": "1120", "level": 4,
     "title_en": "Managing Directors and Chief Executives",
     "title_ar": "المديرون المنفذون والرؤساء التنفيذيون",
     "description": "Determine and formulate the policies and strategic goals of enterprises and organisations"},
    {"code": "1211", "level": 4,
     "title_en": "Finance Managers",
     "title_ar": "مديرو المالية",
     "description": "Plan, direct and coordinate financial operations, budgeting and accounting activities"},
    {"code": "1212", "level": 4,
     "title_en": "Human Resource Managers",
     "title_ar": "مديرو الموارد البشرية",
     "description": "Plan, direct and coordinate human resource management, recruitment and staff relations"},
    {"code": "1221", "level": 4,
     "title_en": "Sales and Marketing Managers",
     "title_ar": "مديرو المبيعات والتسويق",
     "description": "Plan, direct and coordinate sales and marketing policies, programmes and campaigns"},
    {"code": "1223", "level": 4,
     "title_en": "Research and Development Managers",
     "title_ar": "مديرو البحث والتطوير",
     "description": "Plan, direct and coordinate research and development activities and programmes"},
    {"code": "1322", "level": 4,
     "title_en": "Construction Managers",
     "title_ar": "مديرو البناء",
     "description": "Plan, direct and coordinate construction and civil engineering activities on projects"},
    {"code": "1324", "level": 4,
     "title_en": "Supply, Distribution and Related Managers",
     "title_ar": "مديرو التوريد والتوزيع",
     "description": "Plan and direct activities for procuring, distributing and storing goods and supplies"},
    {"code": "1431", "level": 4,
     "title_en": "Restaurant Managers",
     "title_ar": "مديرو المطاعم",
     "description": "Plan, organise and direct operations of restaurants and food service establishments"},

    # Science and Engineering Professionals
    {"code": "2141", "level": 4,
     "title_en": "Industrial and Production Engineers",
     "title_ar": "مهندسو الصناعة والإنتاج",
     "description": "Design and develop efficient systems for producing goods and services"},
    {"code": "2142", "level": 4,
     "title_en": "Civil Engineers",
     "title_ar": "المهندسون المدنيون",
     "description": "Design, develop and supervise construction of roads, bridges, tunnels, dams and buildings"},
    {"code": "2143", "level": 4,
     "title_en": "Environmental Engineers",
     "title_ar": "مهندسو البيئة",
     "description": "Design engineering solutions to environmental problems including waste management and pollution control"},
    {"code": "2144", "level": 4,
     "title_en": "Mechanical Engineers",
     "title_ar": "المهندسون الميكانيكيون",
     "description": "Design, develop and oversee production and operation of mechanical systems and machinery"},
    {"code": "2145", "level": 4,
     "title_en": "Chemical and Petroleum Engineers",
     "title_ar": "مهندسو الكيمياء والبترول",
     "description": "Design and develop processes for large-scale manufacture of chemical and petroleum products"},
    {"code": "2151", "level": 4,
     "title_en": "Electrical Engineers",
     "title_ar": "المهندسون الكهربائيون",
     "description": "Design, develop and oversee production and installation of electrical equipment and power systems"},
    {"code": "2152", "level": 4,
     "title_en": "Electronics Engineers",
     "title_ar": "مهندسو الإلكترونيات",
     "description": "Design and develop electronic devices, circuits and systems"},
    {"code": "2153", "level": 4,
     "title_en": "Telecommunications Engineers",
     "title_ar": "مهندسو الاتصالات",
     "description": "Design, develop and maintain telecommunications systems and networks"},
    {"code": "2161", "level": 4,
     "title_en": "Building Architects",
     "title_ar": "مهندسو العمارة",
     "description": "Design buildings and advise on construction methods, materials and energy efficiency"},
    {"code": "2166", "level": 4,
     "title_en": "Graphic and Multimedia Designers",
     "title_ar": "مصممو الجرافيك والوسائط المتعددة",
     "description": "Create visual concepts and design graphic elements for print and digital media"},

    # Health Professionals
    {"code": "2211", "level": 4,
     "title_en": "Generalist Medical Practitioners",
     "title_ar": "الأطباء الممارسون العامون",
     "description": "Diagnose, treat and prevent illness, disease, injury and other health conditions"},
    {"code": "2212", "level": 4,
     "title_en": "Specialist Medical Practitioners",
     "title_ar": "الأطباء المتخصصون",
     "description": "Provide specialised medical care, diagnosis and treatment in a specific field of medicine"},
    {"code": "2221", "level": 4,
     "title_en": "Nursing Professionals",
     "title_ar": "المهنيون في التمريض",
     "description": "Provide nursing care, treatment and health education to patients in hospitals and community settings"},
    {"code": "2261", "level": 4,
     "title_en": "Dentists",
     "title_ar": "أطباء الأسنان",
     "description": "Diagnose, treat and prevent diseases and conditions affecting the teeth, mouth and jaw"},
    {"code": "2262", "level": 4,
     "title_en": "Pharmacists",
     "title_ar": "الصيادلة",
     "description": "Compound and dispense drugs and medicines and advise patients and healthcare staff on their use"},

    # Teaching Professionals
    {"code": "2310", "level": 4,
     "title_en": "University and Higher Education Teachers",
     "title_ar": "أساتذة الجامعات والتعليم العالي",
     "description": "Teach and conduct research at university and postgraduate level"},
    {"code": "2320", "level": 4,
     "title_en": "Vocational Education Teachers",
     "title_ar": "معلمو التعليم المهني",
     "description": "Teach practical and vocational subjects including trades and technical skills"},
    {"code": "2330", "level": 4,
     "title_en": "Secondary Education Teachers",
     "title_ar": "معلمو التعليم الثانوي",
     "description": "Teach academic subjects to students at secondary school level"},
    {"code": "2341", "level": 4,
     "title_en": "Primary School Teachers",
     "title_ar": "معلمو المدارس الابتدائية",
     "description": "Teach reading, writing, arithmetic and other subjects at primary school level"},

    # Business and Administration Professionals
    {"code": "2411", "level": 4,
     "title_en": "Accountants",
     "title_ar": "المحاسبون",
     "description": "Prepare and examine financial records, compute taxes and advise on financial matters"},
    {"code": "2412", "level": 4,
     "title_en": "Financial and Investment Advisers",
     "title_ar": "المستشارون الماليون والاستثماريون",
     "description": "Advise clients on financial planning, investment strategies and wealth management"},
    {"code": "2421", "level": 4,
     "title_en": "Management and Organisation Analysts",
     "title_ar": "محللو الإدارة والتنظيم",
     "description": "Analyse and recommend improvements to management systems and organisational structures"},
    {"code": "2431", "level": 4,
     "title_en": "Advertising and Marketing Professionals",
     "title_ar": "مهنيو الإعلان والتسويق",
     "description": "Plan and direct advertising and marketing programmes and brand communication campaigns"},
    {"code": "2433", "level": 4,
     "title_en": "Technical and Medical Sales Professionals",
     "title_ar": "مهنيو المبيعات التقنية والطبية",
     "description": "Sell technical and scientific products and services to businesses and medical institutions"},

    # ICT Professionals
    {"code": "2511", "level": 4,
     "title_en": "Systems Analysts",
     "title_ar": "محللو الأنظمة",
     "description": "Analyse user requirements, procedures and problems to design efficient IT solutions"},
    {"code": "2512", "level": 4,
     "title_en": "Software Developers",
     "title_ar": "مطورو البرمجيات",
     "description": "Design, develop, test and maintain software applications and systems"},
    {"code": "2513", "level": 4,
     "title_en": "Web and Multimedia Developers",
     "title_ar": "مطورو الويب والوسائط المتعددة",
     "description": "Design and develop websites, web applications and interactive multimedia content"},
    {"code": "2519", "level": 4,
     "title_en": "Software and Applications Developers NEC",
     "title_ar": "مطورو البرمجيات والتطبيقات غير المصنفين في مكان آخر",
     "description": "Develop, test and maintain software applications not classified elsewhere"},
    {"code": "2521", "level": 4,
     "title_en": "Database Designers and Administrators",
     "title_ar": "مصممو قواعد البيانات ومديروها",
     "description": "Design, implement and maintain database management systems and data storage infrastructure"},
    {"code": "2522", "level": 4,
     "title_en": "Systems Administrators",
     "title_ar": "مديرو الأنظمة",
     "description": "Install, configure and maintain servers, operating systems and computer infrastructure"},

    # Legal and Social Professionals
    {"code": "2611", "level": 4,
     "title_en": "Lawyers",
     "title_ar": "المحامون",
     "description": "Advise clients on legal matters, draft documents and represent them before courts"},
    {"code": "2635", "level": 4,
     "title_en": "Social Work and Counselling Professionals",
     "title_ar": "مهنيو العمل الاجتماعي والإرشاد",
     "description": "Assess needs of individuals and families and provide social support and counselling services"},

    # Technicians
    {"code": "3112", "level": 4,
     "title_en": "Civil Engineering Technicians",
     "title_ar": "فنيو الهندسة المدنية",
     "description": "Perform technical tasks in civil engineering, surveying and construction projects"},
    {"code": "3113", "level": 4,
     "title_en": "Electrical Engineering Technicians",
     "title_ar": "فنيو الهندسة الكهربائية",
     "description": "Perform technical tasks in electrical engineering and power systems"},
    {"code": "3114", "level": 4,
     "title_en": "Electronics Engineering Technicians",
     "title_ar": "فنيو الهندسة الإلكترونية",
     "description": "Assist in designing, testing and maintaining electronic equipment and systems"},
    {"code": "3131", "level": 4,
     "title_en": "Power Production Plant Operators",
     "title_ar": "مشغلو محطات توليد الطاقة",
     "description": "Operate and monitor power plant equipment and machinery to generate electricity"},
    {"code": "3221", "level": 4,
     "title_en": "Nursing Associate Professionals",
     "title_ar": "المهنيون المساعدون في التمريض",
     "description": "Provide basic nursing care and personal assistance under supervision of nursing professionals"},
    {"code": "3311", "level": 4,
     "title_en": "Securities and Finance Dealers and Brokers",
     "title_ar": "تجار الأوراق المالية والوسطاء الماليون",
     "description": "Buy and sell stocks, bonds and other financial instruments on behalf of clients"},
    {"code": "3322", "level": 4,
     "title_en": "Commercial Sales Representatives",
     "title_ar": "مندوبو المبيعات التجارية",
     "description": "Sell goods and services to businesses and consumers on behalf of companies"},
    {"code": "3433", "level": 4,
     "title_en": "Bookkeepers",
     "title_ar": "المحاسبون الدفتريون",
     "description": "Record financial transactions and maintain accounting books and records"},
    {"code": "3512", "level": 4,
     "title_en": "ICT Support Technicians",
     "title_ar": "فنيو دعم تكنولوجيا المعلومات والاتصالات",
     "description": "Provide technical support and troubleshooting assistance for computer hardware and software"},

    # Clerical
    {"code": "4110", "level": 4,
     "title_en": "General Office Clerks",
     "title_ar": "موظفو المكاتب العامون",
     "description": "Perform general clerical duties including filing, photocopying and scheduling"},
    {"code": "4120", "level": 4,
     "title_en": "Secretaries (General)",
     "title_ar": "السكرتيرون العامون",
     "description": "Perform secretarial, administrative and organisational support duties"},
    {"code": "4222", "level": 4,
     "title_en": "Receptionists and Information Clerks",
     "title_ar": "موظفو الاستقبال وخدمة المعلومات",
     "description": "Receive visitors, handle enquiries and provide information in offices, hotels and public buildings"},
    {"code": "4311", "level": 4,
     "title_en": "Accounting and Bookkeeping Clerks",
     "title_ar": "كتبة المحاسبة ومسك الدفاتر",
     "description": "Perform routine clerical tasks related to accounting and financial record keeping"},
    {"code": "4321", "level": 4,
     "title_en": "Stock Clerks and Storekeepers",
     "title_ar": "كتبة المخزون وأمناء المستودعات",
     "description": "Receive, store, issue and record goods and materials in warehouses and stores"},

    # Service and Sales
    {"code": "5120", "level": 4,
     "title_en": "Cooks",
     "title_ar": "الطهاة",
     "description": "Prepare, season and cook food in restaurants, hotels and other food establishments"},
    {"code": "5131", "level": 4,
     "title_en": "Waiters",
     "title_ar": "النادلون",
     "description": "Serve food and beverages to customers in restaurants, hotels and catering establishments"},
    {"code": "5141", "level": 4,
     "title_en": "Hairdressers",
     "title_ar": "مصففو الشعر والحلاقون",
     "description": "Cut, colour, perm and style hair and provide beauty treatments for clients"},
    {"code": "5221", "level": 4,
     "title_en": "Shop Salespersons",
     "title_ar": "البائعون في المحلات التجارية",
     "description": "Sell goods and assist customers in retail shops and department stores"},
    {"code": "5311", "level": 4,
     "title_en": "Child Care Workers",
     "title_ar": "عمال رعاية الأطفال",
     "description": "Provide care, supervision and educational activities for children in nurseries and daycare"},
    {"code": "5412", "level": 4,
     "title_en": "Police Officers",
     "title_ar": "ضباط الشرطة",
     "description": "Maintain law and order, prevent and detect crime and enforce laws and regulations"},
    {"code": "5414", "level": 4,
     "title_en": "Security Guards and Related Workers",
     "title_ar": "حراس الأمن والعمال ذوو الصلة",
     "description": "Guard and protect property, premises and individuals against theft and other hazards"},

    # Craft and Trades
    {"code": "7111", "level": 4,
     "title_en": "House Builders",
     "title_ar": "عمال بناء المنازل",
     "description": "Construct and repair houses and small residential buildings using traditional methods"},
    {"code": "7115", "level": 4,
     "title_en": "Carpenters and Joiners",
     "title_ar": "النجارون",
     "description": "Cut, shape and fit timber and other materials to construct and repair buildings and furniture"},
    {"code": "7131", "level": 4,
     "title_en": "Painters and Related Workers",
     "title_ar": "الدهانون والعمال ذوو الصلة",
     "description": "Apply paint, varnish, wallpaper and other finishes to buildings and structures"},
    {"code": "7411", "level": 4,
     "title_en": "Building Electricians",
     "title_ar": "الكهربائيون في المباني",
     "description": "Install, test and maintain electrical wiring systems and equipment in buildings"},
    {"code": "7422", "level": 4,
     "title_en": "ICT Equipment Installers and Servicers",
     "title_ar": "فنيو تركيب وصيانة معدات تكنولوجيا المعلومات والاتصالات",
     "description": "Install, maintain and repair ICT equipment including computers and network infrastructure"},
    {"code": "7512", "level": 4,
     "title_en": "Bakers, Pastry-Cooks and Confectionery Makers",
     "title_ar": "الخبازون وصانعو المعجنات والحلوى",
     "description": "Prepare and bake bread, cakes, pastries and other bakery and confectionery products"},
    {"code": "7531", "level": 4,
     "title_en": "Tailors, Dressmakers and Hatters",
     "title_ar": "الخياطون ومصممو الأزياء",
     "description": "Design, make, alter and repair garments and accessories"},

    # Operators and Drivers
    {"code": "8322", "level": 4,
     "title_en": "Car, Taxi and Van Drivers",
     "title_ar": "سائقو السيارات والتاكسي والشاحنات الصغيرة",
     "description": "Drive cars, taxis and light vans to transport passengers and light goods"},
    {"code": "8331", "level": 4,
     "title_en": "Bus and Tram Drivers",
     "title_ar": "سائقو الحافلات والترام",
     "description": "Drive buses and trams on scheduled routes to transport passengers"},
    {"code": "8332", "level": 4,
     "title_en": "Heavy Truck and Lorry Drivers",
     "title_ar": "سائقو الشاحنات الثقيلة",
     "description": "Drive heavy trucks and lorries to collect and deliver goods and materials"},
    {"code": "8343", "level": 4,
     "title_en": "Crane, Hoist and Related Plant Operators",
     "title_ar": "مشغلو الرافعات والرافعات الشحنية",
     "description": "Operate cranes, hoists and similar lifting equipment on construction sites and warehouses"},

    # Elementary
    {"code": "9111", "level": 4,
     "title_en": "Domestic Cleaners and Helpers",
     "title_ar": "عمال التنظيف والمساعدة في المنازل",
     "description": "Clean houses, wash dishes, do laundry and perform other domestic tasks in private households"},
    {"code": "9112", "level": 4,
     "title_en": "Cleaners and Helpers in Offices, Hotels and Similar Establishments",
     "title_ar": "عمال التنظيف والمساعدة في المكاتب والفنادق",
     "description": "Sweep, mop, scrub and polish floors and clean offices, hotels and other public buildings"},
    {"code": "9312", "level": 4,
     "title_en": "Civil Engineering Labourers",
     "title_ar": "العمال في الهندسة المدنية",
     "description": "Perform simple labouring tasks on roads, railways, tunnels and other civil engineering projects"},
    {"code": "9333", "level": 4,
     "title_en": "Freight Handlers",
     "title_ar": "عمال الشحن",
     "description": "Load and unload freight from vehicles, vessels and aircraft at docks and terminals"},
    {"code": "9411", "level": 4,
     "title_en": "Fast-Food Preparers",
     "title_ar": "معدّو الوجبات السريعة",
     "description": "Prepare and serve fast food in canteens, cafeterias and fast-food restaurants"},
    {"code": "9412", "level": 4,
     "title_en": "Kitchen Helpers",
     "title_ar": "مساعدو المطبخ",
     "description": "Perform routine tasks to assist cooks including washing, peeling and basic food preparation"},
    {"code": "9613", "level": 4,
     "title_en": "Garbage and Recycling Collectors",
     "title_ar": "عمال جمع القمامة وإعادة التدوير",
     "description": "Collect and transport garbage, recyclable materials and other refuse"},

    # Armed Forces
    {"code": "0110", "level": 4,
     "title_en": "Commissioned Armed Forces Officers",
     "title_ar": "ضباط القوات المسلحة المكلّفون",
     "description": "Command and manage military operations, strategy and personnel"},
    {"code": "0210", "level": 4,
     "title_en": "Non-Commissioned Armed Forces Officers",
     "title_ar": "ضباط الصف في القوات المسلحة",
     "description": "Lead and supervise enlisted military personnel in operational and administrative roles"},
    {"code": "0310", "level": 4,
     "title_en": "Armed Forces Other Ranks",
     "title_ar": "رتب القوات المسلحة الأخرى",
     "description": "Carry out military duties as enlisted soldiers, sailors, marines or airmen"},
]


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Manages ISCO-08 occupation embeddings in Qdrant.

    First call to __init__ (or get_vector_store()) will:
      1. Load the multilingual-e5-large model (~560 MB download on first use).
      2. Connect to Qdrant.
      3. Create the collection if it does not exist.
      4. Embed and upsert all ISCO entries if the collection is empty.

    Subsequent instantiations skip steps 3–4 if the collection is already
    populated (checked via point count).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        recreate: bool = False,
    ) -> None:
        _host = host or os.getenv("QDRANT_HOST", "localhost")
        _port = int(port or os.getenv("QDRANT_PORT", 6333))

        self._client = QdrantClient(host=_host, port=_port)
        self._model  = SentenceTransformer(MODEL_NAME)

        self._ensure_collection(recreate=recreate)
        self._ensure_populated()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[OccupationMatch]:
        """
        Find the top-k ISCO occupations semantically closest to *query*.

        Works for English, Arabic, and code-switched input.

        Parameters
        ----------
        query : str
            Free-text job title, description, or any occupation-related text.
        top_k : int
            Number of results to return (default 5).

        Returns
        -------
        list[OccupationMatch]
            Sorted by confidence descending.  Each item includes the ISCO code,
            English and Arabic titles, hierarchy level, description, and a
            confidence score ∈ [0, 1].
        """
        if not query.strip():
            return []

        query_vec = self._embed([query], is_query=True)[0]

        hits = self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )

        results: list[OccupationMatch] = []
        for hit in hits:
            p = hit.payload
            confidence = round(max(0.0, min(1.0, float(hit.score))), 4)
            results.append(OccupationMatch(
                code=p["code"],
                title_en=p["title_en"],
                title_ar=p["title_ar"],
                level=p["level"],
                description=p["description"],
                confidence=confidence,
            ))

        return results

    def rebuild_index(self) -> None:
        """Drop and fully rebuild the Qdrant collection from the ISCO dataset."""
        self._ensure_collection(recreate=True)
        self._ensure_populated()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """
        Encode *texts* with multilingual-e5-large.

        E5 models require task-specific prefixes:
          - "passage: "  when indexing documents
          - "query: "    when encoding search queries
        Embeddings are L2-normalised so cosine sim == dot product.
        """
        prefix   = "query: " if is_query else "passage: "
        prefixed = [f"{prefix}{t.strip()}" for t in texts]
        vectors  = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=_BATCH_SIZE,
        )
        return vectors.tolist()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self, recreate: bool = False) -> None:
        existing = {c.name for c in self._client.get_collections().collections}

        if COLLECTION_NAME in existing:
            if recreate:
                self._client.delete_collection(COLLECTION_NAME)
            else:
                return

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    def _ensure_populated(self) -> None:
        """Upsert all ISCO entries if the collection is empty."""
        count = self._client.count(
            collection_name=COLLECTION_NAME, exact=True
        ).count
        if count > 0:
            return

        # Build one embedding text per entry (EN title + AR title + description)
        texts = [
            f"{d['title_en']} {d['title_ar']} {d['description']}"
            for d in _ISCO_DATA
        ]
        vectors = self._embed(texts, is_query=False)

        points = [
            PointStruct(
                id=_stable_id(d["code"]),
                vector=vec,
                payload={
                    "code":        d["code"],
                    "title_en":    d["title_en"],
                    "title_ar":    d["title_ar"],
                    "level":       d["level"],
                    "description": d["description"],
                },
            )
            for d, vec in zip(_ISCO_DATA, vectors)
        ]

        for i in range(0, len(points), _BATCH_SIZE):
            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + _BATCH_SIZE],
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_id(code: str) -> int:
    """
    Convert an ISCO code string to a stable, collision-resistant integer ID.

    Uses the first 15 hex digits of the SHA-256 hash, giving a 60-bit
    non-negative integer that fits comfortably in Qdrant's uint64 ID space.
    """
    return int(hashlib.sha256(code.encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[VectorStore] = None


def get_vector_store(recreate: bool = False) -> VectorStore:
    """
    Return the module-level VectorStore singleton.

    The first call creates the instance (loads the model, connects to Qdrant,
    populates the index if needed).  Subsequent calls return the cached
    instance unless *recreate=True* is passed to force a full rebuild.
    """
    global _instance
    if _instance is None or recreate:
        _instance = VectorStore(recreate=recreate)
    return _instance
