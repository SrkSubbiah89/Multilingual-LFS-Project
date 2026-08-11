"""
backend/rag/load_full_isco.py

Populates four Qdrant collections with the full ILO ISCO-08 hierarchy.

    isco08_major_groups     (10 entries  — 1-digit codes)
    isco08_submajor_groups  (43 entries  — 2-digit codes)
    isco08_minor_groups     (130 entries — 3-digit codes)
    isco08_unit_groups      (436 entries — 4-digit codes)

Each Qdrant point payload:
    code         : ISCO-08 code string
    label_en     : English label
    label_ar     : Arabic label
    parent_code  : parent code (empty string for major groups)
    description  : short English description

Embedding: intfloat/multilingual-e5-small (384-dim, "passage: " prefix).

Usage
-----
    python -m backend.rag.load_full_isco
    python -m backend.rag.load_full_isco --recreate   # drop and recreate collections
"""

from __future__ import annotations

import argparse
import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MODEL_NAME  = "intfloat/multilingual-e5-small"
VECTOR_DIM  = 384
_BATCH_SIZE = 32
_PREFIX     = "passage: "

COL_MAJOR    = "isco08_major_groups"
COL_SUBMAJOR = "isco08_submajor_groups"
COL_MINOR    = "isco08_minor_groups"
COL_UNIT     = "isco08_unit_groups"

# ---------------------------------------------------------------------------
# Raw data  (code, label_en, label_ar)
# parent_code is derived: code[:-1]  (empty string for 1-digit codes)
# description is generated from label_en for embedding richness
# ---------------------------------------------------------------------------

_MAJOR: list[tuple[str, str, str]] = [
    ("0", "Armed Forces Occupations",                        "أفراد القوات المسلحة"),
    ("1", "Managers",                                        "المديرون"),
    ("2", "Professionals",                                   "المهنيون"),
    ("3", "Technicians and Associate Professionals",         "الفنيون والمهنيون المساعدون"),
    ("4", "Clerical Support Workers",                        "موظفو الدعم الكتابي"),
    ("5", "Service and Sales Workers",                       "عمال الخدمات والمبيعات"),
    ("6", "Skilled Agricultural, Forestry and Fishery Workers", "العمال المهرة في الزراعة والغابات والصيد"),
    ("7", "Craft and Related Trades Workers",                "عمال الحرف اليدوية والمهن ذات الصلة"),
    ("8", "Plant and Machine Operators, and Assemblers",     "مشغلو المنشآت والآلات والمجمّعون"),
    ("9", "Elementary Occupations",                          "المهن الأولية"),
]

_SUBMAJOR: list[tuple[str, str, str]] = [
    # 0
    ("01", "Commissioned Armed Forces Officers",             "ضباط القوات المسلحة المعيّنون"),
    ("02", "Non-commissioned Armed Forces Officers",         "ضباط الصف في القوات المسلحة"),
    ("03", "Armed Forces Occupations, Other Ranks",          "مهن القوات المسلحة، الرتب الأخرى"),
    # 1
    ("11", "Chief Executives, Senior Officials and Legislators", "المديرون التنفيذيون والمسؤولون الكبار"),
    ("12", "Administrative and Commercial Managers",         "المديرون الإداريون والتجاريون"),
    ("13", "Production and Specialised Services Managers",   "مديرو الإنتاج والخدمات المتخصصة"),
    ("14", "Hospitality, Retail and Other Services Managers","مديرو الضيافة والتجزئة والخدمات الأخرى"),
    # 2
    ("21", "Science and Engineering Professionals",          "متخصصو العلوم والهندسة"),
    ("22", "Health Professionals",                           "المتخصصون الصحيون"),
    ("23", "Teaching Professionals",                         "المتخصصون في التدريس"),
    ("24", "Business and Administration Professionals",      "متخصصو الأعمال والإدارة"),
    ("25", "Information and Communications Technology Professionals", "متخصصو تكنولوجيا المعلومات والاتصالات"),
    ("26", "Legal, Social and Cultural Professionals",       "المتخصصون القانونيون والاجتماعيون والثقافيون"),
    # 3
    ("31", "Science and Engineering Associate Professionals","الفنيون المساعدون في العلوم والهندسة"),
    ("32", "Health Associate Professionals",                 "الفنيون الصحيون المساعدون"),
    ("33", "Business and Administration Associate Professionals", "الفنيون المساعدون في الأعمال والإدارة"),
    ("34", "Legal, Social, Cultural and Related Associate Professionals", "الفنيون المساعدون القانونيون والاجتماعيون"),
    ("35", "Information and Communications Technicians",     "فنيو المعلومات والاتصالات"),
    # 4
    ("41", "General and Keyboard Clerks",                    "الكتّاب العموميون وموظفو لوحة المفاتيح"),
    ("42", "Customer Services Clerks",                       "موظفو خدمة العملاء"),
    ("43", "Numerical and Material Recording Clerks",        "موظفو التسجيل الرقمي والمادي"),
    ("44", "Other Clerical Support Workers",                 "موظفو الدعم الكتابي الآخرون"),
    # 5
    ("51", "Personal Service Workers",                       "عمال الخدمات الشخصية"),
    ("52", "Sales Workers",                                  "عمال المبيعات"),
    ("53", "Personal Care Workers",                          "عمال الرعاية الشخصية"),
    ("54", "Protective Services Workers",                    "عمال الخدمات الوقائية"),
    # 6
    ("61", "Market-oriented Skilled Agricultural Workers",   "العمال الزراعيون المهرة الموجّهون للسوق"),
    ("62", "Market-oriented Skilled Forestry, Fishing and Hunting Workers", "عمال الغابات والصيد المهرة"),
    ("63", "Subsistence Farmers, Fishers, Hunters and Gatherers", "المزارعون وصيادو الكفاف"),
    # 7
    ("71", "Building and Related Trades Workers (exc Electricians)", "عمال البناء والمهن ذات الصلة"),
    ("72", "Metal, Machinery and Related Trades Workers",    "عمال المعادن والآلات"),
    ("73", "Handicraft and Printing Workers",                "عمال الحرف اليدوية والطباعة"),
    ("74", "Electrical and Electronics Trades Workers",      "عمال الكهرباء والإلكترونيات"),
    ("75", "Food Processing, Wood Working, Garment and Other Craft Workers", "عمال الغذاء والخشب والملابس"),
    # 8
    ("81", "Stationary Plant and Machine Operators",         "مشغلو الآلات والمنشآت الثابتة"),
    ("82", "Assemblers",                                     "عمال التجميع"),
    ("83", "Drivers and Mobile Plant Operators",             "السائقون ومشغلو المعدات المتنقلة"),
    # 9
    ("91", "Cleaners and Helpers",                           "عمال التنظيف والمساعدة"),
    ("92", "Agricultural, Forestry and Fishery Labourers",   "عمال الزراعة والغابات والصيد الأولية"),
    ("93", "Labourers in Mining, Construction, Manufacturing and Transport", "العمال في التعدين والبناء والنقل"),
    ("94", "Food Preparation Assistants",                    "مساعدو تحضير الطعام"),
    ("95", "Street and Related Sales and Service Workers",   "عمال المبيعات والخدمات في الشوارع"),
    ("96", "Refuse Workers and Other Elementary Workers",    "عمال جمع النفايات والعمال الأوليون"),
]

# Minor groups: (code, label_en, label_ar)
_MINOR: list[tuple[str, str, str]] = [
    ("011","Commissioned Armed Forces Officers","ضباط القوات المسلحة"),
    ("021","Non-commissioned Armed Forces Officers","ضباط الصف"),
    ("031","Armed Forces Occupations, Other Ranks","مهن القوات المسلحة"),
    ("111","Legislators and Senior Officials","المشرعون والمسؤولون الكبار"),
    ("112","Managing Directors and Chief Executives","المديرون التنفيذيون والرؤساء"),
    ("121","Business Services and Administration Managers","مديرو خدمات الأعمال والإدارة"),
    ("122","Sales, Marketing and Development Managers","مديرو المبيعات والتسويق"),
    ("131","Production Managers in Agriculture, Forestry and Fisheries","مديرو الإنتاج الزراعي"),
    ("132","Manufacturing, Mining, Construction and Distribution Managers","مديرو التصنيع والتعدين والبناء"),
    ("133","Information and Communications Technology Service Managers","مديرو خدمات تكنولوجيا المعلومات"),
    ("134","Professional Services Managers","مديرو الخدمات المهنية"),
    ("141","Hotel and Restaurant Managers","مديرو الفنادق والمطاعم"),
    ("142","Retail and Wholesale Trade Managers","مديرو التجزئة والجملة"),
    ("143","Other Services Managers","مديرو الخدمات الأخرى"),
    ("211","Physical and Earth Science Professionals","متخصصو علوم الفيزياء والأرض"),
    ("212","Mathematicians, Actuaries and Statisticians","علماء الرياضيات والإحصاء"),
    ("213","Life Science Professionals","متخصصو علوم الحياة"),
    ("214","Engineering Professionals (exc Electrotechnology)","المهندسون (غير الكهربائيين)"),
    ("215","Electrotechnology Engineers","مهندسو الكهروتقنية"),
    ("216","Architects, Planners, Surveyors and Designers","المهندسون المعماريون والمخططون"),
    ("221","Medical Doctors","الأطباء"),
    ("222","Nursing and Midwifery Professionals","الممرضون وقابلات الولادة"),
    ("223","Traditional and Complementary Medicine Professionals","متخصصو الطب التقليدي"),
    ("224","Paramedical Practitioners","الممارسون الطبيون المساعدون"),
    ("225","Veterinarians","الأطباء البيطريون"),
    ("226","Other Health Professionals","متخصصون صحيون آخرون"),
    ("231","University and Higher Education Teachers","أساتذة الجامعات"),
    ("232","Vocational Education Teachers","مدرسو التعليم المهني"),
    ("233","Secondary Education Teachers","مدرسو التعليم الثانوي"),
    ("234","Primary School and Early Childhood Teachers","مدرسو التعليم الأساسي"),
    ("235","Other Teaching Professionals","متخصصون في التدريس آخرون"),
    ("241","Finance Professionals","المتخصصون الماليون"),
    ("242","Administration Professionals","متخصصو الإدارة"),
    ("243","Sales, Marketing and Public Relations Professionals","متخصصو المبيعات والتسويق"),
    ("251","Software and Applications Developers and Analysts","مطورو البرمجيات والتطبيقات"),
    ("252","Database and Network Professionals","متخصصو قواعد البيانات والشبكات"),
    ("261","Legal Professionals","المتخصصون القانونيون"),
    ("262","Librarians, Archivists and Curators","أمناء المكتبات والمحفوظات"),
    ("263","Social and Religious Professionals","المتخصصون الاجتماعيون والدينيون"),
    ("264","Authors, Journalists and Linguists","الكتّاب والصحفيون واللغويون"),
    ("265","Creative and Performing Artists","الفنانون المبدعون والمؤدّون"),
    ("311","Physical and Engineering Science Technicians","فنيو العلوم الفيزيائية والهندسية"),
    ("312","Mining, Manufacturing and Construction Supervisors","مشرفو التعدين والتصنيع والبناء"),
    ("313","Process Control Technicians","فنيو التحكم في العمليات"),
    ("314","Life Science Technicians and Related Associate Professionals","فنيو علوم الحياة"),
    ("315","Ship and Aircraft Controllers and Technicians","مراقبو السفن والطائرات"),
    ("321","Medical and Pharmaceutical Technicians","الفنيون الطبيون والصيدلانيون"),
    ("322","Nursing and Midwifery Associate Professionals","مساعدو التمريض"),
    ("323","Traditional and Complementary Medicine Associate Professionals","مساعدو الطب التقليدي"),
    ("324","Veterinary Technicians and Assistants","المساعدون البيطريون"),
    ("325","Other Health Associate Professionals","فنيون صحيون آخرون"),
    ("331","Financial and Mathematical Associate Professionals","الفنيون الماليون والرياضيون"),
    ("332","Sales and Purchasing Agents and Brokers","وكلاء المبيعات والمشتريات"),
    ("333","Business Services Agents","وكلاء خدمات الأعمال"),
    ("334","Administrative and Specialised Secretaries","السكرتاريون الإداريون"),
    ("335","Government Regulatory Associate Professionals","المفتشون الحكوميون"),
    ("341","Legal, Social and Religious Associate Professionals","المساعدون القانونيون والاجتماعيون"),
    ("342","Sports and Fitness Workers","عمال الرياضة واللياقة"),
    ("343","Artistic, Cultural and Culinary Associate Professionals","الفنيون الثقافيون والطهاة"),
    ("351","Information and Communications Technicians","فنيو المعلومات والاتصالات"),
    ("352","Telecommunications and Broadcasting Technicians","فنيو الاتصالات والبث"),
    ("411","General Office Clerks","الكتّاب العموميون"),
    ("412","Secretaries (General)","السكرتاريون العموميون"),
    ("413","Keyboard Operators","مشغلو لوحة المفاتيح"),
    ("421","Tellers, Money Collectors and Related Clerks","الصرافون وجامعو النقود"),
    ("422","Client Information Workers","موظفو معلومات العملاء"),
    ("431","Numerical Clerks","الموظفون الرقميون"),
    ("432","Material-recording and Transport Clerks","موظفو التسجيل المادي والنقل"),
    ("441","Other Clerical Support Workers","موظفو الدعم الكتابي الآخرون"),
    ("511","Travel Attendants, Conductors and Guides","مرافقو السفر والمرشدون"),
    ("512","Cooks","الطهاة"),
    ("513","Waiters and Bartenders","النادلون والبارمانيون"),
    ("514","Hairdressers, Beauticians and Related Workers","الحلاقون ومتخصصو التجميل"),
    ("515","Building and Housekeeping Supervisors","مشرفو المباني والتدبير المنزلي"),
    ("516","Other Personal Services Workers","عمال الخدمات الشخصية الآخرون"),
    ("521","Street and Market Salespersons","بائعو الشوارع والأسواق"),
    ("522","Shop Salespersons","بائعو المحلات"),
    ("523","Cashiers and Ticket Clerks","أمناء الصناديق وموظفو التذاكر"),
    ("524","Other Sales Workers","عمال المبيعات الآخرون"),
    ("531","Child Care Workers and Teachers' Aides","عمال رعاية الأطفال"),
    ("532","Personal Care Workers in Health Services","عمال الرعاية الشخصية الصحية"),
    ("541","Protective Services Workers","عمال الخدمات الوقائية"),
    ("611","Market Gardeners and Crop Growers","العمال الزراعيون الموجّهون للسوق"),
    ("612","Animal Producers","منتجو الحيوانات"),
    ("613","Mixed Crop and Animal Producers","منتجو المحاصيل والحيوانات"),
    ("621","Forestry and Related Workers","عمال الغابات"),
    ("622","Fishery Workers, Hunters and Trappers","الصيادون وصائدو الحيوانات"),
    ("631","Subsistence Crop Farmers","مزارعو الكفاف"),
    ("632","Subsistence Livestock Farmers","مربو الكفاف"),
    ("633","Subsistence Mixed Crop and Livestock Farmers","مزارعو الكفاف المختلطون"),
    ("634","Subsistence Fishers, Hunters, Trappers and Gatherers","صيادو الكفاف"),
    ("711","Building Frame and Related Trades Workers","عمال هياكل البناء"),
    ("712","Building Finishers and Related Trades Workers","عمال تشطيبات البناء"),
    ("713","Painters, Building Structure Cleaners and Related Trades Workers","الدهانون ومنظفو المباني"),
    ("721","Sheet and Structural Metal Workers, Moulders and Welders","عمال المعادن الصفائحية واللحامون"),
    ("722","Blacksmiths, Tool Makers and Related Trades Workers","الحدادون وصانعو الأدوات"),
    ("723","Machinery Mechanics and Repairers","ميكانيكيو الآلات"),
    ("731","Handicraft Workers","عمال الحرف اليدوية"),
    ("732","Printing Trades Workers","عمال الطباعة"),
    ("741","Electrical Equipment Installers and Repairers","مثبتو وصائنو المعدات الكهربائية"),
    ("742","Electronics and Telecommunications Installers and Repairers","فنيو الإلكترونيات والاتصالات"),
    ("751","Food Processing and Related Trades Workers","عمال معالجة الأغذية"),
    ("752","Wood Treaters, Cabinet Makers and Related Trades Workers","عمال معالجة الخشب"),
    ("753","Garment and Related Trades Workers","عمال الملابس"),
    ("754","Other Craft and Related Workers","عمال الحرف الآخرون"),
    ("811","Mining and Mineral Processing Plant Operators","مشغلو مصانع التعدين"),
    ("812","Metal Processing and Finishing Plant Operators","مشغلو مصانع المعادن"),
    ("813","Chemical and Photographic Products Plant Operators","مشغلو مصانع المواد الكيماوية"),
    ("814","Rubber, Plastic and Paper Products Machine Operators","مشغلو آلات المطاط والبلاستيك"),
    ("815","Textile, Fur and Leather Products Machine Operators","مشغلو آلات النسيج والجلود"),
    ("816","Food and Related Products Machine Operators","مشغلو آلات الأغذية"),
    ("817","Wood Processing and Papermaking Plant Operators","مشغلو مصانع الخشب والورق"),
    ("818","Other Stationary Plant and Machine Operators","مشغلو الآلات الأخرى"),
    ("821","Assemblers","عمال التجميع"),
    ("831","Locomotive Engine Drivers and Related Workers","سائقو القطارات"),
    ("832","Car, Van and Motorcycle Drivers","سائقو السيارات والدراجات"),
    ("833","Heavy Truck and Bus Drivers","سائقو الشاحنات الثقيلة والحافلات"),
    ("834","Mobile Plant Operators","مشغلو المعدات المتنقلة"),
    ("835","Ships' Deck Crews and Related Workers","طواقم سطح السفن"),
    ("911","Domestic, Hotel and Office Cleaners and Helpers","عمال التنظيف المنزلي والفندقي"),
    ("912","Vehicle, Window, Laundry and Other Hand Cleaning Workers","عمال تنظيف المركبات"),
    ("913","Building and Related Caretakers","حراس المباني"),
    ("921","Agricultural, Forestry and Fishery Labourers","عمال الزراعة والغابات"),
    ("931","Mining and Construction Labourers","عمال التعدين والبناء"),
    ("932","Manufacturing Labourers","عمال التصنيع"),
    ("933","Transport and Storage Labourers","عمال النقل والتخزين"),
    ("941","Food Preparation Assistants","مساعدو تحضير الطعام"),
    ("951","Street and Related Service Workers","عمال الخدمات في الشوارع"),
    ("952","Street Vendors (exc Food)","الباعة المتجولون"),
    ("961","Refuse Workers","عمال جمع النفايات"),
    ("962","Other Elementary Workers","عمال أوليون آخرون"),
]

# Unit groups: (code, label_en)  — Arabic derived from parent label
_UNIT: list[tuple[str, str]] = [
    # 0 Armed Forces
    ("0110","Commissioned Armed Forces Officers"),
    ("0210","Non-commissioned Armed Forces Officers"),
    ("0310","Armed Forces Occupations, Other Ranks"),
    # 11
    ("1111","Legislators"),
    ("1112","Senior Government Officials"),
    ("1113","Traditional Chiefs and Heads of Village"),
    ("1114","Senior Officials of Special-Interest Organizations"),
    ("1120","Managing Directors and Chief Executives"),
    # 12
    ("1211","Finance Managers"),
    ("1212","Human Resource Managers"),
    ("1213","Policy and Planning Managers"),
    ("1219","Business Services and Administration Managers, NEC"),
    ("1221","Sales and Marketing Managers"),
    ("1222","Advertising and Public Relations Managers"),
    ("1223","Research and Development Managers"),
    # 13
    ("1311","Agricultural and Forestry Production Managers"),
    ("1312","Aquaculture and Fisheries Production Managers"),
    ("1321","Manufacturing Managers"),
    ("1322","Mining Managers"),
    ("1323","Construction Managers"),
    ("1324","Supply, Distribution and Related Managers"),
    ("1330","Information and Communications Technology Managers"),
    ("1341","Child Care Services Managers"),
    ("1342","Health Services Managers"),
    ("1343","Aged Care Services Managers"),
    ("1344","Social Welfare Managers"),
    ("1345","Education Managers"),
    ("1346","Financial and Insurance Services Branch Managers"),
    ("1347","Professional Services Managers, NEC"),
    ("1349","Other Services Managers, NEC"),
    # 14
    ("1411","Hotel Managers"),
    ("1412","Restaurant Managers"),
    ("1420","Retail and Wholesale Trade Managers"),
    ("1431","Sports, Recreation and Cultural Centre Managers"),
    ("1439","Other Services Managers, NEC"),
    # 21
    ("2111","Physicists and Astronomers"),
    ("2112","Meteorologists"),
    ("2113","Chemists"),
    ("2114","Geologists and Geophysicists"),
    ("2120","Mathematicians, Actuaries and Statisticians"),
    ("2131","Biologists, Botanists, Zoologists and Related Professionals"),
    ("2132","Farming, Forestry and Fisheries Advisers"),
    ("2133","Environmental Protection Professionals"),
    ("2141","Industrial and Production Engineers"),
    ("2142","Civil Engineers"),
    ("2143","Environmental Engineers"),
    ("2144","Mechanical Engineers"),
    ("2145","Chemical Engineers"),
    ("2146","Mining Engineers, Metallurgists and Related Professionals"),
    ("2149","Engineering Professionals, NEC"),
    ("2151","Electrical Engineers"),
    ("2152","Electronics Engineers"),
    ("2153","Telecommunications Engineers"),
    ("2161","Building Architects"),
    ("2162","Landscape Architects"),
    ("2163","Product and Garment Designers"),
    ("2164","Town and Traffic Planners"),
    ("2165","Cartographers and Surveyors"),
    ("2166","Graphic and Multimedia Designers"),
    # 22
    ("2211","Generalist Medical Practitioners"),
    ("2212","Specialist Medical Practitioners"),
    ("2221","Nursing Professionals"),
    ("2222","Midwifery Professionals"),
    ("2230","Traditional and Complementary Medicine Professionals"),
    ("2240","Paramedical Practitioners"),
    ("2250","Veterinarians"),
    ("2261","Dentists"),
    ("2262","Pharmacists"),
    ("2263","Environmental and Occupational Health Professionals"),
    ("2264","Physiotherapists"),
    ("2265","Dieticians and Nutritionists"),
    ("2266","Audiologists and Speech Therapists"),
    ("2267","Optometrists and Ophthalmic Opticians"),
    ("2269","Health Professionals, NEC"),
    # 23
    ("2310","University and Higher Education Teachers"),
    ("2320","Vocational Education Teachers"),
    ("2330","Secondary Education Teachers"),
    ("2341","Primary School Teachers"),
    ("2342","Early Childhood Educators"),
    ("2351","Education Methods Specialists"),
    ("2352","Special Needs Teachers"),
    ("2353","Other Language Teachers"),
    ("2354","Other Music Teachers"),
    ("2355","Other Arts Teachers"),
    ("2356","Information Technology Trainers"),
    ("2359","Teaching Professionals, NEC"),
    # 24
    ("2411","Accountants"),
    ("2412","Financial and Investment Advisers"),
    ("2413","Financial Analysts"),
    ("2421","Management and Organization Analysts"),
    ("2422","Policy Administration Professionals"),
    ("2423","Personnel and Career Professionals"),
    ("2424","Training and Staff Development Professionals"),
    ("2431","Advertising and Marketing Professionals"),
    ("2432","Public Relations Professionals"),
    ("2433","Technical and Medical Sales Professionals"),
    ("2434","Information and Communications Technology Sales Professionals"),
    # 25
    ("2511","Systems Analysts"),
    ("2512","Software Developers"),
    ("2513","Web and Multimedia Developers"),
    ("2514","Applications Programmers"),
    ("2519","Software and Applications Developers and Analysts, NEC"),
    ("2521","Database Designers and Administrators"),
    ("2522","Systems Administrators"),
    ("2523","Computer Network Professionals"),
    ("2529","Database and Network Professionals, NEC"),
    # 26
    ("2611","Lawyers"),
    ("2612","Judges"),
    ("2619","Legal Professionals, NEC"),
    ("2621","Archivists and Curators"),
    ("2622","Librarians and Related Information Professionals"),
    ("2631","Economists"),
    ("2632","Sociologists, Anthropologists and Related Professionals"),
    ("2633","Philosophers, Historians and Political Scientists"),
    ("2634","Psychologists"),
    ("2635","Social Work and Counselling Professionals"),
    ("2636","Religious Professionals"),
    ("2641","Authors and Related Writers"),
    ("2642","Journalists"),
    ("2643","Translators, Interpreters and Other Linguists"),
    ("2651","Visual Artists"),
    ("2652","Musicians, Singers and Composers"),
    ("2653","Dancers and Choreographers"),
    ("2654","Film, Stage and Related Directors and Producers"),
    ("2655","Actors"),
    ("2656","Announcers on Radio, Television and Other Media"),
    ("2659","Creative and Performing Artists, NEC"),
    # 31
    ("3111","Chemical and Physical Science Technicians"),
    ("3112","Civil Engineering Technicians"),
    ("3113","Electrical Engineering Technicians"),
    ("3114","Electronics Engineering Technicians"),
    ("3115","Mechanical Engineering Technicians"),
    ("3116","Chemical Engineering Technicians"),
    ("3117","Mining and Metallurgical Technicians"),
    ("3118","Draughtspersons and Related Workers"),
    ("3119","Physical and Engineering Science Technicians, NEC"),
    ("3121","Mining Supervisors and Technicians"),
    ("3122","Manufacturing Supervisors"),
    ("3123","Construction Supervisors"),
    ("3131","Power Production Plant Operators"),
    ("3132","Incinerator and Water Treatment Plant Operators"),
    ("3133","Chemical Processing Plant Controllers"),
    ("3134","Petroleum and Natural Gas Refining Plant Operators"),
    ("3135","Metal Production Process Controllers"),
    ("3139","Process Control Technicians, NEC"),
    ("3141","Life Science Technicians (excluding Medical)"),
    ("3142","Agricultural Technicians"),
    ("3143","Forestry Technicians"),
    ("3151","Ships' Engineers"),
    ("3152","Ships' Deck Officers and Pilots"),
    ("3153","Aircraft Pilots and Related Associate Professionals"),
    ("3154","Air Traffic Controllers"),
    ("3155","Air Traffic Safety Electronics Technicians"),
    # 32
    ("3211","Medical Imaging and Therapeutic Equipment Technicians"),
    ("3212","Medical and Pathology Laboratory Technicians"),
    ("3213","Pharmaceutical Technicians and Assistants"),
    ("3214","Medical and Dental Prosthetic Technicians"),
    ("3221","Nursing Associate Professionals"),
    ("3222","Midwifery Associate Professionals"),
    ("3230","Traditional and Complementary Medicine Associate Professionals"),
    ("3240","Veterinary Technicians and Assistants"),
    ("3251","Dental Assistants and Therapists"),
    ("3252","Medical Records and Health Information Technicians"),
    ("3253","Community Health Workers"),
    ("3254","Dispensing Opticians"),
    ("3255","Physiotherapy Technicians and Assistants"),
    ("3256","Medical Assistants"),
    ("3257","Environmental and Occupational Health Inspectors and Associates"),
    ("3258","Ambulance Workers"),
    ("3259","Health Associate Professionals, NEC"),
    # 33
    ("3311","Securities and Finance Dealers and Brokers"),
    ("3312","Credit and Loans Officers"),
    ("3313","Accounting Associate Professionals"),
    ("3314","Statistical, Mathematical and Related Associate Professionals"),
    ("3315","Valuers and Loss Assessors"),
    ("3321","Insurance Representatives"),
    ("3322","Commercial Sales Representatives"),
    ("3323","Buyers"),
    ("3324","Trade Brokers"),
    ("3331","Clearing and Forwarding Agents"),
    ("3332","Conference and Event Planners"),
    ("3333","Employment Agents and Contractors"),
    ("3334","Real Estate Agents and Property Managers"),
    ("3339","Business Services Agents, NEC"),
    ("3341","Office Supervisors"),
    ("3342","Legal Secretaries"),
    ("3343","Administrative and Executive Secretaries"),
    ("3344","Medical Secretaries"),
    ("3351","Customs and Border Inspectors"),
    ("3352","Government Tax and Excise Officials"),
    ("3353","Government Social Benefits Officials"),
    ("3354","Government Licensing Officials"),
    ("3355","Police Inspectors and Detectives"),
    ("3359","Government Regulatory Associate Professionals, NEC"),
    # 34
    ("3411","Legal and Related Associate Professionals"),
    ("3412","Social Work Associate Professionals"),
    ("3413","Religious Associate Professionals"),
    ("3421","Athletes and Sports Players"),
    ("3422","Sports Coaches, Instructors and Officials"),
    ("3423","Fitness and Recreation Instructors and Programme Leaders"),
    ("3431","Photographers"),
    ("3432","Interior Designers and Decorators"),
    ("3433","Gallery, Museum and Library Technicians"),
    ("3434","Chefs"),
    ("3435","Other Artistic and Cultural Associate Professionals"),
    # 35
    ("3511","Information and Communications Technology Operations Technicians"),
    ("3512","Information and Communications Technology User Support Technicians"),
    ("3513","Computer Network and Systems Technicians"),
    ("3514","Web Technicians"),
    ("3521","Broadcasting and Audiovisual Technicians"),
    ("3522","Telecommunications Engineering Technicians"),
    # 41
    ("4110","General Office Clerks"),
    ("4120","Secretaries (General)"),
    ("4131","Typists and Word Processing Operators"),
    ("4132","Data Entry Clerks"),
    # 42
    ("4211","Bank Tellers and Related Clerks"),
    ("4212","Bookmakers, Lottery and Related Gaming Clerks"),
    ("4213","Pawnbrokers and Money Lenders"),
    ("4214","Debt Collectors and Related Workers"),
    ("4221","Travel Consultants and Clerks"),
    ("4222","Contact Centre Information Clerks"),
    ("4223","Telephone Switchboard Operators"),
    ("4224","Hotel Receptionists"),
    ("4225","Inquiry Clerks"),
    ("4226","Receptionists (General)"),
    ("4227","Survey and Market Research Interviewers"),
    ("4229","Client Information Workers, NEC"),
    # 43
    ("4311","Accounting and Bookkeeping Clerks"),
    ("4312","Statistical, Finance and Insurance Clerks"),
    ("4313","Payroll Clerks"),
    ("4321","Stock Clerks and Storekeepers"),
    ("4322","Production Clerks"),
    ("4323","Transport Clerks"),
    # 44
    ("4411","Library Clerks"),
    ("4412","Mail Carriers and Sorting Clerks"),
    ("4413","Coding, Proofreading and Related Clerks"),
    ("4414","Scribes and Related Workers"),
    ("4415","Filing and Copying Clerks"),
    ("4416","Personnel Clerks"),
    ("4419","Clerical Support Workers, NEC"),
    # 51
    ("5111","Travel Attendants and Travel Stewards"),
    ("5112","Transport Conductors"),
    ("5113","Travel Guides"),
    ("5120","Cooks"),
    ("5131","Waiters"),
    ("5132","Bartenders"),
    ("5141","Hairdressers"),
    ("5142","Beauty Therapists and Related Workers"),
    ("5151","Cleaning and Housekeeping Supervisors"),
    ("5152","Domestic Housekeepers"),
    ("5153","Building Caretakers"),
    ("5161","Astrologers, Fortune Tellers and Related Workers"),
    ("5162","Companions and Valets"),
    ("5163","Undertakers and Embalmers"),
    ("5164","Pet Groomers and Related Workers"),
    ("5165","Driving Instructors"),
    ("5169","Personal Services Workers, NEC"),
    # 52
    ("5211","Stall and Market Salespersons"),
    ("5212","Street Food Salespersons"),
    ("5221","Shop Salespersons"),
    ("5222","Shop Supervisors"),
    ("5223","Cashiers and Ticket Clerks"),
    ("5230","Fuel Station Attendants"),
    ("5241","Fashion and Other Models"),
    ("5242","Sales Demonstrators"),
    ("5243","Door-to-Door Salespersons"),
    ("5244","Contact Centre Salespersons"),
    ("5245","Service Station Attendants"),
    ("5246","Food Service Counter Attendants and Food Preparers"),
    ("5249","Sales Workers, NEC"),
    # 53
    ("5311","Child Care Workers"),
    ("5312","Teachers' Aides"),
    ("5321","Health Care Assistants"),
    ("5322","Home-Based Personal Care Workers"),
    ("5329","Personal Care Workers, NEC"),
    # 54
    ("5411","Fire Fighters"),
    ("5412","Police Officers"),
    ("5413","Prison Guards"),
    ("5414","Security Guards and Related Workers"),
    ("5419","Protective Services Workers, NEC"),
    # 61
    ("6111","Crop Growers"),
    ("6112","Vegetable and Related Crop Growers"),
    ("6113","Gardeners and Horticultural Producers"),
    ("6114","Mixed Crop Growers"),
    ("6121","Livestock Producers"),
    ("6122","Dairy and Livestock Producers"),
    ("6123","Poultry Producers"),
    ("6124","Apiarists and Sericulturists"),
    ("6129","Animal Producers, NEC"),
    ("6130","Mixed Crop and Animal Producers"),
    # 62
    ("6141","Forestry Workers"),
    ("6142","Charcoal Burners and Related Workers"),
    ("6150","Aquaculture Workers"),
    # Task: Phase II Module A Week 1 found these 4 unit codes filed under
    # the wrong sub-major group -- minor groups 613/621/622 and submajor
    # 61/62 have no matching "616x" parent anywhere in this file's own
    # _MINOR/_SUBMAJOR tables (verified: no "614", "615", or "616" minor
    # group is defined at all), while submajor 63 ("Subsistence Farmers,
    # Fishers, Hunters and Gatherers") and minor groups 631-634 already
    # exist correctly and were simply missing their unit-group children.
    # Same occupational content, same labels -- only the code numbers
    # were wrong. See Documentation/Phase_2/Week_1/module_a_week1_report.md
    # Sec.5.3. (The remaining 15 project-only / 10 official-only code
    # discrepancies that report also found are NOT fixed here -- they
    # need a full cross-check against the official ISCO-08 structure
    # document, not a guess; see that report's own scope boundary.)
    ("6310","Subsistence Crop Farmers"),
    ("6320","Subsistence Livestock Farmers"),
    ("6330","Subsistence Mixed Crop and Livestock Farmers"),
    ("6340","Subsistence Fishers, Hunters, Trappers and Gatherers"),
    # 71
    ("7111","House Builders"),
    ("7112","Bricklayers and Related Workers"),
    ("7113","Stone Cutters and Carvers"),
    ("7114","Concrete Placers, Concrete Finishers and Related Workers"),
    ("7115","Carpenters and Joiners"),
    ("7116","Other Building Frame and Related Trades Workers"),
    ("7121","Roofers"),
    ("7122","Floor Layers and Tile Setters"),
    ("7123","Plasterers"),
    ("7124","Insulation Workers"),
    ("7125","Glaziers"),
    ("7126","Plumbers and Pipe Fitters"),
    ("7127","Air Conditioning and Refrigeration Mechanics"),
    ("7131","Painters and Related Workers"),
    ("7132","Lacquerers and Varnishers"),
    ("7133","Building Structure Cleaners"),
    # 72
    ("7211","Metal Moulders and Core Makers"),
    ("7212","Welders and Flamecutters"),
    ("7213","Sheet Metal Workers"),
    ("7214","Structural Metal Preparers and Erectors"),
    ("7215","Riggers and Cable Splicers"),
    ("7221","Blacksmiths, Hammersmiths and Forging Press Workers"),
    ("7222","Tool Makers and Related Workers"),
    ("7223","Metal Working Machine Tool Setters and Operators"),
    ("7224","Metal Polishers, Wheel Grinders and Tool Sharpeners"),
    ("7231","Motor Vehicle Mechanics and Repairers"),
    ("7232","Aircraft Engine Mechanics and Repairers"),
    ("7233","Agricultural and Industrial Machinery Mechanics and Repairers"),
    ("7234","Bicycle and Related Repairers"),
    # 73
    ("7311","Precision Instrument Makers and Repairers"),
    ("7312","Musical Instrument Makers and Tuners"),
    ("7313","Jewellery and Precious Metal Workers"),
    ("7314","Potters and Related Workers"),
    ("7315","Glass Makers, Cutters, Grinders and Polishers"),
    ("7316","Sign Writers, Decorative Painters, Engravers and Etchers"),
    ("7317","Handicraft Workers in Textiles, Leather and Related Materials"),
    ("7318","Handicraft Workers in Hard Materials"),
    ("7319","Handicraft Workers, NEC"),
    ("7321","Pre-press Technicians"),
    ("7322","Printers"),
    ("7323","Print Finishing and Binding Workers"),
    # 74
    ("7411","Building and Related Electricians"),
    ("7412","Electrical Mechanics and Fitters"),
    ("7413","Electrical Line Installers and Repairers"),
    ("7421","Electronics Mechanics and Servicers"),
    ("7422","Information and Communications Technology Installers and Servicers"),
    # 75
    ("7511","Butchers, Fishmongers and Related Food Preparers"),
    ("7512","Bakers, Pastry Cooks and Confectionery Makers"),
    ("7513","Dairy Products Makers"),
    ("7514","Fruit, Vegetable and Related Preservers"),
    ("7515","Food and Beverage Tasters and Graders"),
    ("7516","Tobacco Preparers and Tobacco Products Makers"),
    ("7521","Wood Treaters"),
    ("7522","Cabinet Makers and Related Workers"),
    ("7523","Wood Products Machine Setters and Operators"),
    ("7531","Tailors, Dressmakers, Furriers and Hatters"),
    ("7532","Garment and Related Pattern Makers and Cutters"),
    ("7533","Sewing, Embroidery and Related Workers"),
    ("7534","Upholsterers and Related Workers"),
    ("7535","Pelt Dressers, Tanners and Fellmongers"),
    ("7536","Shoemakers and Related Workers"),
    ("7541","Underwater Divers"),
    ("7542","Shotfirers and Blasters"),
    ("7543","Product Graders and Testers (excluding Foods and Beverages)"),
    ("7544","Fumigators and Other Pest and Weed Controllers"),
    ("7549","Craft and Related Workers, NEC"),
    # 81
    ("8111","Miners and Quarriers"),
    ("8112","Mineral and Stone Processing Plant Operators"),
    ("8113","Well Drillers and Borers and Related Workers"),
    ("8114","Cement, Stone and Other Mineral Products Machine Operators"),
    ("8121","Metal Processing Plant Operators"),
    ("8122","Metal Finishing, Plating and Coating Machine Operators"),
    ("8131","Chemical Products Plant and Machine Operators"),
    ("8132","Photographic Products Machine Operators"),
    ("8141","Rubber Products Machine Operators"),
    ("8142","Plastic Products Machine Operators"),
    ("8143","Paper and Paperboard Products Machine Operators"),
    ("8151","Fibre Preparing, Spinning and Winding Machine Operators"),
    ("8152","Weaving and Knitting Machine Operators"),
    ("8153","Sewing Machine Operators"),
    ("8154","Bleaching, Dyeing and Fabric Cleaning Machine Operators"),
    ("8155","Fur and Leather Preparing Machine Operators"),
    ("8156","Shoe Making and Related Machine Operators"),
    ("8157","Laundry Machine Operators"),
    ("8159","Textile, Fur and Leather Products Machine Operators, NEC"),
    ("8160","Food and Related Products Machine Operators"),
    ("8171","Pulp and Papermaking Plant Operators"),
    ("8172","Wood Processing Plant Operators"),
    ("8181","Glass and Ceramics Plant Operators"),
    ("8182","Steam Engine and Boiler Operators"),
    ("8183","Packing, Bottling and Labelling Machine Operators"),
    ("8189","Stationary Plant and Machine Operators, NEC"),
    # 82
    ("8211","Mechanical Machinery Assemblers"),
    ("8212","Electrical and Electronic Equipment Assemblers"),
    ("8219","Assemblers, NEC"),
    # 83
    ("8311","Locomotive Engine Drivers and Related Workers"),
    ("8312","Railway Brake, Signal and Switches Operators"),
    ("8321","Motorcycle Drivers"),
    ("8322","Car, Taxi and Van Drivers"),
    ("8331","Bus and Tram Drivers"),
    ("8332","Heavy Truck and Lorry Drivers"),
    ("8341","Mobile Farm and Forestry Plant Operators"),
    ("8342","Earthmoving and Related Plant Operators"),
    ("8343","Crane, Hoist and Related Plant Operators"),
    ("8344","Lifting Truck Operators"),
    ("8350","Ships' Deck Crews and Related Workers"),
    # 91
    ("9111","Domestic Cleaners and Helpers"),
    ("9112","Cleaners and Helpers in Offices, Hotels and Other Establishments"),
    ("9121","Hand Launderers and Pressers"),
    ("9122","Vehicle Cleaners"),
    ("9123","Window Cleaners"),
    ("9129","Cleaning Workers, NEC"),
    ("9131","Domestic Housekeepers"),
    ("9132","Restaurant Services Workers"),
    ("9141","Building Caretakers"),
    ("9151","Messengers, Package Deliverers and Luggage Porters"),
    ("9152","Doorkeepers and Related Workers"),
    ("9153","Vending Machine Operators and Related Workers"),
    # 92
    ("9161","Refuse Workers"),
    ("9162","Sweepers and Related Labourers"),
    ("9211","Crop Farm Labourers"),
    ("9212","Livestock Farm Labourers"),
    ("9213","Mixed Crop and Livestock Farm Labourers"),
    ("9214","Garden and Horticultural Labourers"),
    ("9215","Forestry Labourers"),
    ("9216","Fishery and Aquaculture Labourers"),
    # 93
    ("9311","Mining and Quarrying Labourers"),
    ("9312","Civil Engineering Labourers"),
    ("9313","Building Construction Labourers"),
    ("9321","Hand Packers"),
    ("9329","Manufacturing Labourers, NEC"),
    ("9331","Hand and Pedal Vehicle Drivers"),
    ("9332","Drivers of Animal-Drawn Vehicles and Machinery"),
    ("9333","Freight Handlers"),
    ("9334","Shelf Fillers"),
    # 94
    ("9411","Fast Food Preparers"),
    ("9412","Kitchen Helpers"),
    # 95
    ("9420","Street and Related Service Workers"),
    ("9510","Refuse Sorters"),
    ("9520","Odd Job Workers"),
    # 96
    ("9611","Water and Firewood Collectors"),
    ("9612","Odd Job Workers"),
    ("9613","Scrap Collectors and Recyclers"),
    ("9621","Subsistence Agricultural, Forestry, Fishing and Hunting Labourers"),
]

# Build label_ar for unit groups from parent minor group
_MINOR_AR_MAP: dict[str, str] = {c: ar for c, _, ar in _MINOR}


def _unit_ar(code: str, label_en: str) -> str:
    """Best-effort Arabic: use parent minor group label."""
    return _MINOR_AR_MAP.get(code[:3], label_en)


# ---------------------------------------------------------------------------
# Rich synonym descriptions for unit groups
# ---------------------------------------------------------------------------
# These are appended to the embedding text so the model can match common
# synonyms and alternative phrasings (e.g. "chef" → 5120 Cooks, not 2151).
# Only the most commonly queried occupations are listed; unlisted codes fall
# back to the standard "code label" text.
# ---------------------------------------------------------------------------

_UNIT_SYNONYMS: dict[str, str] = {
    # ── Food service ────────────────────────────────────────────────────
    "5120": "chef head chef sous chef line cook prep cook kitchen staff culinary طاهي شيف",
    "3434": "head chef executive chef culinary artist pastry chef طاهي رئيسي",
    "7511": "butcher fishmonger meat cutter food preparer جزار",
    "7512": "baker pastry cook confectionery bread maker خباز حلواني",
    "9411": "fast food worker burger flipper food court prep kitchen helper",
    "9412": "kitchen assistant dishwasher kitchen helper مساعد مطبخ",

    # ── Healthcare ───────────────────────────────────────────────────────
    "2211": "doctor GP general practitioner family physician medical officer طبيب",
    "2212": "specialist doctor consultant physician specialist طبيب متخصص",
    "2221": "nurse registered nurse RN staff nurse ممرض",
    "2222": "midwife labour nurse قابلة",
    "3211": "lab technician medical laboratory technician blood tests فني مختبر",
    "3221": "enrolled nurse practical nurse LPN ممرض معاون",
    "5321": "nursing aide health care assistant ward helper مساعد تمريض",

    # ── Engineering ──────────────────────────────────────────────────────
    "2141": "industrial engineer production engineer process engineer مهندس صناعي",
    "2142": "civil engineer structural engineer infrastructure roads bridges مهندس مدني",
    "2143": "environmental engineer sustainability green energy مهندس بيئي",
    "2144": "mechanical engineer HVAC manufacturing مهندس ميكانيكي",
    "2151": "electrical engineer power systems wiring circuits مهندس كهربائي",
    "2152": "electronics engineer circuit design semiconductor مهندس إلكترونيات",
    "2153": "telecoms engineer network infrastructure communications مهندس اتصالات",
    "2161": "architect building design urban planning مهندس معماري",

    # ── IT / Software ────────────────────────────────────────────────────
    "2512": "software developer software engineer programmer coder app developer مطور برمجيات",
    "2511": "systems analyst business analyst IT analyst محلل نظم",
    "2513": "web developer front end back end full stack مطور ويب",
    "2514": "application programmer mobile developer iOS Android مبرمج تطبيقات",
    "2519": "ICT developer database developer AI developer مطور تقنية",
    "2521": "database administrator DBA database engineer مدير قواعد بيانات",
    "2522": "system administrator sysadmin network admin مدير شبكات",
    "3511": "IT support tech support helpdesk operations دعم تقني",
    "3512": "IT user support helpdesk technical support representative فني دعم",

    # ── Business / Finance ───────────────────────────────────────────────
    "2411": "financial analyst investment analyst portfolio manager محلل مالي",
    "2412": "financial adviser wealth manager financial planner مستشار مالي",
    "2421": "management consultant business consultant strategy مستشار أعمال",
    "2431": "advertising account executive brand manager مدير إعلان",
    "3311": "securities broker stock broker financial dealer وسيط مالي",
    "3312": "insurance agent insurance broker وكيل تأمين",
    "3313": "estate agent real estate agent property agent وكيل عقارات",
    "4311": "bookkeeper accounts clerk accounting assistant محاسب",

    # ── Teaching ─────────────────────────────────────────────────────────
    "2310": "university professor lecturer academic researcher أستاذ جامعي",
    "2320": "vocational teacher trade instructor training teacher مدرب مهني",
    "2330": "secondary school teacher high school teacher مدرس ثانوي",
    "2341": "primary school teacher elementary teacher مدرس ابتدائي",
    "2342": "kindergarten teacher early childhood educator مدرس روضة",
    "5312": "teaching assistant classroom aide teacher aide مساعد مدرس",

    # ── Trades ───────────────────────────────────────────────────────────
    "7111": "house builder construction worker builder بناء",
    "7112": "bricklayer mason block layer عامل بناء",
    "7115": "carpenter joiner woodworker نجار",
    "7121": "roofer roof installer سقف",
    "7411": "electrician electrical installer wiring كهربائي",
    "7412": "electrical fitter electrical technician فني كهربائي",
    "7231": "car mechanic auto mechanic vehicle repair ميكانيكي سيارات",

    # ── Drivers / Transport ──────────────────────────────────────────────
    "8322": "taxi driver Uber driver car driver Careem سائق تاكسي سائق",
    "8331": "bus driver public transport driver سائق حافلة",
    "8332": "truck driver lorry driver heavy vehicle driver سائق شاحنة",

    # ── Cleaning / Elementary ────────────────────────────────────────────
    "9111": "maid domestic worker cleaner housemaid عاملة منزل",
    "9112": "office cleaner hotel cleaner janitor عامل نظافة",
    "5153": "building caretaker concierge facilities caretaker حارس عمارة",

    # ── Sales / Retail ───────────────────────────────────────────────────
    "5221": "shop assistant sales assistant retail worker بائع محل",
    "5223": "cashier checkout operator أمين صندوق",
    "5244": "call centre agent telesales phone sales موظف مبيعات",

    # ── Security / Protection ────────────────────────────────────────────
    "5414": "security guard watchman security officer حارس أمن",
    "5412": "police officer police constable ضابط شرطة",

    # ── Construction ─────────────────────────────────────────────────────
    "9313": "construction labourer building site worker عامل بناء",
    "9312": "civil engineering labourer road worker عامل طرق",
}


def _unit_desc(code: str, label_en: str) -> str:
    """Return enriched text for embedding: label + any known synonyms."""
    synonyms = _UNIT_SYNONYMS.get(code, "")
    if synonyms:
        return f"{label_en} {synonyms}"
    return label_en


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True).tolist()
        all_vecs.extend(vecs)
    return all_vecs


# ---------------------------------------------------------------------------
# Collection builder
# ---------------------------------------------------------------------------

def _build_points(
    entries: list[tuple],          # (code, label_en, label_ar) or (code, label_en)
    has_ar: bool = True,
    id_offset: int = 0,
) -> tuple[list[str], list[dict]]:
    """Return (texts_for_embedding, payloads)."""
    texts, payloads = [], []
    for i, row in enumerate(entries):
        code     = row[0]
        label_en = row[1]
        label_ar = row[2] if has_ar else _unit_ar(code, label_en)
        parent   = code[:-1] if len(code) > 1 else ""
        # Enrich description with synonyms (improves embedding recall for unit groups)
        enriched = _unit_desc(code, label_en) if not has_ar else label_en
        desc     = f"ISCO-08 {code}: {enriched}"
        text     = f"{_PREFIX}{code} {enriched} {label_ar}"
        texts.append(text)
        payloads.append({
            "code": code, "label_en": label_en, "label_ar": label_ar,
            "parent_code": parent, "description": desc,
        })
    return texts, payloads


def _upsert(
    client: QdrantClient,
    model: SentenceTransformer,
    collection: str,
    entries: list[tuple],
    has_ar: bool = True,
    recreate: bool = False,
) -> int:
    if recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
    else:
        try:
            client.get_collection(collection)
        except Exception:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    texts, payloads = _build_points(entries, has_ar=has_ar)
    vectors = _embed(model, texts)

    points = [
        PointStruct(id=i + 1, vector=vectors[i], payload=payloads[i])
        for i in range(len(payloads))
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(recreate: bool = False) -> None:
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} …")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"Loading embedding model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    datasets = [
        (COL_MAJOR,    _MAJOR,   True),
        (COL_SUBMAJOR, _SUBMAJOR, True),
        (COL_MINOR,    _MINOR,   True),
        (COL_UNIT,     _UNIT,    False),
    ]

    for col, data, has_ar in datasets:
        n = _upsert(client, model, col, data, has_ar=has_ar, recreate=recreate)
        print(f"  {col}: {n} points upserted")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load full ISCO-08 hierarchy into Qdrant.")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate collections before loading.")
    args = parser.parse_args()
    main(recreate=args.recreate)
