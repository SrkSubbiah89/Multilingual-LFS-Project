/**
 * Survey Report page — research-grade light-themed bilingual output.
 */

import { useState, useEffect } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { getReport } from "../components/api";
import LanguageToggle from "../components/LanguageToggle";

// ── WISCO v2 controlled benchmark — real, published, sourced results ──────────
// Every number below matches
// Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md
// character-for-character. This is a controlled, externally sourced benchmark
// (WISCO v2, CC-BY-4.0, Zenodo DOI 10.5281/zenodo.8262593) — NOT real Labour
// Force Survey respondent data, and does not constitute field validation.
// Flat retrieval was substantially MORE accurate than strict hierarchical
// retrieval in this configuration; this must never be shown as a hierarchical
// win. See MANUSCRIPT_SAFE_WISCO_WORDING.md for the wording this section
// follows.
const WISCO_EVAL = {
  heldoutCases: 18747,
  headline: [
    { key: "flat", name: "Flat retrieval", correct: 3973, accuracy: 0.2119, ciLow: 0.2061, ciHigh: 0.2178 },
    { key: "hier", name: "Strict hierarchical retrieval", correct: 1941, accuracy: 0.1035, ciLow: 0.0993, ciHigh: 0.1080 },
  ],
  diffPct: -10.84,
  mcnemarP: "1.86 × 10⁻³⁰¹",
  perMajor: [
    { code: "0", flat: 0.1888, hier: 0.2937 },
    { code: "1", flat: 0.2923, hier: 0.1789 },
    { code: "2", flat: 0.2632, hier: 0.0839 },
    { code: "3", flat: 0.2364, hier: 0.0881 },
    { code: "4", flat: 0.2338, hier: 0.1829 },
    { code: "5", flat: 0.1947, hier: 0.0730 },
    { code: "6", flat: 0.1336, hier: 0.1064 },
    { code: "7", flat: 0.1444, hier: 0.0705 },
    { code: "8", flat: 0.1969, hier: 0.1843 },
    { code: "9", flat: 0.1585, hier: 0.0856 },
  ],
  perLanguage: [
    { code: "ar", label: "Arabic",  flat: 0.1462, hier: 0.0978 },
    { code: "en", label: "English", flat: 0.3798, hier: 0.2252 },
    { code: "hi", label: "Hindi",   flat: 0.2307, hier: 0.0809 },
    { code: "tl", label: "Tagalog", flat: 0.1447, hier: 0.0531 },
    { code: "ur", label: "Urdu",    flat: 0.1533, hier: 0.0571 },
  ],
};

// ── ISCO major group labels ────────────────────────────────────────────────────
const ISCO_MAJOR = {
  "0": { label: "Armed Forces",               color: "text-slate-700   bg-slate-100  border-slate-300"  },
  "1": { label: "Managers",                   color: "text-blue-700    bg-blue-50    border-blue-300"   },
  "2": { label: "Professionals",              color: "text-violet-700  bg-violet-50  border-violet-300" },
  "3": { label: "Technicians & Assoc. Prof.", color: "text-cyan-700    bg-cyan-50    border-cyan-300"   },
  "4": { label: "Clerical Support",           color: "text-teal-700    bg-teal-50    border-teal-300"   },
  "5": { label: "Service & Sales",            color: "text-emerald-700 bg-emerald-50 border-emerald-300"},
  "6": { label: "Skilled Agricultural",       color: "text-lime-700    bg-lime-50    border-lime-300"   },
  "7": { label: "Craft & Related Trades",     color: "text-orange-700  bg-orange-50  border-orange-300" },
  "8": { label: "Plant & Machine Operators",  color: "text-amber-700   bg-amber-50   border-amber-300"  },
  "9": { label: "Elementary Occupations",     color: "text-red-700     bg-red-50     border-red-300"    },
};

// ── Quality colours ────────────────────────────────────────────────────────────
const QUALITY_STYLES = {
  pass:      { bar: "bg-emerald-500", badge: "bg-emerald-50  border-emerald-300 text-emerald-700", dot: "bg-emerald-500" },
  fail:      { bar: "bg-amber-500",   badge: "bg-amber-50   border-amber-300  text-amber-700",   dot: "bg-amber-500"   },
  escalated: { bar: "bg-red-500",     badge: "bg-red-50     border-red-300    text-red-700",     dot: "bg-red-500"     },
  unknown:   { bar: "bg-gray-400",    badge: "bg-gray-100   border-gray-300   text-gray-600",    dot: "bg-gray-400"    },
};

// ── i18n ───────────────────────────────────────────────────────────────────────
const T = {
  en: {
    pageTitle: "Survey Report",
    loading: "Generating research report…",
    errorTitle: "Report unavailable",
    backToHome: "Back to home",
    signOut: "Sign out",
    sessionLabel: "Session",
    reportId: "Report",
    generatedAt: "Generated",
    iscoSection: "ISCO-08 Classification",
    iscoCode: "Unit Group Code",
    iscoMajor: "Major Group",
    iscoConfidence: "Classification Confidence",
    iscoMethod: "Method",
    hitlRequired: "Human Review Required",
    hitlOk: "AI Verified",
    profileSection: "Employment Profile",
    sectionB: "Section B — Demographics & Education",
    sectionC: "Section C/D/E — Employment Details",
    sectionFG: "Section F/G — Unemployment / Outside LF",
    sectionK: "Section K — Respondent Feedback",
    narrativeSection: "AI-Generated Narrative Report",
    qualitySection: "Data Quality Assessment",
    qualityScore: "Quality Score",
    flaggedFields: "Flagged Fields",
    qualityStatus: { pass: "Passed", fail: "Needs Review", escalated: "Escalated", unknown: "Not Assessed" },
    isicSection: "ISIC Rev.4 Industry Classification",
    isicSection4: "Section",
    isicDivision4: "Division",
    isicGroup4: "Group",
    isicClass4: "Class (4-digit)",
    isicConfidence: "Classification Confidence",
    isicNoData: "No ISIC classification available.",
    iscedSection: "ISCED 2011 Level + ISCED-F 2013 Field of Specialisation",
    iscedLevel: "Education Level",
    iscedBroad: "Broad Field (2-digit)",
    iscedNarrow: "Narrow Field (3-digit)",
    iscedDetailed: "Detailed Field (4-digit)",
    iscedConfidence: "Classification Confidence",
    iscedNoData: "No ISCED classification available.",
    semanticSection: "Cross-Standard Coherence (ISCO ↔ ISIC ↔ ISCED) — Core Thesis Contribution: the Semantic Relation Engine",
    semanticScore: "Coherence Score",
    semanticCoherent: "COHERENT",
    semanticIncoherent: "INCONSISTENT",
    semanticViolations: "Detected Violations",
    semanticNoData: "No cross-standard analysis available.",
    evalSection: "Retrieval Comparison — WISCO v2 Controlled Benchmark",
    evalSubtitle: "Controlled exact-code evaluation on an externally sourced multilingual occupation-title reference benchmark (WISCO v2, 18,747 held-out cases, official ILO 2021 ISCO-08 catalogue, LLM reranking disabled). This is not a Labour Force Survey field validation.",
    evalSystem: "Method", evalCases: "Cases", evalCorrect: "Exact matches", evalAccuracy: "Accuracy", evalCi: "95% Wilson CI",
    evalDiff: "Accuracy difference (hierarchical − flat)",
    evalMcnemar: "McNemar exact two-sided p-value",
    evalPerMajor: "Exact-match accuracy by ISCO-08 major group",
    evalPerLanguage: "Exact-match accuracy by input language",
    evalMajorGroup: "Major Group",
    evalLanguage: "Language",
    evalInterpretHeader: "Interpretation",
    evalInterpret: "In this specific controlled configuration — the official ILO 2021 ISCO-08 profile, exact four-digit-code matching, no LLM reranking, full 18,747-case WISCO v2 heldout split — flat retrieval was substantially more accurate than strict hierarchical retrieval. This is a negative finding for hierarchical retrieval relative to flat retrieval in this one configuration; it is not evidence that the system as a whole underperforms, and does not generalise beyond this benchmark, catalogue profile, and reranking-off setting.",
    evalNotEstablishHeader: "What this does not establish",
    evalNotEstablish: [
      "Not real Labour Force Survey validation — WISCO is externally sourced reference data, not survey respondent data.",
      "Not an ISIC, ISCED, or Semantic Relation Engine evaluation — none of those components were exercised.",
      "Not an LLM reranking comparison — reranking was disabled in both arms.",
      "Not a cost, memory, throughput, scalability, or production-latency claim.",
    ],
    evalSource: "Source: OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md (WISCO v2, CC-BY-4.0, Zenodo DOI 10.5281/zenodo.8262593).",
    recommendationsSection: "Data Collector Recommendations",
    researchFooter: "ILO ICLS-19-aligned employment classification · UAE PDPL & GDPR Art. 15 · Multilingual Conversational AI LFS System",
    toggleAr: "عربي",
    toggleEn: "English",
    fields: {
      employment_status:    "Employment status",
      education_level:      "Education Attainment Level (ISCED 2011)",
      employment_nature:    "Employment nature",
      employment_sector:    "Sector",
      job_title:            "Occupation / Job title (ISCO-08)",
      industry:             "Establishment Economic Activity (ISIC Rev.4)",
      hours_per_week:       "Hours / week",
      employment_type:      "Employment type",
      monthly_wage_range:   "Monthly wage (AED)",
      job_search_active:    "Actively searching",
      available_for_work:   "Available for work",
      unemployment_duration:"Unemployed duration",
      last_job_title:       "Last job title",
      reason_left_job:      "Reason left job",
      outside_lf_reason:    "Outside LF reason",
      ai_preference:        "AI interviewer preference",
      data_confidence:      "Data privacy confidence",
    },
  },
  ar: {
    pageTitle: "تقرير المسح",
    loading: "جارٍ إعداد التقرير البحثي…",
    errorTitle: "التقرير غير متاح",
    backToHome: "العودة إلى الرئيسية",
    signOut: "تسجيل الخروج",
    sessionLabel: "الجلسة",
    reportId: "التقرير",
    generatedAt: "تاريخ الإنشاء",
    iscoSection: "تصنيف ISCO-08",
    iscoCode: "رمز المجموعة الوحدوية",
    iscoMajor: "المجموعة الرئيسية",
    iscoConfidence: "ثقة التصنيف",
    iscoMethod: "الطريقة",
    hitlRequired: "مراجعة بشرية مطلوبة",
    hitlOk: "تحقق الذكاء الاصطناعي",
    profileSection: "الملف الوظيفي",
    sectionB: "القسم ب — الديموغرافيا والتعليم",
    sectionC: "القسم ج/د/ه — تفاصيل التوظيف",
    sectionFG: "القسم و/ز — البطالة / خارج سوق العمل",
    sectionK: "القسم ك — ملاحظات المستجيب",
    narrativeSection: "التقرير السردي (ذكاء اصطناعي)",
    qualitySection: "تقييم جودة البيانات",
    qualityScore: "نقاط الجودة",
    flaggedFields: "الحقول المُعلَّمة",
    qualityStatus: { pass: "اجتاز", fail: "يحتاج مراجعة", escalated: "محال للمراجعة", unknown: "لم يُقيَّم" },
    isicSection: "تصنيف ISIC Rev.4 للنشاط الاقتصادي",
    isicSection4: "القطاع",
    isicDivision4: "الفئة الرئيسية",
    isicGroup4: "المجموعة",
    isicClass4: "الفئة (4 أرقام)",
    isicConfidence: "ثقة التصنيف",
    isicNoData: "لا يوجد تصنيف ISIC متاح.",
    iscedSection: "مستوى ISCED 2011 + تخصص ISCED-F 2013",
    iscedLevel: "المستوى التعليمي",
    iscedBroad: "المجال الرئيسي (رقمان)",
    iscedNarrow: "المجال الفرعي (3 أرقام)",
    iscedDetailed: "التخصص الدقيق (4 أرقام)",
    iscedConfidence: "ثقة التصنيف",
    iscedNoData: "لا يوجد تصنيف ISCED متاح.",
    semanticSection: "التوافق المعياري (ISCO ↔ ISIC ↔ ISCED) — المساهمة الأساسية للأطروحة: محرك العلاقة الدلالية",
    semanticScore: "درجة التوافق",
    semanticCoherent: "متوافق",
    semanticIncoherent: "غير متوافق",
    semanticViolations: "الانتهاكات المكتشفة",
    semanticNoData: "لا يوجد تحليل توافق متاح.",
    evalSection: "مقارنة الاسترجاع — معيار WISCO v2 المُتحكَّم به",
    evalSubtitle: "تقييم مُتحكَّم به لدقة الرمز الكامل على معيار مرجعي متعدد اللغات لعناوين المهن مصدره خارجي (WISCO v2، 18,747 حالة اختبار محجوزة، كتالوج ISCO-08 الرسمي لمنظمة العمل الدولية 2021، مع تعطيل إعادة الترتيب بالذكاء الاصطناعي). هذا ليس تحققًا ميدانيًا لمسح القوى العاملة.",
    evalSystem: "الطريقة", evalCases: "عدد الحالات", evalCorrect: "تطابق تام", evalAccuracy: "الدقة", evalCi: "فاصل ويلسون 95%",
    evalDiff: "فرق الدقة (الهرمي − المسطح)",
    evalMcnemar: "قيمة McNemar الاحتمالية (ثنائية الاتجاه)",
    evalPerMajor: "دقة التطابق التام حسب المجموعة الرئيسية لـ ISCO-08",
    evalPerLanguage: "دقة التطابق التام حسب لغة الإدخال",
    evalMajorGroup: "المجموعة الرئيسية",
    evalLanguage: "اللغة",
    evalInterpretHeader: "التفسير",
    evalInterpret: "في هذا الإعداد المُتحكَّم به تحديدًا — كتالوج ISCO-08 الرسمي 2021، مطابقة الرمز الكامل من أربعة أرقام، دون إعادة ترتيب بالذكاء الاصطناعي، على كامل مجموعة WISCO v2 المحجوزة (18,747 حالة) — كان الاسترجاع المسطح أكثر دقة بشكل ملحوظ من الاسترجاع الهرمي الصارم. هذه نتيجة سلبية للاسترجاع الهرمي مقارنةً بالمسطح في هذا الإعداد فقط؛ ولا تعني أن النظام ككل أقل كفاءة، ولا تُعمَّم خارج نطاق هذا المعيار وكتالوجه وإعداد تعطيل إعادة الترتيب.",
    evalNotEstablishHeader: "ما لا يثبته هذا القياس",
    evalNotEstablish: [
      "ليس تحققًا حقيقيًا لمسح القوى العاملة — بيانات WISCO مرجعية مصدرها خارجي، وليست بيانات مستجيبين حقيقية.",
      "ليس تقييمًا لـ ISIC أو ISCED أو محرك العلاقة الدلالية — لم يُفعَّل أي من هذه المكونات.",
      "ليس مقارنة لإعادة الترتيب بالذكاء الاصطناعي — كانت معطّلة في كلا الجانبين.",
      "ليس ادعاءً متعلقًا بالتكلفة أو الذاكرة أو الإنتاجية أو قابلية التوسع أو زمن الاستجابة الإنتاجي.",
    ],
    evalSource: "المصدر: OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md (معيار WISCO v2، رخصة CC-BY-4.0، معرّف Zenodo الرقمي 10.5281/zenodo.8262593).",
    recommendationsSection: "توصيات جامع البيانات",
    researchFooter: "تصنيف التوظيف متوافق مع مبادئ ILO ICLS-19 · PDPL الإماراتي و GDPR المادة 15 · نظام مسح قوة العمل بالذكاء الاصطناعي",
    toggleAr: "عربي",
    toggleEn: "English",
    fields: {
      employment_status:    "حالة التوظيف",
      education_level:      "مستوى التعليم (ISCED 2011)",
      employment_nature:    "طبيعة التوظيف",
      employment_sector:    "القطاع",
      job_title:            "المهنة / المسمى الوظيفي (ISCO-08)",
      industry:             "النشاط الاقتصادي للمنشأة (ISIC Rev.4)",
      hours_per_week:       "ساعات / أسبوع",
      employment_type:      "نوع التوظيف",
      monthly_wage_range:   "الراتب الشهري (AED)",
      job_search_active:    "بحث نشط عن عمل",
      available_for_work:   "متاح للعمل",
      unemployment_duration:"مدة البطالة",
      last_job_title:       "آخر مسمى وظيفي",
      reason_left_job:      "سبب ترك العمل",
      outside_lf_reason:    "سبب الخروج من سوق العمل",
      ai_preference:        "تفضيل المقابلة بالذكاء الاصطناعي",
      data_confidence:      "الثقة في خصوصية البيانات",
    },
  },
  // Urdu/Hindi/Tagalog UI chrome, added 2026-08-29 — machine-drafted, not yet
  // reviewed by a native speaker. Note: the AI-generated narrative report
  // itself (report_en/report_ar, recommendations_en/recommendations_ar) is
  // only ever produced in English or Arabic by the backend's
  // ReportGenerator — that is a real, separate, backend-side limitation this
  // frontend translation does not and cannot close. The reportLang toggle
  // below stays English/Arabic-only for that reason; only the surrounding
  // page chrome (section headers, field labels) is translated here.
  ur: {
    pageTitle: "سروے کی رپورٹ",
    loading: "تحقیقی رپورٹ تیار کی جا رہی ہے…",
    errorTitle: "رپورٹ دستیاب نہیں",
    backToHome: "ہوم پیج پر واپس جائیں",
    signOut: "سائن آؤٹ",
    sessionLabel: "سیشن",
    reportId: "رپورٹ",
    generatedAt: "تیار کی گئی",
    iscoSection: "ISCO-08 درجہ بندی",
    iscoCode: "یونٹ گروپ کوڈ",
    iscoMajor: "بڑا گروپ",
    iscoConfidence: "درجہ بندی کا اعتماد",
    iscoMethod: "طریقہ",
    hitlRequired: "انسانی جائزہ درکار",
    hitlOk: "AI کی تصدیق شدہ",
    profileSection: "روزگار پروفائل",
    sectionB: "سیکشن بی — آبادیات اور تعلیم",
    sectionC: "سیکشن سی/ڈی/ای — روزگار کی تفصیلات",
    sectionFG: "سیکشن ایف/جی — بے روزگاری / لیبر فورس سے باہر",
    sectionK: "سیکشن کے — جواب دہندہ کی رائے",
    narrativeSection: "AI کی تیار کردہ تفصیلی رپورٹ",
    qualitySection: "ڈیٹا کوالٹی کا جائزہ",
    qualityScore: "کوالٹی سکور",
    flaggedFields: "نشان زدہ فیلڈز",
    qualityStatus: { pass: "منظور", fail: "جائزہ درکار", escalated: "بھیج دیا گیا", unknown: "جانچا نہیں گیا" },
    isicSection: "ISIC Rev.4 صنعتی درجہ بندی",
    isicSection4: "سیکشن",
    isicDivision4: "ڈویژن",
    isicGroup4: "گروپ",
    isicClass4: "کلاس (4 ہندسے)",
    isicConfidence: "درجہ بندی کا اعتماد",
    isicNoData: "کوئی ISIC درجہ بندی دستیاب نہیں۔",
    iscedSection: "ISCED 2011 سطح + ISCED-F 2013 خصوصی شعبہ",
    iscedLevel: "تعلیمی سطح",
    iscedBroad: "وسیع شعبہ (2 ہندسے)",
    iscedNarrow: "محدود شعبہ (3 ہندسے)",
    iscedDetailed: "تفصیلی شعبہ (4 ہندسے)",
    iscedConfidence: "درجہ بندی کا اعتماد",
    iscedNoData: "کوئی ISCED درجہ بندی دستیاب نہیں۔",
    semanticSection: "کراس-اسٹینڈرڈ مطابقت (ISCO ↔ ISIC ↔ ISCED) — مقالے کی بنیادی شراکت: سیمینٹک ریلیشن انجن",
    semanticScore: "مطابقت کا سکور",
    semanticCoherent: "مطابقت پذیر",
    semanticIncoherent: "غیر مطابقت پذیر",
    semanticViolations: "پائی جانے والی خلاف ورزیاں",
    semanticNoData: "کوئی کراس-اسٹینڈرڈ تجزیہ دستیاب نہیں۔",
    evalSection: "ریٹریول موازنہ — WISCO v2 کنٹرولڈ بینچ مارک",
    evalSubtitle: "ایک بیرونی ماخذ سے حاصل کردہ کثیر لسانی پیشہ وارانہ عنوان کے حوالہ بینچ مارک (WISCO v2، 18,747 مخصوص ٹیسٹ کیسز، سرکاری ILO 2021 ISCO-08 کیٹلاگ، AI ری رینکنگ غیر فعال) پر مکمل کوڈ کی درست کنٹرولڈ تشخیص۔ یہ لیبر فورس سروے کی فیلڈ توثیق نہیں ہے۔",
    evalSystem: "طریقہ", evalCases: "کیسز", evalCorrect: "مکمل مماثلت", evalAccuracy: "درستگی", evalCi: "95% ولسن CI",
    evalDiff: "درستگی کا فرق (درجہ بندی شدہ − مسطح)",
    evalMcnemar: "McNemar کی درست دو طرفہ p-value",
    evalPerMajor: "ISCO-08 کے بڑے گروپ کے مطابق مکمل مماثلت کی درستگی",
    evalPerLanguage: "ان پٹ زبان کے مطابق مکمل مماثلت کی درستگی",
    evalMajorGroup: "بڑا گروپ",
    evalLanguage: "زبان",
    evalInterpretHeader: "تشریح",
    evalInterpret: "اس مخصوص کنٹرولڈ ترتیب میں — سرکاری ILO 2021 ISCO-08 پروفائل، مکمل چار ہندسوں کے کوڈ کی مماثلت، بغیر AI ری رینکنگ کے، مکمل 18,747 کیسز کے WISCO v2 ہولڈ آؤٹ اسپلٹ پر — فلیٹ ریٹریول سخت درجہ بندی شدہ ریٹریول سے کافی زیادہ درست تھا۔ یہ صرف اس ایک ترتیب میں درجہ بندی شدہ ریٹریول کے لیے ایک منفی نتیجہ ہے؛ اس کا مطلب یہ نہیں کہ مجموعی نظام کم کارکردگی دکھاتا ہے، اور یہ اس بینچ مارک، کیٹلاگ پروفائل، اور ری رینکنگ آف سیٹنگ سے آگے عام نہیں ہوتا۔",
    evalNotEstablishHeader: "یہ کیا ثابت نہیں کرتا",
    evalNotEstablish: [
      "حقیقی لیبر فورس سروے کی توثیق نہیں — WISCO بیرونی ماخذ کا حوالہ ڈیٹا ہے، حقیقی جواب دہندگان کا ڈیٹا نہیں۔",
      "ISIC، ISCED، یا سیمینٹک ریلیشن انجن کی تشخیص نہیں — ان میں سے کسی بھی جزو کو استعمال نہیں کیا گیا۔",
      "AI ری رینکنگ کا موازنہ نہیں — دونوں طرف ری رینکنگ غیر فعال تھی۔",
      "لاگت، میموری، تھرو پٹ، اسکیل ایبلٹی، یا پروڈکشن لیٹنسی کا کوئی دعویٰ نہیں۔",
    ],
    evalSource: "ماخذ: OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md (WISCO v2، CC-BY-4.0 لائسنس، Zenodo DOI 10.5281/zenodo.8262593)۔",
    recommendationsSection: "ڈیٹا اکٹھا کرنے والے کے لیے سفارشات",
    researchFooter: "ILO ICLS-19 کے مطابق روزگار کی درجہ بندی · UAE PDPL اور GDPR آرٹیکل 15 · کثیر لسانی گفتگو پر مبنی AI LFS سسٹم",
    toggleAr: "عربي",
    toggleEn: "English",
    fields: {
      employment_status:    "روزگار کی حیثیت",
      education_level:      "تعلیمی حصولیابی کی سطح (ISCED 2011)",
      employment_nature:    "روزگار کی نوعیت",
      employment_sector:    "شعبہ",
      job_title:            "پیشہ / جاب ٹائٹل (ISCO-08)",
      industry:             "ادارے کی اقتصادی سرگرمی (ISIC Rev.4)",
      hours_per_week:       "گھنٹے / ہفتہ",
      employment_type:      "روزگار کی قسم",
      monthly_wage_range:   "ماہانہ تنخواہ (AED)",
      job_search_active:    "فعال طور پر تلاش",
      available_for_work:   "کام کے لیے دستیاب",
      unemployment_duration:"بے روزگاری کی مدت",
      last_job_title:       "آخری جاب ٹائٹل",
      reason_left_job:      "نوکری چھوڑنے کی وجہ",
      outside_lf_reason:    "لیبر فورس سے باہر ہونے کی وجہ",
      ai_preference:        "AI انٹرویو لینے والے کی ترجیح",
      data_confidence:      "ڈیٹا کی رازداری پر اعتماد",
    },
  },
  hi: {
    pageTitle: "सर्वेक्षण रिपोर्ट",
    loading: "शोध रिपोर्ट तैयार की जा रही है…",
    errorTitle: "रिपोर्ट अनुपलब्ध",
    backToHome: "होम पर वापस जाएं",
    signOut: "साइन आउट",
    sessionLabel: "सत्र",
    reportId: "रिपोर्ट",
    generatedAt: "तैयार की गई",
    iscoSection: "ISCO-08 वर्गीकरण",
    iscoCode: "यूनिट ग्रुप कोड",
    iscoMajor: "प्रमुख समूह",
    iscoConfidence: "वर्गीकरण विश्वास",
    iscoMethod: "विधि",
    hitlRequired: "मानव समीक्षा आवश्यक",
    hitlOk: "AI सत्यापित",
    profileSection: "रोजगार प्रोफ़ाइल",
    sectionB: "खंड बी — जनसांख्यिकी और शिक्षा",
    sectionC: "खंड सी/डी/ई — रोजगार विवरण",
    sectionFG: "खंड एफ/जी — बेरोजगारी / श्रम शक्ति से बाहर",
    sectionK: "खंड के — उत्तरदाता प्रतिक्रिया",
    narrativeSection: "AI-जनित विवरणात्मक रिपोर्ट",
    qualitySection: "डेटा गुणवत्ता मूल्यांकन",
    qualityScore: "गुणवत्ता स्कोर",
    flaggedFields: "चिह्नित फ़ील्ड",
    qualityStatus: { pass: "उत्तीर्ण", fail: "समीक्षा आवश्यक", escalated: "आगे भेजा गया", unknown: "मूल्यांकित नहीं" },
    isicSection: "ISIC Rev.4 उद्योग वर्गीकरण",
    isicSection4: "सेक्शन",
    isicDivision4: "डिवीजन",
    isicGroup4: "ग्रुप",
    isicClass4: "क्लास (4-अंक)",
    isicConfidence: "वर्गीकरण विश्वास",
    isicNoData: "कोई ISIC वर्गीकरण उपलब्ध नहीं है।",
    iscedSection: "ISCED 2011 स्तर + ISCED-F 2013 विशेषज्ञता क्षेत्र",
    iscedLevel: "शिक्षा स्तर",
    iscedBroad: "व्यापक क्षेत्र (2-अंक)",
    iscedNarrow: "संकीर्ण क्षेत्र (3-अंक)",
    iscedDetailed: "विस्तृत क्षेत्र (4-अंक)",
    iscedConfidence: "वर्गीकरण विश्वास",
    iscedNoData: "कोई ISCED वर्गीकरण उपलब्ध नहीं है।",
    semanticSection: "क्रॉस-स्टैंडर्ड सुसंगति (ISCO ↔ ISIC ↔ ISCED) — थीसिस का मुख्य योगदान: सिमेंटिक रिलेशन इंजन",
    semanticScore: "सुसंगति स्कोर",
    semanticCoherent: "सुसंगत",
    semanticIncoherent: "असंगत",
    semanticViolations: "पाई गई त्रुटियां",
    semanticNoData: "कोई क्रॉस-स्टैंडर्ड विश्लेषण उपलब्ध नहीं है।",
    evalSection: "पुनर्प्राप्ति तुलना — WISCO v2 नियंत्रित बेंचमार्क",
    evalSubtitle: "एक बाहरी स्रोत से प्राप्त बहुभाषी व्यवसाय-शीर्षक संदर्भ बेंचमार्क (WISCO v2, 18,747 होल्ड-आउट मामले, आधिकारिक ILO 2021 ISCO-08 कैटलॉग, LLM रीरैंकिंग अक्षम) पर नियंत्रित सटीक-कोड मूल्यांकन। यह श्रम बल सर्वेक्षण क्षेत्र सत्यापन नहीं है।",
    evalSystem: "विधि", evalCases: "मामले", evalCorrect: "सटीक मिलान", evalAccuracy: "सटीकता", evalCi: "95% विल्सन CI",
    evalDiff: "सटीकता अंतर (पदानुक्रमित − फ्लैट)",
    evalMcnemar: "McNemar सटीक द्विपक्षीय p-मान",
    evalPerMajor: "ISCO-08 प्रमुख समूह द्वारा सटीक-मिलान सटीकता",
    evalPerLanguage: "इनपुट भाषा द्वारा सटीक-मिलान सटीकता",
    evalMajorGroup: "प्रमुख समूह",
    evalLanguage: "भाषा",
    evalInterpretHeader: "व्याख्या",
    evalInterpret: "इस विशिष्ट नियंत्रित विन्यास में — आधिकारिक ILO 2021 ISCO-08 प्रोफ़ाइल, सटीक चार-अंकीय कोड मिलान, बिना LLM रीरैंकिंग के, पूर्ण 18,747-मामले WISCO v2 होल्डआउट स्प्लिट पर — फ्लैट पुनर्प्राप्ति सख्त पदानुक्रमित पुनर्प्राप्ति की तुलना में काफी अधिक सटीक थी। यह केवल इस एक विन्यास में पदानुक्रमित पुनर्प्राप्ति के लिए एक नकारात्मक निष्कर्ष है; इसका मतलब यह नहीं है कि समग्र प्रणाली कम प्रदर्शन करती है, और यह इस बेंचमार्क, कैटलॉग प्रोफ़ाइल, और रीरैंकिंग-बंद सेटिंग से आगे सामान्यीकृत नहीं होता।",
    evalNotEstablishHeader: "यह क्या स्थापित नहीं करता",
    evalNotEstablish: [
      "वास्तविक श्रम बल सर्वेक्षण सत्यापन नहीं — WISCO बाहरी स्रोत से प्राप्त संदर्भ डेटा है, वास्तविक उत्तरदाता डेटा नहीं।",
      "ISIC, ISCED, या सिमेंटिक रिलेशन इंजन का मूल्यांकन नहीं — इनमें से किसी भी घटक का उपयोग नहीं किया गया।",
      "LLM रीरैंकिंग तुलना नहीं — दोनों पक्षों में रीरैंकिंग अक्षम थी।",
      "लागत, मेमोरी, थ्रूपुट, स्केलेबिलिटी, या उत्पादन-विलंबता का कोई दावा नहीं।",
    ],
    evalSource: "स्रोत: OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md (WISCO v2, CC-BY-4.0, Zenodo DOI 10.5281/zenodo.8262593)।",
    recommendationsSection: "डेटा संग्रहकर्ता के लिए सिफारिशें",
    researchFooter: "ILO ICLS-19-संरेखित रोजगार वर्गीकरण · UAE PDPL और GDPR अनुच्छेद 15 · बहुभाषी संवादात्मक AI LFS प्रणाली",
    toggleAr: "عربي",
    toggleEn: "English",
    fields: {
      employment_status:    "रोजगार स्थिति",
      education_level:      "शिक्षा प्राप्ति स्तर (ISCED 2011)",
      employment_nature:    "रोजगार प्रकृति",
      employment_sector:    "क्षेत्र",
      job_title:            "व्यवसाय / पद का नाम (ISCO-08)",
      industry:             "प्रतिष्ठान आर्थिक गतिविधि (ISIC Rev.4)",
      hours_per_week:       "घंटे / सप्ताह",
      employment_type:      "रोजगार प्रकार",
      monthly_wage_range:   "मासिक वेतन (AED)",
      job_search_active:    "सक्रिय रूप से खोज रहे हैं",
      available_for_work:   "काम के लिए उपलब्ध",
      unemployment_duration:"बेरोजगारी अवधि",
      last_job_title:       "पिछला पद",
      reason_left_job:      "नौकरी छोड़ने का कारण",
      outside_lf_reason:    "श्रम शक्ति से बाहर होने का कारण",
      ai_preference:        "AI साक्षात्कारकर्ता प्राथमिकता",
      data_confidence:      "डेटा गोपनीयता में विश्वास",
    },
  },
  tl: {
    pageTitle: "Survey Report",
    loading: "Ginagawa ang research report…",
    errorTitle: "Hindi available ang report",
    backToHome: "Bumalik sa home",
    signOut: "Mag-sign out",
    sessionLabel: "Session",
    reportId: "Report",
    generatedAt: "Ginawa noong",
    iscoSection: "ISCO-08 Classification",
    iscoCode: "Unit Group Code",
    iscoMajor: "Major Group",
    iscoConfidence: "Kumpiyansa ng Classification",
    iscoMethod: "Paraan",
    hitlRequired: "Kailangan ng Human Review",
    hitlOk: "Na-verify ng AI",
    profileSection: "Profile ng Trabaho",
    sectionB: "Seksyon B — Demograpiko at Edukasyon",
    sectionC: "Seksyon C/D/E — Mga Detalye ng Trabaho",
    sectionFG: "Seksyon F/G — Kawalan ng Trabaho / Wala sa Labor Force",
    sectionK: "Seksyon K — Feedback ng Respondent",
    narrativeSection: "AI-Generated na Narrative Report",
    qualitySection: "Pagtatasa ng Kalidad ng Datos",
    qualityScore: "Quality Score",
    flaggedFields: "Mga Na-flag na Field",
    qualityStatus: { pass: "Pumasa", fail: "Kailangan ng Review", escalated: "Na-escalate", unknown: "Hindi Natasa" },
    isicSection: "ISIC Rev.4 Industry Classification",
    isicSection4: "Section",
    isicDivision4: "Division",
    isicGroup4: "Group",
    isicClass4: "Class (4-digit)",
    isicConfidence: "Kumpiyansa ng Classification",
    isicNoData: "Walang available na ISIC classification.",
    iscedSection: "ISCED 2011 Level + ISCED-F 2013 Field of Specialisation",
    iscedLevel: "Antas ng Edukasyon",
    iscedBroad: "Broad Field (2-digit)",
    iscedNarrow: "Narrow Field (3-digit)",
    iscedDetailed: "Detailed Field (4-digit)",
    iscedConfidence: "Kumpiyansa ng Classification",
    iscedNoData: "Walang available na ISCED classification.",
    semanticSection: "Cross-Standard Coherence (ISCO ↔ ISIC ↔ ISCED) — Pangunahing Kontribusyon ng Thesis: ang Semantic Relation Engine",
    semanticScore: "Coherence Score",
    semanticCoherent: "MAAYOS (COHERENT)",
    semanticIncoherent: "HINDI MAAYOS (INCONSISTENT)",
    semanticViolations: "Mga Nakitang Paglabag",
    semanticNoData: "Walang available na cross-standard analysis.",
    evalSection: "Paghahambing ng Retrieval — WISCO v2 Controlled Benchmark",
    evalSubtitle: "Kontroladong eksaktong-code na pagtatasa sa isang panlabas na multilingguwal na occupation-title reference benchmark (WISCO v2, 18,747 held-out na kaso, opisyal na ILO 2021 ISCO-08 catalogue, naka-disable ang LLM reranking). Hindi ito field validation ng Labour Force Survey.",
    evalSystem: "Paraan", evalCases: "Mga Kaso", evalCorrect: "Eksaktong tugma", evalAccuracy: "Katumpakan", evalCi: "95% Wilson CI",
    evalDiff: "Pagkakaiba sa katumpakan (hierarchical − flat)",
    evalMcnemar: "McNemar exact two-sided p-value",
    evalPerMajor: "Katumpakan ng eksaktong tugma ayon sa ISCO-08 major group",
    evalPerLanguage: "Katumpakan ng eksaktong tugma ayon sa wika ng input",
    evalMajorGroup: "Major Group",
    evalLanguage: "Wika",
    evalInterpretHeader: "Interpretasyon",
    evalInterpret: "Sa partikular na kontroladong konpigurasyong ito — ang opisyal na ILO 2021 ISCO-08 profile, eksaktong apat-digit na code matching, walang LLM reranking, buong 18,747-kaso na WISCO v2 heldout split — mas malaki ang katumpakan ng flat retrieval kaysa sa mahigpit na hierarchical retrieval. Isa itong negatibong natuklasan para sa hierarchical retrieval kumpara sa flat retrieval sa konpigurasyong ito lamang; hindi ito nangangahulugan na mas mababa ang performance ng buong sistema, at hindi ito nag-a-apply sa labas ng benchmark, catalogue profile, at reranking-off na setting na ito.",
    evalNotEstablishHeader: "Ano ang Hindi Nito Pinapatunayan",
    evalNotEstablish: [
      "Hindi tunay na Labour Force Survey validation — ang WISCO ay panlabas na reference data, hindi datos mula sa tunay na respondent.",
      "Hindi ebalwasyon ng ISIC, ISCED, o Semantic Relation Engine — wala sa mga bahaging ito ang na-exercise.",
      "Hindi paghahambing ng LLM reranking — naka-disable ang reranking sa magkabilang panig.",
      "Walang claim tungkol sa gastos, memorya, throughput, scalability, o production-latency.",
    ],
    evalSource: "Pinagmulan: OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md (WISCO v2, CC-BY-4.0, Zenodo DOI 10.5281/zenodo.8262593).",
    recommendationsSection: "Mga Rekomendasyon para sa Data Collector",
    researchFooter: "ILO ICLS-19-aligned na employment classification · UAE PDPL at GDPR Art. 15 · Multilingual Conversational AI LFS System",
    toggleAr: "عربي",
    toggleEn: "English",
    fields: {
      employment_status:    "Katayuan sa trabaho",
      education_level:      "Antas ng Natapos na Edukasyon (ISCED 2011)",
      employment_nature:    "Kalikasan ng trabaho",
      employment_sector:    "Sektor",
      job_title:            "Trabaho / Job title (ISCO-08)",
      industry:             "Economic Activity ng Establishment (ISIC Rev.4)",
      hours_per_week:       "Oras / linggo",
      employment_type:      "Uri ng trabaho",
      monthly_wage_range:   "Buwanang sahod (AED)",
      job_search_active:    "Aktibong naghahanap",
      available_for_work:   "Available para sa trabaho",
      unemployment_duration:"Tagal ng kawalan ng trabaho",
      last_job_title:       "Huling job title",
      reason_left_job:      "Dahilan ng pag-alis sa trabaho",
      outside_lf_reason:    "Dahilan ng pagiging wala sa labor force",
      ai_preference:        "Kagustuhan sa AI interviewer",
      data_confidence:      "Kumpiyansa sa privacy ng data",
    },
  },
};

// ── Small reusable components ──────────────────────────────────────────────────

function ProfileRow({ label, value }) {
  if (!value) return null;
  return (
    <div className="flex items-start gap-3 py-2 border-b border-gray-100 last:border-0">
      <dt className="w-44 flex-shrink-0 text-xs text-gray-500 pt-0.5">{label}</dt>
      <dd className="text-sm text-gray-800 font-medium capitalize">{String(value).replace(/_/g, " ")}</dd>
    </div>
  );
}

function SectionBlock({ title, children }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
      <div className="px-4 py-2.5 border-b border-gray-200 bg-gray-50">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">{title}</h3>
      </div>
      <div className="px-4 py-3">{children}</div>
    </div>
  );
}

function ConfidenceBar({ value, label }) {
  const pct = Math.round((value || 0) * 100);
  const color = pct >= 80 ? "bg-emerald-500" : pct >= 70 ? "bg-amber-500" : "bg-red-500";
  const textColor = pct >= 80 ? "text-emerald-600" : pct >= 70 ? "text-amber-600" : "text-red-600";
  return (
    <div className="space-y-1">
      {label && <p className="text-xs text-gray-500">{label}</p>}
      <div className="flex items-center gap-3">
        <div className="flex-1 bg-gray-200 rounded-full h-2">
          <div className={`${color} h-2 rounded-full transition-all`} style={{ width: `${pct}%` }} />
        </div>
        <span className={`text-sm font-bold tabular-nums ${textColor}`}>{pct}%</span>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function ReportPage() {
  const router = useRouter();
  const { session: sessionId } = router.query;

  const [lang, setLang]             = useState("en");
  const [reportLang, setReportLang] = useState("en");
  const [report, setReport]         = useState(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState("");

  const t   = T[lang] || T.en;
  const dir = (lang === "ar" || lang === "ur") ? "rtl" : "ltr";

  useEffect(() => {
    if (!sessionId) return;
    const token      = localStorage.getItem("lfs_token");
    const storedLang = localStorage.getItem("lfs_lang") || "en";
    setLang(storedLang);
    setReportLang(storedLang);
    if (!token) { router.replace("/"); return; }
    getReport(token, sessionId)
      .then((data) => { setReport(data); setLoading(false); })
      .catch((err) => { setError(err.message || "Failed to load report."); setLoading(false); });
  }, [sessionId, router]);

  function handleSignOut() {
    localStorage.removeItem("lfs_token");
    localStorage.removeItem("lfs_lang");
    router.push("/");
  }

  // ── Derived values ──────────────────────────────────────────────────────────
  const p         = report?.profile || {};
  const qKey      = report?.quality_status || "unknown";
  const qStyles   = QUALITY_STYLES[qKey] || QUALITY_STYLES.unknown;
  const qPct      = report?.quality_score != null ? Math.round(report.quality_score * 100) : null;
  const majorCode = p.isco_code ? p.isco_code[0] : null;
  const majorInfo = majorCode ? ISCO_MAJOR[majorCode] : null;
  const narrative = reportLang === "ar" ? report?.report_ar    : report?.report_en;
  const recText   = reportLang === "ar" ? report?.recommendations_ar : report?.recommendations_en;
  const isAr      = lang === "ar";

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <>
      <Head>
        <title>{t.pageTitle}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div dir={dir} className="min-h-screen bg-gray-50 text-gray-900">

        {/* ── Header ─────────────────────────────────────────────────────────── */}
        <header className="border-b border-gray-200 px-6 py-3 flex items-center justify-between sticky top-0 bg-white/90 backdrop-blur z-10 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-sm font-bold text-white">
              LFS
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-900">{t.pageTitle}</p>
              {sessionId && (
                <p className="text-[10px] text-gray-400">
                  {t.sessionLabel} #{sessionId}
                  {report?.report_id && <span className="ml-2">· {t.reportId} #{report.report_id}</span>}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <LanguageToggle lang={lang} onToggle={setLang} />
            <button onClick={handleSignOut}
              className="text-xs text-gray-500 hover:text-gray-900 border border-gray-300 rounded-lg px-3 py-1.5 transition-colors bg-white hover:bg-gray-50">
              {t.signOut}
            </button>
          </div>
        </header>

        {/* ── Body ───────────────────────────────────────────────────────────── */}
        <main className="max-w-3xl mx-auto px-4 py-8 space-y-5">

          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-32 gap-4 text-gray-400">
              <svg className="animate-spin h-8 w-8 text-blue-500" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              <p className="text-sm">{t.loading}</p>
            </div>
          )}

          {/* Error */}
          {!loading && error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl">
              <p className="font-semibold">{t.errorTitle}</p>
              <p className="text-sm mt-1 text-red-500">{error}</p>
              <button onClick={() => router.push("/")}
                className="mt-3 text-sm underline text-red-500 hover:text-red-700">
                {t.backToHome}
              </button>
            </div>
          )}

          {/* ── Report content ─────────────────────────────────────────────── */}
          {!loading && report && (
            <>
              {/* ── Meta bar ─────────────────────────────────────────────── */}
              <div className="flex flex-wrap items-center justify-between gap-3">
                <span className={`inline-flex items-center gap-2 text-xs font-medium border px-3 py-1.5 rounded-full ${qStyles.badge}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${qStyles.dot}`} />
                  {t.qualityStatus[qKey]}
                  {qPct != null && <span className="opacity-70">· {qPct}%</span>}
                </span>
                <span className="text-xs text-gray-400">
                  {t.generatedAt}: {new Date(report.generated_at).toLocaleString()}
                </span>
              </div>

              {/* ── 1. ISCO-08 Classification ─────────────────────────────── */}
              <div className="bg-white border border-violet-200 rounded-xl overflow-hidden shadow-sm">
                <div className="px-4 py-2.5 border-b border-violet-100 bg-violet-50 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <h3 className="text-xs font-semibold text-violet-700 uppercase tracking-widest">
                      {t.iscoSection}
                    </h3>
                    <span className="text-[10px] bg-violet-100 text-violet-700 border border-violet-300 px-1.5 py-0.5 rounded-full font-medium">
                      Best-Tested Result: 40.95%
                    </span>
                  </div>
                  <span className="text-[10px] text-gray-400 font-mono">ILO ISCO-08</span>
                </div>

                <div className="px-4 py-4 space-y-4">
                  {p.isco_code ? (
                    <>
                      {/* Code + major group */}
                      <div className="flex items-start gap-4 flex-wrap">
                        <div className="text-center">
                          <p className="text-[10px] text-gray-500 mb-1">{t.iscoCode}</p>
                          <div className="text-4xl font-bold font-mono text-gray-900 bg-gray-100 border border-gray-200 rounded-xl px-5 py-3">
                            {p.isco_code}
                          </div>
                        </div>
                        {majorInfo && (
                          <div className="flex-1 min-w-[160px]">
                            <p className="text-[10px] text-gray-500 mb-1">{t.iscoMajor} ({majorCode})</p>
                            <span className={`inline-block text-sm font-semibold border px-3 py-1.5 rounded-lg ${majorInfo.color}`}>
                              {majorInfo.label}
                            </span>
                          </div>
                        )}
                        {/* HITL flag */}
                        <div className="flex-shrink-0 self-center">
                          {p.isco_confidence != null && p.isco_confidence < 0.70 ? (
                            <span className="inline-flex items-center gap-1.5 text-xs font-medium bg-amber-50 border border-amber-300 text-amber-700 px-2.5 py-1.5 rounded-lg">
                              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                              {t.hitlRequired}
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1.5 text-xs font-medium bg-emerald-50 border border-emerald-300 text-emerald-700 px-2.5 py-1.5 rounded-lg">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                              {t.hitlOk}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Confidence bar */}
                      {p.isco_confidence != null && (
                        <ConfidenceBar value={p.isco_confidence} label={t.iscoConfidence} />
                      )}

                      {/* Hierarchy path */}
                      {p.isco_code && (
                        <div>
                          <p className="text-[10px] text-gray-500 mb-2">4-Stage Hierarchical Pipeline</p>
                          <div className="flex items-center gap-1 flex-wrap">
                            {[
                              { stage: "Major",     code: p.isco_code[0],         cols: "bg-blue-50    border-blue-200   text-blue-700"   },
                              { stage: "Sub-major", code: p.isco_code.slice(0,2),  cols: "bg-violet-50  border-violet-200  text-violet-700" },
                              { stage: "Minor",     code: p.isco_code.slice(0,3),  cols: "bg-cyan-50    border-cyan-200    text-cyan-700"   },
                              { stage: "Unit",      code: p.isco_code,             cols: "bg-emerald-50 border-emerald-200 text-emerald-700"},
                            ].map((s, i) => (
                              <div key={i} className="flex items-center gap-1">
                                <div className={`border rounded-lg px-2.5 py-1.5 text-center ${s.cols}`}>
                                  <p className="text-[9px] opacity-60 uppercase tracking-wide">{s.stage}</p>
                                  <p className="text-sm font-bold font-mono">{s.code}</p>
                                </div>
                                {i < 3 && <span className="text-gray-400 text-xs">›</span>}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="text-sm text-gray-400 italic">No ISCO classification available for this session.</p>
                  )}
                </div>
              </div>

              {/* ── 2. ISIC Rev.4 Industry Classification ────────────────── */}
              <SectionBlock title={t.isicSection}>
                {report.isic_classification ? (() => {
                  const ic = report.isic_classification;
                  const pct = Math.round((ic.confidence || 0) * 100);
                  return (
                    <div className="space-y-3">
                      <div className="flex items-center gap-1 flex-wrap text-xs">
                        <div className="bg-orange-50 border border-orange-200 rounded-lg px-2.5 py-1.5 text-center">
                          <p className="text-[9px] text-orange-500 uppercase tracking-wide mb-0.5">{t.isicSection4}</p>
                          <p className="font-bold font-mono text-orange-700">{ic.section}</p>
                        </div>
                        <span className="text-gray-400">›</span>
                        <div className="bg-amber-50 border border-amber-200 rounded-lg px-2.5 py-1.5 text-center">
                          <p className="text-[9px] text-amber-500 uppercase tracking-wide mb-0.5">{t.isicDivision4}</p>
                          <p className="font-bold font-mono text-amber-700">{ic.division_code}</p>
                        </div>
                        <span className="text-gray-400">›</span>
                        <div className="bg-yellow-50 border border-yellow-200 rounded-lg px-2.5 py-1.5 text-center">
                          <p className="text-[9px] text-yellow-600 uppercase tracking-wide mb-0.5">{t.isicGroup4}</p>
                          <p className="font-bold font-mono text-yellow-700">{ic.group_code}</p>
                        </div>
                        <span className="text-gray-400">›</span>
                        <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-1.5 text-center">
                          <p className="text-[9px] text-emerald-600 uppercase tracking-wide mb-0.5">{t.isicClass4}</p>
                          <p className="font-bold font-mono text-emerald-700 text-sm">{ic.class_code}</p>
                        </div>
                      </div>
                      <div className="space-y-1 text-xs">
                        <div className="flex gap-2">
                          <span className="text-gray-500 w-28 shrink-0">{t.isicSection4}:</span>
                          <span className="text-orange-700">{ic.section} — {ic.section_title}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-gray-500 w-28 shrink-0">{t.isicDivision4}:</span>
                          <span className="text-amber-700">{ic.division_code} — {ic.division_title}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-gray-500 w-28 shrink-0">{t.isicGroup4}:</span>
                          <span className="text-yellow-700">{ic.group_code} — {ic.group_title}</span>
                        </div>
                        <div className="flex gap-2 font-semibold">
                          <span className="text-gray-600 w-28 shrink-0">{t.isicClass4}:</span>
                          <span className="text-emerald-700">{ic.class_code} — {ic.class_title}</span>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs text-gray-500 mb-1">
                          <span>{t.isicConfidence} ({ic.method})</span>
                          <span className="font-mono font-bold">{pct}%</span>
                        </div>
                        <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${pct >= 80 ? "bg-emerald-500" : pct >= 60 ? "bg-amber-500" : "bg-orange-500"}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })() : (
                  <p className="text-sm text-gray-400 italic">{t.isicNoData}</p>
                )}
              </SectionBlock>

              {/* ── 3. ISCED 2011 Level + ISCED-F 2013 Field ─────────────── */}
              <SectionBlock title={t.iscedSection}>
                {report.isced_classification ? (() => {
                  const ic = report.isced_classification;
                  const pct = Math.round((ic.confidence || 0) * 100);
                  return (
                    <div className="space-y-3">
                      <div className="flex items-center gap-1 flex-wrap text-xs">
                        <div className="bg-sky-50 border border-sky-200 rounded-lg px-2.5 py-1.5 text-center">
                          <p className="text-[9px] text-sky-500 uppercase tracking-wide mb-0.5">{t.iscedLevel}</p>
                          <p className="font-bold font-mono text-sky-700 text-sm">{ic.level}</p>
                        </div>
                        <span className="text-gray-500 text-base">+</span>
                        <div className="bg-indigo-50 border border-indigo-200 rounded-lg px-2.5 py-1.5 text-center">
                          <p className="text-[9px] text-indigo-500 uppercase tracking-wide mb-0.5">{t.iscedBroad}</p>
                          <p className="font-bold font-mono text-indigo-700">{ic.broad_code}</p>
                        </div>
                        <span className="text-gray-400">›</span>
                        <div className="bg-blue-50 border border-blue-200 rounded-lg px-2.5 py-1.5 text-center">
                          <p className="text-[9px] text-blue-500 uppercase tracking-wide mb-0.5">{t.iscedNarrow}</p>
                          <p className="font-bold font-mono text-blue-700">{ic.narrow_code}</p>
                        </div>
                        <span className="text-gray-400">›</span>
                        <div className="bg-cyan-50 border border-cyan-200 rounded-lg px-3 py-1.5 text-center">
                          <p className="text-[9px] text-cyan-600 uppercase tracking-wide mb-0.5">{t.iscedDetailed}</p>
                          <p className="font-bold font-mono text-cyan-700 text-sm">{ic.detailed_code}</p>
                        </div>
                      </div>
                      <div className="space-y-1 text-xs">
                        <div className="flex gap-2">
                          <span className="text-gray-500 w-36 shrink-0">{t.iscedLevel}:</span>
                          <span className="text-sky-700">{ic.level} — {ic.level_title}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-gray-500 w-36 shrink-0">{t.iscedBroad}:</span>
                          <span className="text-indigo-700">{ic.broad_code} — {ic.broad_title}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-gray-500 w-36 shrink-0">{t.iscedNarrow}:</span>
                          <span className="text-blue-700">{ic.narrow_code} — {ic.narrow_title}</span>
                        </div>
                        <div className="flex gap-2 font-semibold">
                          <span className="text-gray-600 w-36 shrink-0">{t.iscedDetailed}:</span>
                          <span className="text-cyan-700">{ic.detailed_code} — {ic.detailed_title}</span>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs text-gray-500 mb-1">
                          <span>{t.iscedConfidence} ({ic.method})</span>
                          <span className="font-mono font-bold">{pct}%</span>
                        </div>
                        <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${pct >= 80 ? "bg-cyan-500" : pct >= 60 ? "bg-blue-500" : "bg-indigo-500"}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })() : (
                  <p className="text-sm text-gray-400 italic">{t.iscedNoData}</p>
                )}
              </SectionBlock>

              {/* ── 4. Semantic Cross-Standard Coherence ─────────────────── */}
              <SectionBlock title={t.semanticSection}>
                {report.semantic_coherence ? (() => {
                  const sc = report.semantic_coherence;
                  const pct = Math.round((sc.score || 0) * 100);
                  const isCoherent = sc.is_coherent;
                  const barColor = pct >= 70 ? "bg-emerald-500" : pct >= 50 ? "bg-amber-500" : "bg-red-500";
                  const badgeColor = isCoherent
                    ? "bg-emerald-50 border border-emerald-300 text-emerald-700"
                    : "bg-red-50 border border-red-300 text-red-700";
                  return (
                    <div className="space-y-4">
                      <div className="flex items-center gap-4">
                        <div className="flex-1">
                          <div className="flex justify-between text-xs text-gray-500 mb-1">
                            <span>{t.semanticScore}</span>
                            <span className="font-mono font-bold">{pct}%</span>
                          </div>
                          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
                          </div>
                        </div>
                        <span className={`text-xs font-semibold px-2 py-1 rounded ${badgeColor}`}>
                          {isCoherent ? t.semanticCoherent : t.semanticIncoherent}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
                          <span className="text-gray-500">ISCO ↔ ISIC: </span>
                          <span className={sc.isco_isic_compatible ? "text-emerald-600 font-medium" : "text-red-600 font-medium"}>
                            {sc.isco_isic_compatible ? "✓" : "✗"} {sc.isic_label || sc.isic_section || "—"}
                          </span>
                        </div>
                        <div className="bg-gray-50 border border-gray-200 rounded px-3 py-2">
                          <span className="text-gray-500">ISCO ↔ ISCED: </span>
                          <span className={sc.isco_isced_compatible ? "text-emerald-600 font-medium" : "text-red-600 font-medium"}>
                            {sc.isco_isced_compatible ? "✓" : "✗"} {sc.isced_label || (sc.isced_level != null ? `Level ${sc.isced_level}` : "—")}
                          </span>
                        </div>
                      </div>
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {isAr ? sc.explanation_ar : sc.explanation_en}
                      </p>
                      {sc.violations && sc.violations.length > 0 && (
                        <div>
                          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">{t.semanticViolations}</p>
                          <ul className="space-y-1">
                            {sc.violations.map((v, i) => {
                              const sev = (v.severity || "").toUpperCase();
                              const sevColor = sev === "HIGH" ? "text-red-600" : sev === "MODERATE" ? "text-amber-600" : "text-gray-500";
                              return (
                                <li key={i} className="flex gap-2 text-xs bg-gray-50 border border-gray-200 rounded px-3 py-1.5">
                                  <span className={`font-bold shrink-0 ${sevColor}`}>[{sev}]</span>
                                  <span className="text-gray-700">{isAr ? v.message_ar : v.message_en}</span>
                                </li>
                              );
                            })}
                          </ul>
                        </div>
                      )}
                      {sc.confidence_adjustment !== 0 && (
                        <p className="text-xs text-gray-500">
                          ISCO confidence adjustment:&nbsp;
                          <span className={sc.confidence_adjustment > 0 ? "text-emerald-600 font-medium" : "text-red-600 font-medium"}>
                            {sc.confidence_adjustment > 0 ? "+" : ""}{(sc.confidence_adjustment * 100).toFixed(0)}%
                          </span>
                        </p>
                      )}
                    </div>
                  );
                })() : (
                  <p className="text-sm text-gray-400 italic">{t.semanticNoData}</p>
                )}
              </SectionBlock>

              {/* ── 5. Employment Profile ─────────────────────────────────── */}
              <SectionBlock title={t.profileSection}>
                <div className="space-y-4">
                  {(p.employment_status || p.education_level) && (
                    <div>
                      <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-2">{t.sectionB}</p>
                      <dl>
                        <ProfileRow label={t.fields.employment_status} value={p.employment_status} />
                        <ProfileRow label={t.fields.education_level}   value={p.education_level}   />
                      </dl>
                    </div>
                  )}
                  {p.employment_status === "employed" && (
                    <div>
                      <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-2">{t.sectionC}</p>
                      <dl>
                        <ProfileRow label={t.fields.employment_nature}  value={p.employment_nature}  />
                        <ProfileRow label={t.fields.employment_sector}  value={p.employment_sector}  />
                        <ProfileRow label={t.fields.job_title}          value={p.job_title}          />
                        <ProfileRow label={t.fields.industry}           value={p.industry}           />
                        <ProfileRow label={t.fields.hours_per_week}     value={p.hours_per_week}     />
                        <ProfileRow label={t.fields.employment_type}    value={p.employment_type}    />
                        <ProfileRow label={t.fields.monthly_wage_range} value={p.monthly_wage_range} />
                      </dl>
                    </div>
                  )}
                  {(p.job_search_active || p.available_for_work || p.unemployment_duration ||
                    p.last_job_title || p.outside_lf_reason) && (
                    <div>
                      <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-2">{t.sectionFG}</p>
                      <dl>
                        <ProfileRow label={t.fields.job_search_active}     value={p.job_search_active}     />
                        <ProfileRow label={t.fields.available_for_work}    value={p.available_for_work}    />
                        <ProfileRow label={t.fields.unemployment_duration} value={p.unemployment_duration} />
                        <ProfileRow label={t.fields.last_job_title}        value={p.last_job_title}        />
                        <ProfileRow label={t.fields.reason_left_job}       value={p.reason_left_job}       />
                        <ProfileRow label={t.fields.outside_lf_reason}     value={p.outside_lf_reason}     />
                      </dl>
                    </div>
                  )}
                  {(p.ai_preference || p.data_confidence) && (
                    <div>
                      <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-2">{t.sectionK}</p>
                      <dl>
                        <ProfileRow label={t.fields.ai_preference}  value={p.ai_preference}  />
                        <ProfileRow label={t.fields.data_confidence} value={p.data_confidence} />
                      </dl>
                    </div>
                  )}
                </div>
              </SectionBlock>

              {/* ── 6. AI Narrative Report ───────────────────────────────── */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <div className="px-4 py-2.5 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
                    {t.narrativeSection}
                  </h3>
                  <div className="flex items-center bg-gray-100 rounded-lg p-0.5 gap-0.5">
                    {["en","ar"].map((l) => (
                      <button key={l}
                        onClick={() => setReportLang(l)}
                        className={`text-xs px-2.5 py-1 rounded-md transition-colors font-medium ${
                          reportLang === l
                            ? "bg-blue-600 text-white"
                            : "text-gray-500 hover:text-gray-900"
                        }`}>
                        {l === "en" ? t.toggleEn : t.toggleAr}
                      </button>
                    ))}
                  </div>
                </div>
                <div
                  dir={reportLang === "ar" ? "rtl" : "ltr"}
                  className="px-4 py-4 text-sm text-gray-700 leading-relaxed whitespace-pre-wrap"
                >
                  {narrative || <span className="text-gray-400 italic">No narrative available.</span>}
                </div>
              </div>

              {/* ── 7. Data Quality Assessment ──────────────────────────── */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <div className="px-4 py-2.5 border-b border-gray-200 bg-gray-50">
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
                    {t.qualitySection}
                  </h3>
                </div>
                <div className="px-4 py-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 space-y-2">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider">{t.qualityScore}</p>
                      {qPct != null ? (
                        <>
                          <p className="text-2xl font-bold text-gray-900">{qPct}%</p>
                          <div className="bg-gray-200 rounded-full h-1.5">
                            <div className={`${qStyles.bar} h-1.5 rounded-full`} style={{ width: `${qPct}%` }} />
                          </div>
                        </>
                      ) : (
                        <p className="text-sm text-gray-400 italic">Not assessed</p>
                      )}
                    </div>
                    <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 space-y-2">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider">{t.flaggedFields}</p>
                      <p className={`text-2xl font-bold ${(report.flagged_count || 0) > 0 ? "text-amber-600" : "text-emerald-600"}`}>
                        {report.flagged_count || 0}
                      </p>
                      <span className={`inline-flex items-center gap-1.5 text-xs font-medium border px-2 py-0.5 rounded-full ${qStyles.badge}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${qStyles.dot}`} />
                        {t.qualityStatus[qKey]}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* ── 8. Recommendations ──────────────────────────────────── */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <div className="px-4 py-2.5 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
                    {t.recommendationsSection}
                  </h3>
                  <div className="flex items-center bg-gray-100 rounded-lg p-0.5 gap-0.5">
                    {["en","ar"].map((l) => (
                      <button key={l}
                        onClick={() => setReportLang(l)}
                        className={`text-xs px-2.5 py-1 rounded-md transition-colors font-medium ${
                          reportLang === l
                            ? "bg-blue-600 text-white"
                            : "text-gray-500 hover:text-gray-900"
                        }`}>
                        {l === "en" ? t.toggleEn : t.toggleAr}
                      </button>
                    ))}
                  </div>
                </div>
                <div
                  dir={reportLang === "ar" ? "rtl" : "ltr"}
                  className="px-4 py-4 text-sm text-gray-700 leading-relaxed whitespace-pre-wrap"
                >
                  {recText || <span className="text-gray-400 italic">No recommendations recorded.</span>}
                </div>
              </div>

              {/* ── 9. WISCO v2 Retrieval Comparison ─────────────────────── */}
              <div className="bg-white border border-blue-200 rounded-xl overflow-hidden shadow-sm">
                <div className="px-4 py-2.5 border-b border-blue-100 bg-blue-50 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <h3 className="text-xs font-semibold text-blue-700 uppercase tracking-widest">
                      {t.evalSection}
                    </h3>
                    <span className="text-[10px] bg-blue-100 text-blue-700 border border-blue-300 px-1.5 py-0.5 rounded-full font-medium">
                      Controlled Benchmark
                    </span>
                  </div>
                  <span className="text-[10px] text-gray-400 font-mono">n={WISCO_EVAL.heldoutCases.toLocaleString()}</span>
                </div>

                <div className="px-4 py-4 space-y-5">
                  <p className="text-[11px] text-gray-500">{t.evalSubtitle}</p>

                  {/* ── Headline table ── */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="text-left py-2 pr-3 text-gray-500 font-semibold">{t.evalSystem}</th>
                          <th className="text-center py-2 px-2 text-gray-500 font-semibold">{t.evalCases}</th>
                          <th className="text-center py-2 px-2 text-gray-500 font-semibold">{t.evalCorrect}</th>
                          <th className="text-center py-2 px-2 text-gray-500 font-semibold">{t.evalAccuracy}</th>
                          <th className="text-center py-2 pl-2 text-gray-500 font-semibold">{t.evalCi}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {WISCO_EVAL.headline.map((sys) => {
                          const pct   = (sys.accuracy * 100).toFixed(2);
                          const color = sys.key === "flat" ? "text-emerald-700" : "text-amber-700";
                          const barW  = `${sys.accuracy * 100}%`;
                          const barCol = sys.key === "flat" ? "bg-emerald-500" : "bg-amber-500";
                          return (
                            <tr key={sys.key} className="border-b border-gray-100">
                              <td className="py-3 pr-3">
                                <p className={`font-semibold ${color}`}>{sys.name}</p>
                              </td>
                              <td className="py-3 px-2 text-center font-mono tabular-nums text-gray-700">
                                {sys.correct.toLocaleString()}/{WISCO_EVAL.heldoutCases.toLocaleString()}
                              </td>
                              <td className="py-3 px-2 text-center font-mono tabular-nums text-gray-500">
                                {sys.correct.toLocaleString()}
                              </td>
                              <td className="py-3 px-2 text-center">
                                <div className="flex flex-col items-center gap-1">
                                  <span className={`font-bold tabular-nums ${color}`}>{pct}%</span>
                                  <div className="w-16 bg-gray-200 rounded-full h-1.5">
                                    <div className={`${barCol} h-1.5 rounded-full`} style={{ width: barW }} />
                                  </div>
                                </div>
                              </td>
                              <td className="py-3 pl-2 text-center font-mono tabular-nums text-gray-500">
                                [{(sys.ciLow * 100).toFixed(2)}%, {(sys.ciHigh * 100).toFixed(2)}%]
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                    <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-[10px] text-gray-500">
                      <span>{t.evalDiff}: <strong className="text-red-600">{WISCO_EVAL.diffPct.toFixed(2)} pp</strong></span>
                      <span>{t.evalMcnemar}: <strong>p ≈ {WISCO_EVAL.mcnemarP}</strong></span>
                    </div>
                  </div>

                  {/* ── Per-major-group accuracy table ── */}
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">{t.evalPerMajor}</p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs border-collapse">
                        <thead>
                          <tr className="border-b border-gray-200">
                            <th className="text-left py-1.5 pr-3 text-gray-500 font-semibold">{t.evalMajorGroup}</th>
                            <th className="text-center py-1.5 px-2 text-emerald-700 font-semibold">Flat</th>
                            <th className="text-center py-1.5 pl-2 text-amber-700 font-semibold">Hierarchical</th>
                          </tr>
                        </thead>
                        <tbody>
                          {WISCO_EVAL.perMajor.map((row) => {
                            const info = ISCO_MAJOR[row.code];
                            return (
                              <tr key={row.code} className="border-b border-gray-100 hover:bg-gray-50">
                                <td className="py-1.5 pr-3">
                                  <span className="text-gray-400 font-mono mr-1.5">{row.code}</span>
                                  <span className="text-gray-700">{info ? info.label : row.code}</span>
                                </td>
                                <td className="py-1.5 px-2 text-center font-mono tabular-nums text-emerald-700">{(row.flat * 100).toFixed(2)}%</td>
                                <td className="py-1.5 pl-2 text-center font-mono tabular-nums text-amber-700">{(row.hier * 100).toFixed(2)}%</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* ── Per-language accuracy table ── */}
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">{t.evalPerLanguage}</p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs border-collapse">
                        <thead>
                          <tr className="border-b border-gray-200">
                            <th className="text-left py-1.5 pr-3 text-gray-500 font-semibold">{t.evalLanguage}</th>
                            <th className="text-center py-1.5 px-2 text-emerald-700 font-semibold">Flat</th>
                            <th className="text-center py-1.5 pl-2 text-amber-700 font-semibold">Hierarchical</th>
                          </tr>
                        </thead>
                        <tbody>
                          {WISCO_EVAL.perLanguage.map((row) => (
                            <tr key={row.code} className="border-b border-gray-100 hover:bg-gray-50">
                              <td className="py-1.5 pr-3">
                                <span className="text-gray-400 font-mono mr-1.5 uppercase">{row.code}</span>
                                <span className="text-gray-700">{row.label}</span>
                              </td>
                              <td className="py-1.5 px-2 text-center font-mono tabular-nums text-emerald-700">{(row.flat * 100).toFixed(2)}%</td>
                              <td className="py-1.5 pl-2 text-center font-mono tabular-nums text-amber-700">{(row.hier * 100).toFixed(2)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* ── Interpretation ── */}
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-1.5">
                    <p className="text-xs font-semibold text-gray-700">{t.evalInterpretHeader}</p>
                    <p className="text-[11px] text-gray-600 leading-relaxed">{t.evalInterpret}</p>
                  </div>

                  {/* ── What this does not establish ── */}
                  <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
                    <p className="text-xs font-semibold text-amber-800 mb-1.5">{t.evalNotEstablishHeader}</p>
                    <ul className="space-y-1">
                      {t.evalNotEstablish.map((item, i) => (
                        <li key={i} className="text-[11px] text-amber-800 leading-relaxed flex gap-2">
                          <span className="flex-shrink-0">·</span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <p className="text-[10px] text-gray-400 italic">{t.evalSource}</p>
                </div>
              </div>

              {/* ── Research metadata footer ─────────────────────────────── */}
              <div className="bg-white border border-gray-200 rounded-xl px-4 py-3 shadow-sm">
                <div className="flex flex-wrap gap-2 justify-center">
                  {["ILO ICLS-19", "ISCO-08", "ISIC Rev.4", "ISCED 2011", "UN M49"].map((s) => (
                    <span key={s} className="text-[10px] font-mono bg-gray-100 border border-gray-200 text-gray-600 px-2 py-0.5 rounded">
                      {s}
                    </span>
                  ))}
                </div>
                <p className="text-[10px] text-gray-400 text-center mt-2">{t.researchFooter}</p>
              </div>

              {/* Back link */}
              <div className="text-center pb-4">
                <button onClick={() => router.push("/")}
                  className="text-xs text-gray-400 hover:text-gray-700 underline transition-colors">
                  {t.backToHome}
                </button>
              </div>
            </>
          )}
        </main>
      </div>
    </>
  );
}
