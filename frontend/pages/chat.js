/**
 * Chat page — LFS survey conversation.
 * Prototype-matching light theme: DM Mono + Inter, green #007a62, right panel stats.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { createSession, sendMessage } from "../components/api";

// ── i18n strings ──────────────────────────────────────────────────────────────

const T = {
  en: {
    title: "LFS Survey",
    startingSession: "Starting your survey session…",
    inputPlaceholder: "Type your message…",
    sendButton: "Send",
    completedBanner: "Thank you! Your survey responses have been recorded.",
    codeSwitched: "Code-switched input detected",
    iscoLabel: "ISCO classification:",
    confidence: "confidence",
    signOut: "Sign out",
    sessionError: "Could not start session. Please reload the page.",
    sendError: "Could not send message. Please try again.",
    entities: "Entities found",
    stateLabel: "State",
    orType: "or tap an option above",
  },
  ar: {
    title: "مسح القوى العاملة",
    startingSession: "جارٍ بدء جلسة المسح…",
    inputPlaceholder: "اكتب رسالتك…",
    sendButton: "إرسال",
    completedBanner: "شكرًا لك! تم تسجيل إجاباتك.",
    codeSwitched: "تم اكتشاف تبديل رمز اللغة",
    iscoLabel: ":تصنيف ISCO",
    confidence: "الثقة",
    signOut: "تسجيل الخروج",
    sessionError: "تعذّر بدء الجلسة. يرجى إعادة تحميل الصفحة.",
    sendError: "تعذّر إرسال الرسالة. يرجى المحاولة مرة أخرى.",
    entities: "الكيانات المكتشفة",
    stateLabel: "الحالة",
    orType: "أو اكتب إجابتك أدناه",
  },
  ur: {
    title: "لیبر فورس سروے",
    startingSession: "آپ کا سروے سیشن شروع ہو رہا ہے…",
    inputPlaceholder: "اپنا پیغام ٹائپ کریں…",
    sendButton: "بھیجیں",
    completedBanner: "شکریہ! آپ کے جوابات محفوظ کر لیے گئے ہیں۔",
    codeSwitched: "ملی جلی زبان کا ان پٹ",
    iscoLabel: ":ISCO درجہ بندی",
    confidence: "اعتماد",
    signOut: "سائن آؤٹ",
    sessionError: "سیشن شروع نہیں ہو سکا۔ براہ کرم صفحہ دوبارہ لوڈ کریں۔",
    sendError: "پیغام نہیں بھیجا جا سکا۔ براہ کرم دوبارہ کوشش کریں۔",
    entities: "دریافت شدہ ہستیاں",
    stateLabel: "حالت",
    orType: "یا اپنا جواب نیچے ٹائپ کریں",
  },
  hi: {
    title: "श्रम बल सर्वेक्षण",
    startingSession: "आपका सर्वेक्षण सत्र शुरू हो रहा है…",
    inputPlaceholder: "अपना संदेश टाइप करें…",
    sendButton: "भेजें",
    completedBanner: "धन्यवाद! आपके उत्तर दर्ज कर लिए गए हैं।",
    codeSwitched: "मिश्रित भाषा इनपुट",
    iscoLabel: "ISCO वर्गीकरण:",
    confidence: "विश्वास",
    signOut: "साइन आउट",
    sessionError: "सत्र शुरू नहीं हो सका। कृपया पृष्ठ पुनः लोड करें।",
    sendError: "संदेश नहीं भेजा जा सका। कृपया पुनः प्रयास करें।",
    entities: "मिली हस्तियाँ",
    stateLabel: "स्थिति",
    orType: "या नीचे अपना उत्तर टाइप करें",
  },
  tl: {
    title: "Labour Force Survey",
    startingSession: "Sinisimulan ang iyong survey session…",
    inputPlaceholder: "I-type ang iyong mensahe…",
    sendButton: "Ipadala",
    completedBanner: "Salamat! Naitala na ang iyong mga sagot.",
    codeSwitched: "Natukoy ang code-switched na input",
    iscoLabel: "ISCO classification:",
    confidence: "kumpiyansa",
    signOut: "Mag-sign out",
    sessionError: "Hindi masimulan ang session. Mangyaring i-reload ang pahina.",
    sendError: "Hindi maipadala ang mensahe. Mangyaring subukan ulit.",
    entities: "Mga nahanap na entity",
    stateLabel: "Katayuan",
    orType: "o i-type ang iyong sagot sa ibaba",
  },
};

const LANG_KEY_MAP = { "ar-gulf": "ar", other: "en" };
const getLangKey = (l) => LANG_KEY_MAP[l] || (T[l] ? l : "en");
const RTL_LANGS = new Set(["ar", "ar-gulf", "ur"]);

// ── Quick-reply options for structured fields ─────────────────────────────────
const QUICK_OPTIONS = {
  employment_status: {
    en: ["Employed", "Unemployed", "Not in the labour force"],
    ar: ["موظف", "عاطل عن العمل", "خارج سوق العمل"],
    ur: ["ملازم", "بے روزگار", "افرادی قوت سے باہر"],
    hi: ["नियोजित", "बेरोजगार", "श्रम बल से बाहर"],
    tl: ["Employed", "Unemployed", "Not in the labour force"],
  },
  education_level: {
    en: ["No formal education", "Primary", "Secondary", "Diploma", "Bachelor's degree", "Master's degree", "PhD or higher"],
    ar: ["بدون تعليم رسمي", "ابتدائي", "ثانوي", "دبلوم", "بكالوريوس", "ماجستير", "دكتوراه"],
    ur: ["کوئی رسمی تعلیم نہیں", "ابتدائی", "ثانوی", "ڈپلومہ", "بیچلر ڈگری", "ماسٹر ڈگری", "پی ایچ ڈی یا اس سے زیادہ"],
    hi: ["कोई औपचारिक शिक्षा नहीं", "प्राथमिक", "माध्यमिक", "डिप्लोमा", "स्नातक", "स्नातकोत्तर", "पीएचडी या उच्चतर"],
    tl: ["Walang pormal na edukasyon", "Primarya", "Sekundarya", "Diploma", "Batsilyer", "Master's degree", "Doktorado o mas mataas"],
  },
  employment_nature: {
    en: ["Paid employee", "Employer", "Self-employed", "Contributing family worker"],
    ar: ["موظف براتب", "صاحب عمل", "عمل حر", "عامل عائلي مساهم"],
    ur: ["تنخواہ دار ملازم", "آجر", "خود ملازم", "خاندانی کارکن"],
    hi: ["वेतनभोगी कर्मचारी", "नियोक्ता", "स्व-रोज़गार", "परिवार का सहायक कार्यकर्ता"],
    tl: ["Bayad na empleyado", "Employer", "Self-employed", "Contributing family worker"],
  },
  employment_sector: {
    en: ["Government", "Private", "Semi-government", "Non-profit / NGO"],
    ar: ["حكومي", "خاص", "شبه حكومي", "غير ربحي / منظمة غير حكومية"],
    ur: ["حکومتی", "نجی", "نیم حکومتی", "غیر منافع بخش / این جی او"],
    hi: ["सरकारी", "निजी", "अर्ध-सरकारी", "गैर-लाभकारी / एनजीओ"],
    tl: ["Pamahalaan", "Pribado", "Semi-gobyerno", "Non-profit / NGO"],
  },
  employment_type: {
    en: ["Full-time", "Part-time", "Seasonal", "Casual"],
    ar: ["دوام كامل", "دوام جزئي", "موسمي", "عَرَضي"],
    ur: ["کل وقتی", "جزوی وقتی", "موسمی", "عارضی"],
    hi: ["पूर्णकालिक", "अंशकालिक", "मौसमी", "आकस्मिक"],
    tl: ["Full-time", "Part-time", "Seasonal", "Casual"],
  },
  monthly_wage_range: {
    en: ["Less than 5,000", "5,000–10,000", "10,001–20,000", "20,001–50,000", "More than 50,000", "Prefer not to say"],
    ar: ["أقل من 5,000", "5,000–10,000", "10,001–20,000", "20,001–50,000", "أكثر من 50,000", "أفضل عدم الإفصاح"],
    ur: ["5,000 سے کم", "5,000–10,000", "10,001–20,000", "20,001–50,000", "50,000 سے زیادہ", "بتانا نہیں چاہتا"],
    hi: ["5,000 से कम", "5,000–10,000", "10,001–20,000", "20,001–50,000", "50,000 से अधिक", "बताना नहीं चाहते"],
    tl: ["Wala pang 5,000", "5,000–10,000", "10,001–20,000", "20,001–50,000", "Higit sa 50,000", "Ayaw sabihin"],
  },
  job_search_active: {
    en: ["Yes", "No"], ar: ["نعم", "لا"], ur: ["ہاں", "نہیں"], hi: ["हाँ", "नहीं"], tl: ["Oo", "Hindi"],
  },
  available_for_work: {
    en: ["Yes", "No"], ar: ["نعم", "لا"], ur: ["ہاں", "نہیں"], hi: ["हाँ", "नहीں"], tl: ["Oo", "Hindi"],
  },
  reason_left_job: {
    en: ["Made redundant", "Resigned", "Business closed", "Contract ended", "Other"],
    ar: ["فائض عن الحاجة", "استقالة", "إغلاق المنشأة", "انتهاء العقد", "أخرى"],
    ur: ["فاضل", "استعفیٰ", "کاروبار بند", "معاہدہ ختم", "دیگر"],
    hi: ["छंटनी", "इस्तीफा", "व्यापार बंद", "अनुबंध समाप्त", "अन्य"],
    tl: ["Tinanggal", "Nagbitiw", "Nagsara ang negosyo", "Natapos ang kontrata", "Iba pa"],
  },
  outside_lf_reason: {
    en: ["Retired", "Student", "Homemaker", "Discouraged worker", "Illness or disability", "Other"],
    ar: ["متقاعد", "طالب", "ربة منزل", "يأس من إيجاد عمل", "مرض أو إعاقة", "أخرى"],
    ur: ["ریٹائرڈ", "طالب علم", "گھریلو", "مایوس کارکن", "بیماری یا معذوری", "دیگر"],
    hi: ["सेवानिवृत्त", "छात्र", "गृहणी", "निराश श्रमिक", "बीमारी या विकलांगता", "अन्य"],
    tl: ["Retirado", "Mag-aaral", "Nag-aalaga ng tahanan", "Discouraged worker", "Sakit o kapansanan", "Iba pa"],
  },
  ai_preference: {
    en: ["Prefer AI", "Prefer human", "No preference"],
    ar: ["أفضل الذكاء الاصطناعي", "أفضل المحاور البشري", "لا فرق"],
    ur: ["AI کو ترجیح", "انسان کو ترجیح", "کوئی ترجیح نہیں"],
    hi: ["AI को प्राथमिकता", "मानव को प्राथमिकता", "कोई प्राथमिकता नहीं"],
    tl: ["Mas gusto ang AI", "Mas gusto ang tao", "Walang kagustuhan"],
  },
  data_confidence: {
    en: ["Very confident", "Somewhat confident", "Not confident"],
    ar: ["واثق جدًا", "واثق نسبيًا", "غير واثق"],
    ur: ["بہت پراعتماد", "کچھ حد تک پراعتماد", "پراعتماد نہیں"],
    hi: ["बहुत आत्मविश्वास", "थोड़ा आत्मविश्वास", "आत्मविश्वास नहीं"],
    tl: ["Lubos na tiwala", "Medyo tiwala", "Hindi tiwala"],
  },
  gender: {
    en: ["Male", "Female", "Prefer not to say"],
    ar: ["ذكر", "أنثى", "أفضل عدم الإفصاح"],
    ur: ["مرد", "عورت", "بتانا نہیں چاہتا"],
    hi: ["पुरुष", "महिला", "बताना नहीं चाहते"],
    tl: ["Lalaki", "Babae", "Ayaw sabihin"],
  },
  nationality: {
    en: ["Emirati", "Indian", "Pakistani", "Filipino", "Bangladeshi", "Egyptian", "British", "Other"],
    ar: ["إماراتي", "هندي", "باكستاني", "فلبيني", "بنغلاديشي", "مصري", "بريطاني", "أخرى"],
    ur: ["اماراتی", "ہندوستانی", "پاکستانی", "فلپینو", "بنگلادیشی", "مصری", "برطانوی", "دیگر"],
    hi: ["इमिराती", "भारतीय", "पाकिस्तानी", "फिलिपीनो", "बांग्लादेशी", "मिस्री", "ब्रिटिश", "अन्य"],
    tl: ["Emirati", "Indian", "Pakistani", "Filipino", "Bangladeshi", "Egyptian", "British", "Iba pa"],
  },
  marital_status: {
    en: ["Single", "Married", "Divorced", "Widowed"],
    ar: ["أعزب", "متزوج", "مطلق", "أرمل"],
    ur: ["غیر شادی شدہ", "شادی شدہ", "طلاق یافتہ", "بیوہ"],
    hi: ["अविवाहित", "विवाहित", "तलाकशुदा", "विधवा/विधुर"],
    tl: ["Walang asawa", "May asawa", "Hiwalay", "Biyudo/Biyuda"],
  },
  emirate: {
    en: ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", "Ras Al Khaimah", "Fujairah"],
    ar: ["أبوظبي", "دبي", "الشارقة", "عجمان", "أم القيوين", "رأس الخيمة", "الفجيرة"],
    ur: ["ابوظبی", "دبئی", "شارجہ", "عجمان", "ام القیوین", "رأس الخیمہ", "فجیرہ"],
    hi: ["अबू धाबी", "दुबई", "शारजाह", "अजमान", "उम्म अल क्वैन", "रास अल खैमाह", "फुजैरा"],
    tl: ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", "Ras Al Khaimah", "Fujairah"],
  },
  uae_residence_duration: {
    en: ["Born in UAE", "Less than 1 year", "1\u20134 years", "5\u20139 years", "10\u201319 years", "20+ years"],
    ar: ["مولود في الإمارات", "أقل من سنة", "1-4 سنوات", "5-9 سنوات", "10-19 سنة", "20 سنة فأكثر"],
    ur: ["یو اے ای میں پیدا ہوا", "1 سال سے کم", "1-4 سال", "5-9 سال", "10-19 سال", "20+ سال"],
    hi: ["UAE में जन्मे", "1 साल से कम", "1-4 साल", "5-9 साल", "10-19 साल", "20+ साल"],
    tl: ["Ipinanganak sa UAE", "Wala pang 1 taon", "1-4 taon", "5-9 taon", "10-19 taon", "20+ taon"],
  },
  vocational_training: {
    en: ["Yes", "No"], ar: ["نعم", "لا"], ur: ["ہاں", "نہیں"], hi: ["हाँ", "नहीं"], tl: ["Oo", "Hindi"],
  },
  secondary_job: {
    en: ["Yes", "No"], ar: ["نعم", "لا"], ur: ["ہاں", "نہیں"], hi: ["हاँ", "नहीं"], tl: ["Oo", "Hindi"],
  },
  underemployment: {
    en: ["Yes \u2014 I want more hours", "No", "Already overemployed"],
    ar: ["نعم — أريد ساعات أكثر", "لا", "أعمل أكثر من اللازم"],
    ur: ["ہاں — زیادہ گھنٹے چاہیے", "نہیں", "پہلے سے زیادہ کام"],
    hi: ["हाँ — अधिक घंटे चाहिए", "नहीं", "पहले से अधिक काम"],
    tl: ["Oo \u2014 gusto ng mas maraming oras", "Hindi", "Sobra na ang oras ng trabaho"],
  },
  contract_type: {
    en: ["Permanent", "Fixed-term (< 1 year)", "Fixed-term (1\u20133 years)", "Probation period", "No written contract"],
    ar: ["دائم", "عقد محدد المدة (أقل من سنة)", "عقد محدد المدة (1-3 سنوات)", "فترة تجريبية", "بدون عقد مكتوب"],
    ur: ["مستقل", "مقررہ مدت (1 سال سے کم)", "مقررہ مدت (1-3 سال)", "آزمائشی مدت", "کوئی تحریری معاہدہ نہیں"],
    hi: ["स्थायी", "निश्चित अवधि (1 साल से कम)", "निश्चित अवधि (1-3 साल)", "परीविक्षा अवधि", "कोई लिखित अनुबंध नहीं"],
    tl: ["Permanente", "Nakatakdang termino (< 1 taon)", "Nakatakdang termino (1-3 taon)", "Probationary", "Walang kontrata"],
  },
  remote_work: {
    en: ["Always", "Mostly", "Partially", "Never"],
    ar: ["دائمًا", "في الغالب", "جزئيًا", "أبدًا"],
    ur: ["ہمیشہ", "زیادہ تر", "جزوی طور پر", "کبھی نہیں"],
    hi: ["हमेशा", "ज़्यादातर", "आंशिक रूप से", "कभी नहीं"],
    tl: ["Palagi", "Kadalasan", "Bahagi", "Hindi kailanman"],
  },
  health_insurance: {
    en: ["Full coverage", "Partial coverage", "No", "I pay for my own"],
    ar: ["تغطية كاملة", "تغطية جزئية", "لا", "أدفع من راتبي"],
    ur: ["مکمل کوریج", "جزوی کوریج", "نہیں", "اپنا ادا کرتا ہوں"],
    hi: ["पूर्ण कवरेज", "आंशिक कवरेज", "नहीं", "स्वयं भुगतान करते हैं"],
    tl: ["Buong saklaw", "Bahagyang saklaw", "Wala", "Sarili kong binabayaran"],
  },
  desired_job_type: {
    en: ["Same as previous occupation", "Different occupation", "First job"],
    ar: ["نفس مهنتي السابقة", "مهنة مختلفة", "أبحث عن أول وظيفة"],
    ur: ["پہلے جیسا کام", "مختلف کام", "پہلی نوکری"],
    hi: ["पिछले जैसा काम", "अलग काम", "पहली नौकरी"],
    tl: ["Katulad ng dati", "Ibang trabaho", "Unang trabaho"],
  },
  ever_worked: {
    en: ["Yes \u2014 last job was in UAE", "Yes \u2014 last job was outside UAE", "No \u2014 never worked"],
    ar: ["نعم — آخر عمل في الإمارات", "نعم — آخر عمل خارج الإمارات", "لا — لم أعمل أبداً"],
    ur: ["ہاں — آخری کام UAE میں تھا", "ہاں — آخری کام UAE سے باہر تھا", "نہیں — کبھی نہیں کیا"],
    hi: ["हाँ — पिछला काम UAE में था", "हाँ — पिछला काम UAE से बाहर था", "नहीं — कभी नहीं किया"],
    tl: ["Oo \u2014 sa UAE ang huling trabaho", "Oo \u2014 sa labas ng UAE", "Hindi \u2014 hindi pa nagtrabaho"],
  },
  qualification_match: {
    en: ["Overqualified", "Well matched", "Underqualified"],
    ar: ["مؤهل أكثر من اللازم", "مطابق تماماً", "مؤهل أقل من اللازم"],
    ur: ["بہت زیادہ قابل", "مناسب", "کم قابل"],
    hi: ["अति-योग्य", "उचित रूप से मेल", "कम योग्य"],
    tl: ["Sobrang-kwalipikado", "Angkop", "Hindi sapat ang kwalipikasyon"],
  },
  training_participation: {
    en: ["Yes \u2014 employer-funded", "Yes \u2014 self-funded", "Yes \u2014 government program", "No"],
    ar: ["نعم — ممول من صاحب العمل", "نعم — ممول ذاتيًا", "نعم — برنامج حكومي", "لا"],
    ur: ["ہاں — آجر کی طرف سے", "ہاں — اپنے خرچے پر", "ہاں — سرکاری پروگرام", "نہیں"],
    hi: ["हाँ — नियोक्ता द्वारा", "हाँ — स्वयं-वित्त पोषित", "हाँ — सरकारी कार्यक्रम", "नहीं"],
    tl: ["Oo \u2014 pinondohan ng employer", "Oo \u2014 sariling gastos", "Oo \u2014 programa ng gobyerno", "Hindi"],
  },
  platform_work: {
    en: ["Yes \u2014 primary income", "Yes \u2014 supplementary income", "No"],
    ar: ["نعم — دخل رئيسي", "نعم — دخل إضافي", "لا"],
    ur: ["ہاں — بنیادی آمدنی", "ہاں — اضافی آمدنی", "نہیں"],
    hi: ["हाँ — मुख्य आय", "हाँ — अतिरिक्त आय", "नहीं"],
    tl: ["Oo \u2014 pangunahing kita", "Oo \u2014 karagdagang kita", "Hindi"],
  },
  online_business: {
    en: ["Yes \u2014 registered business", "Yes \u2014 informal", "No"],
    ar: ["نعم — نشاط مسجل", "نعم — غير رسمي", "لا"],
    ur: ["ہاں — رجسٹرڈ", "ہاں — غیر رسمی", "نہیں"],
    hi: ["हाँ — पंजीकृत व्यापार", "हाँ — अनौपचारिक", "नहीं"],
    tl: ["Oo \u2014 rehistradong negosyo", "Oo \u2014 impormal", "Hindi"],
  },
  work_life_balance: {
    en: ["Yes", "Somewhat", "No"], ar: ["نعم", "نوعًا ما", "لا"],
    ur: ["ہاں", "کچھ حد تک", "نہیں"], hi: ["हाँ", "कुछ हद तक", "नहीं"],
    tl: ["Oo", "Medyo", "Hindi"],
  },
  question_clarity: {
    en: ["1 \u2014 Very unclear", "2", "3 \u2014 Neutral", "4", "5 \u2014 Very clear"],
    ar: ["1 — غير واضح جداً", "2", "3 — محايد", "4", "5 — واضح جداً"],
    ur: ["1 — بالکل واضح نہیں", "2", "3 — غیر جانبدار", "4", "5 — بہت واضح"],
    hi: ["1 — बिल्कुल स्पष्ट नहीं", "2", "3 — तटस्थ", "4", "5 — बहुत स्पष्ट"],
    tl: ["1 \u2014 Hindi malinaw", "2", "3 \u2014 Neutral", "4", "5 \u2014 Malinaw"],
  },
  job_satisfaction: {
    en: ["1 — Very dissatisfied", "2", "3 — Neutral", "4", "5 — Very satisfied"],
    ar: ["1 — غير راضٍ جداً", "2", "3 — محايد", "4", "5 — راضٍ جداً"],
    ur: ["1 — بہت غیر مطمئن", "2", "3 — غیر جانبدار", "4", "5 — بہت مطمئن"],
    hi: ["1 — बहुत असंतुष्ट", "2", "3 — तटस्थ", "4", "5 — बहुत संतुष्ट"],
    tl: ["1 — Napaka-hindi nasiyahan", "2", "3 — Neutral", "4", "5 — Nasiyahan"],
  },
  work_safety: {
    en: ["Always", "Mostly", "Sometimes", "Rarely", "Never"],
    ar: ["دائمًا", "في الغالب", "أحيانًا", "نادرًا", "أبدًا"],
    ur: ["ہمیشہ", "زیادہ تر", "کبھی کبھی", "شاذ و نادر", "کبھی نہیں"],
    hi: ["हमेशा", "ज़्यादातर", "कभी-कभी", "कभी-कभार", "कभी नहीं"],
    tl: ["Palagi", "Kadalasan", "Minsan", "Bihira", "Hindi kailanman"],
  },
  bonuses: {
    en: ["Yes — annual bonus", "Yes — performance bonus", "Yes — other", "No"],
    ar: ["نعم — مكافأة سنوية", "نعم — حافز أداء", "نعم — أخرى", "لا"],
    ur: ["ہاں — سالانہ بونس", "ہاں — کارکردگی بونس", "ہاں — دیگر", "نہیں"],
    hi: ["हाँ — वार्षिक बोनस", "हाँ — प्रदर्शन बोनस", "हाँ — अन्य", "नहीं"],
    tl: ["Oo — taunang bonus", "Oo — performance bonus", "Oo — iba", "Hindi"],
  },
  pension_scheme: {
    en: ["Yes — GPSSA (UAE National)", "Yes — DIFC/ADGM scheme", "Yes — employer private scheme", "No", "Not sure"],
    ar: ["نعم — هيئة المعاشات", "نعم — نظام DIFC/ADGM", "نعم — خطة صاحب العمل", "لا", "غير متأكد"],
    ur: ["ہاں — GPSSA", "ہاں — DIFC/ADGM", "ہاں — آجر کا نجی منصوبہ", "نہیں", "یقین نہیں"],
    hi: ["हाँ — GPSSA", "हाँ — DIFC/ADGM", "हाँ — नियोक्ता की निजी योजना", "नहीं", "पता नहीं"],
    tl: ["Oo — GPSSA", "Oo — DIFC/ADGM", "Oo — pribadong plano ng employer", "Hindi", "Hindi sigurado"],
  },
  emiratization_program: {
    en: ["Yes — NAFIS", "Yes — other government program", "No"],
    ar: ["نعم — نافس", "نعم — برنامج حكومي آخر", "لا"],
    ur: ["ہاں — NAFIS", "ہاں — دوسرا سرکاری پروگرام", "نہیں"],
    hi: ["हाँ — NAFIS", "हाँ — अन्य सरकारी कार्यक्रम", "नहीं"],
    tl: ["Oo — NAFIS", "Oo — ibang programa ng gobyerno", "Hindi"],
  },
  last_job_sector: {
    en: ["Government", "Private", "Semi-government", "Non-profit / NGO", "Self-employed"],
    ar: ["حكومي", "خاص", "شبه حكومي", "غير ربحي / منظمة غير حكومية", "عمل حر"],
    ur: ["حکومتی", "نجی", "نیم حکومتی", "غیر منافع بخش / این جی او", "خود ملازم"],
    hi: ["सरकारी", "निजी", "अर्ध-सरकारी", "गैर-लाभकारी / एनजीओ", "स्व-रोज़गार"],
    tl: ["Pamahalaan", "Pribado", "Semi-gobyerno", "Non-profit / NGO", "Self-employed"],
  },
  highest_previous_salary: {
    en: ["Less than 5,000", "5,000–10,000", "10,001–20,000", "20,001–50,000", "More than 50,000", "Prefer not to say"],
    ar: ["أقل من 5,000", "5,000–10,000", "10,001–20,000", "20,001–50,000", "أكثر من 50,000", "أفضل عدم الإفصاح"],
    ur: ["5,000 سے کم", "5,000–10,000", "10,001–20,000", "20,001–50,000", "50,000 سے زیادہ", "بتانا نہیں چاہتا"],
    hi: ["5,000 से कम", "5,000–10,000", "10,001–20,000", "20,001–50,000", "50,000 से अधिक", "बताना नहीं चाहते"],
    tl: ["Wala pang 5,000", "5,000–10,000", "10,001–20,000", "20,001–50,000", "Higit sa 50,000", "Ayaw sabihin"],
  },
  difficulty_answering: {
    en: ["No — all questions were clear", "Yes — some questions were unclear"],
    ar: ["لا — جميع الأسئلة كانت واضحة", "نعم — بعض الأسئلة غير واضحة"],
    ur: ["نہیں — سب سوال واضح تھے", "ہاں — کچھ سوال واضح نہیں تھے"],
    hi: ["नहीं — सभी प्रश्न स्पष्ट थे", "हाँ — कुछ प्रश्न अस्पष्ट थे"],
    tl: ["Hindi — malinaw ang lahat ng tanong", "Oo — may hindi malinaw na tanong"],
  },
};

// ── Design tokens ─────────────────────────────────────────────────────────────
const C = {
  green:   "#007a62",
  greenLt: "#e6f4f1",
  bg:      "#ffffff",
  white:   "#ffffff",
  border:  "#e5e7eb",
  text:    "#1a1a2e",
  muted:   "#6b7280",
  faint:   "#9ca3af",
  amber:   "#d97706",
  amberBg: "#fefce8",
  amberBd: "#fde68a",
  purple:  "#7c3aed",
  purpleBg:"#f5f3ff",
  purpleBd:"#e9d5ff",
  panel:   "#f8fafc",
};

const MONO = "'DM Mono', monospace";
const SANS = "'Inter', sans-serif";

// ── Prototype constants ───────────────────────────────────────────────────────
const TOTAL_QUESTIONS = 155;

const FIELD_TO_SECTION = {
  employment_status:"B", education_level:"B", field_of_study:"B", gender:"B",
  nationality:"B", marital_status:"B", emirate:"B", uae_residence_duration:"B", vocational_training:"B",
  employment_nature:"C", employment_sector:"C", job_title:"C", job_duties:"C", industry:"C",
  hours_per_week:"D", secondary_job:"D", underemployment:"D", employment_type:"D",
  contract_type:"D", remote_work:"D",
  monthly_wage_range:"E", health_insurance:"E", bonuses:"E", pension_scheme:"E",
  job_search_active:"F", available_for_work:"F", unemployment_duration:"F",
  outside_lf_reason:"F", desired_job_type:"F", ever_worked:"F",
  last_job_title:"G", reason_left_job:"G", last_job_sector:"G", highest_previous_salary:"G",
  qualification_match:"H", training_participation:"H", emiratization_program:"H",
  platform_work:"I", online_business:"I",
  job_satisfaction:"J", work_safety:"J", work_life_balance:"J",
  question_clarity:"K", difficulty_answering:"K", ai_preference:"K", data_confidence:"K",
};

const SECTIONS = [
  { id:"B", en:"Demographics",       ar:"البيانات الشخصية" },
  { id:"C", en:"Current Employment", ar:"التوظيف الحالي" },
  { id:"D", en:"Working Conditions", ar:"ظروف العمل" },
  { id:"E", en:"Compensation",       ar:"التعويضات" },
  { id:"F", en:"Job Search",         ar:"البحث عن عمل" },
  { id:"G", en:"Previous Employment",ar:"الوظيفة السابقة" },
  { id:"H", en:"Skills & Training",  ar:"المهارات والتدريب" },
  { id:"I", en:"Digital Economy",    ar:"الاقتصاد الرقمي" },
  { id:"J", en:"Wellbeing",          ar:"الرفاهية" },
  { id:"K", en:"Survey Feedback",    ar:"تقييم المسح" },
];

const AGENTS = [
  { id:"A1",  label:"Language Processor" },
  { id:"A2",  label:"ISCO Classifier" },
  { id:"A3",  label:"ISIC Classifier" },
  { id:"A4",  label:"ISCED Classifier" },
  { id:"A5",  label:"RAG Expert" },
  { id:"A6",  label:"Validation Agent" },
  { id:"A7",  label:"Emotional Intel." },
  { id:"A8",  label:"HITL Quality Mgr" },
  { id:"A9",  label:"Audit Logger" },
  { id:"A10", label:"Report Generator" },
];

const LLM_ROLES = {
  greeting:       "Generating warm multilingual greeting",
  collecting_info:"Extracting entities & routing next question",
  clarifying:     "Resolving ambiguous or conflicting answers",
  validating:     "Cross-checking responses for ILO ICLS-19",
  completing:     "Generating personalised completion narrative",
};

const LANG_PILLS = [
  { code:"en", label:"EN" },
  { code:"ar", label:"AR" },
  { code:"ur", label:"UR" },
  { code:"hi", label:"HI" },
  { code:"tl", label:"TL" },
];

// Field labels for the welcome-back banner
const PREFILL_LABELS = {
  employment_status: "Employment Status", education_level: "Education",
  gender: "Gender", nationality: "Nationality", marital_status: "Marital Status",
  emirate: "Emirate", uae_residence_duration: "UAE Residence",
  employment_nature: "Employment Nature", employment_sector: "Sector",
  job_title: "Job Title", industry: "Industry",
  hours_per_week: "Weekly Hours", employment_type: "Employment Type",
  monthly_wage_range: "Monthly Wage",
};

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmtTime = (s) =>
  `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

// ── ChatPage ──────────────────────────────────────────────────────────────────

export default function ChatPage() {
  const router = useRouter();

  // Core survey state
  const [lang, setLang]                   = useState("en");
  const [token, setToken]                 = useState(null);
  const [sessionId, setSessionId]         = useState(null);
  const [messages, setMessages]           = useState([]);
  const [input, setInput]                 = useState("");
  const [sending, setSending]             = useState(false);
  const [completed, setCompleted]         = useState(false);
  const [pageError, setPageError]         = useState("");
  const [initialising, setInitialising]   = useState(true);
  const [nextField, setNextField]         = useState(null);
  const [surveyProgress, setSurveyProgress] = useState(null);

  // Right-panel state
  const [fsmState, setFsmState]       = useState("greeting");
  const [skipLog, setSkipLog]         = useState([]);
  const [lastMeta, setLastMeta]       = useState(null);
  const [elapsed, setElapsed]         = useState(0);
  const [prefilledFields, setPrefilledFields] = useState([]);
  const sessionStartRef               = useRef(Date.now());
  const prevTotalRef                  = useRef(null);

  const messagesEndRef  = useRef(null);
  const inputRef        = useRef(null);
  const sessionStarted  = useRef(false);

  const t    = T[getLangKey(lang)];
  const dir  = RTL_LANGS.has(lang) ? "rtl" : "ltr";
  const isAr = RTL_LANGS.has(lang);

  // Derived stats
  const asked       = surveyProgress?.answered ?? 0;
  const pathTotal   = surveyProgress?.total    ?? TOTAL_QUESTIONS;
  const skipped     = Math.max(0, TOTAL_QUESTIONS - pathTotal);
  const remaining   = Math.max(0, pathTotal - asked);
  const reductionPct= Math.round((skipped / TOTAL_QUESTIONS) * 100);
  const curSection  = nextField ? FIELD_TO_SECTION[nextField] : null;

  // Active agents derived from lastMeta + sending
  const activeAgentIds = new Set();
  if (sending) {
    activeAgentIds.add("A1"); activeAgentIds.add("A7"); activeAgentIds.add("A9");
    if (lastMeta?.isco?.length > 0) { activeAgentIds.add("A2"); activeAgentIds.add("A5"); }
    if (lastMeta?.isic)  activeAgentIds.add("A3");
    if (lastMeta?.isced) activeAgentIds.add("A4");
    if (fsmState === "validating")  activeAgentIds.add("A6");
    if (lastMeta?.hitl_required)    activeAgentIds.add("A8");
    if (fsmState === "completing")  activeAgentIds.add("A10");
  }

  // Section status
  const getSectionStatus = (sId) => {
    if (!curSection) return "pending";
    const ci = SECTIONS.findIndex(s => s.id === curSection);
    const ti = SECTIONS.findIndex(s => s.id === sId);
    if (ti < ci)  return "done";
    if (ti === ci) return "active";
    return "pending";
  };

  // Session timer
  useEffect(() => {
    if (completed) return;
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - sessionStartRef.current) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [completed]);

  // Process API response → update all right-panel state
  const processResponse = useCallback((res, prevTotal) => {
    if (res.state) setFsmState(res.state);
    setNextField(res.next_field || null);
    if (res.survey_progress) {
      setSurveyProgress(res.survey_progress);
      const newTotal = res.survey_progress.total;
      if (prevTotal !== null && newTotal < prevTotal) {
        const diff = prevTotal - newTotal;
        const now  = Math.floor((Date.now() - sessionStartRef.current) / 1000);
        setSkipLog(prev => [...prev, {
          time: fmtTime(now),
          msg:  `${diff} question${diff !== 1 ? "s" : ""} skipped — path optimised`,
        }]);
      }
      prevTotalRef.current = newTotal;
    }
  }, []);

  // ── Auth guard + session init ─────────────────────────────────────────────
  useEffect(() => {
    if (sessionStarted.current) return;
    sessionStarted.current = true;

    const storedToken = localStorage.getItem("lfs_token");
    const storedLang  = localStorage.getItem("lfs_lang") || "en";
    if (!storedToken) { router.replace("/"); return; }

    setToken(storedToken);
    setLang(storedLang);

    createSession(storedToken, storedLang)
      .then((session) => {
        setSessionId(session.id);
        if (session.prefilled_fields?.length > 0) {
          setPrefilledFields(session.prefilled_fields);
        }
        setInitialising(false);
        setSending(true);

        sendMessage(storedToken, session.id, "hello", storedLang)
          .then((res) => {
            const meta = buildMeta(res);
            setMessages([{ role: "assistant", text: res.reply, meta }]);
            setLastMeta(meta);
            processResponse(res, null);
            if (res.session_completed) setCompleted(true);
            setTimeout(() => inputRef.current?.focus(), 100);
          })
          .catch(() => setMessages([{ role: "error", text: T[getLangKey(storedLang)].sendError }]))
          .finally(() => setSending(false));
      })
      .catch((err) => {
        const msg = err?.message || "";
        if (msg.includes("expired") || msg.includes("Invalid") || msg.includes("401") || msg.includes("authenticated")) {
          localStorage.removeItem("lfs_token");
          localStorage.removeItem("lfs_lang");
          router.replace("/");
          return;
        }
        setPageError(T[getLangKey(storedLang)].sessionError);
        setInitialising(false);
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // ── Send message ──────────────────────────────────────────────────────────
  const handleSend = useCallback(async (e) => {
    e?.preventDefault();
    const text = input.trim();
    if (!text || sending || completed || !sessionId) return;
    setMessages(prev => [...prev, { role: "user", text }]);
    setInput("");
    setSending(true);
    const prevTotal = prevTotalRef.current;
    try {
      const res  = await sendMessage(token, sessionId, text, lang);
      const meta = buildMeta(res);
      setMessages(prev => [...prev, { role: "assistant", text: res.reply, meta }]);
      setLastMeta(meta);
      processResponse(res, prevTotal);
      if (res.session_completed) {
        setCompleted(true);
        setTimeout(() => router.push(`/report?session=${sessionId}`), 2500);
      }
    } catch {
      setMessages(prev => prev.slice(0, -1));
      setInput(text);
      setMessages(prev => [...prev, { role: "error", text: T[getLangKey(lang)].sendError }]);
      setTimeout(() => setMessages(prev => prev.filter(m => m.role !== "error")), 4000);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }, [input, sending, completed, sessionId, token, lang, processResponse, router]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  // ── Quick reply ───────────────────────────────────────────────────────────
  const handleQuickReply = useCallback((optText) => {
    if (sending || completed || !sessionId) return;
    const text = optText.trim();
    if (!text) return;
    setMessages(prev => [...prev, { role: "user", text }]);
    setNextField(null);
    setSending(true);
    const prevTotal = prevTotalRef.current;
    sendMessage(token, sessionId, text, lang)
      .then((res) => {
        const meta = buildMeta(res);
        setMessages(prev => [...prev, { role: "assistant", text: res.reply, meta }]);
        setLastMeta(meta);
        processResponse(res, prevTotal);
        if (res.session_completed) {
          setCompleted(true);
          setTimeout(() => router.push(`/report?session=${sessionId}`), 2500);
        }
      })
      .catch(() => {
        setMessages(prev => [...prev, { role: "error", text: T[getLangKey(lang)].sendError }]);
        setTimeout(() => setMessages(prev => prev.filter(m => m.role !== "error")), 4000);
      })
      .finally(() => { setSending(false); setInput(""); inputRef.current?.focus(); });
  }, [sending, completed, sessionId, token, lang, processResponse, router]);

  // ── Language change ───────────────────────────────────────────────────────
  const handleLangChange = useCallback((newLang) => {
    setLang(newLang);
    localStorage.setItem("lfs_lang", newLang);
    const hasUserTurn = messages.some(m => m.role === "user");
    if (!hasUserTurn && sessionId && !sending) {
      setMessages([]);
      setSending(true);
      sendMessage(token, sessionId, "hello", newLang)
        .then((res) => {
          const meta = buildMeta(res);
          setMessages([{ role: "assistant", text: res.reply, meta }]);
          setLastMeta(meta);
          processResponse(res, null);
        })
        .catch(() => setMessages([{ role: "error", text: T[getLangKey(newLang)].sendError }]))
        .finally(() => { setSending(false); inputRef.current?.focus(); });
    }
  }, [messages, sessionId, sending, token, processResponse]);

  function handleSignOut() {
    localStorage.removeItem("lfs_token");
    localStorage.removeItem("lfs_lang");
    router.push("/");
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <>
      <Head>
        <title>{t.title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div dir={dir} style={{ fontFamily: SANS, background: "#ffffff", height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* ── Top bar ───────────────────────────────────────────────────── */}
        <header style={{ background: C.white, borderBottom: `1px solid ${C.border}`, flexShrink: 0, zIndex: 10 }}
                className="flex items-center justify-between px-4 py-2.5">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div style={{ background: C.green, fontFamily: MONO, width: 32, height: 32, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 11, fontWeight: 700, flexShrink: 0 }}>
              LFS
            </div>
            <div>
              <span style={{ fontFamily: MONO, color: C.text, fontSize: 13, fontWeight: 600 }}>LFS · AI</span>
              <span style={{ color: C.faint, fontSize: 11, marginLeft: 8 }}>{fsmState.replace(/_/g, " ")}</span>
            </div>
            {/* FSM badge */}
            <FsmPill state={fsmState} />
          </div>
          {/* Right: lang pills + timer + sign out */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-1">
              {LANG_PILLS.map(lp => (
                <button key={lp.code} onClick={() => handleLangChange(lp.code)}
                  style={{
                    fontFamily: MONO, fontSize: 10, fontWeight: 600,
                    padding: "2px 8px", borderRadius: 99,
                    background: lang === lp.code ? C.green : "transparent",
                    color:      lang === lp.code ? "#fff"  : C.muted,
                    border:     lang === lp.code ? `1px solid ${C.green}` : `1px solid ${C.border}`,
                    cursor: "pointer", transition: "all 0.15s",
                  }}>
                  {lp.label}
                </button>
              ))}
            </div>
            <span style={{ fontFamily: MONO, color: C.muted, fontSize: 11 }} className="hidden sm:block">
              {fmtTime(elapsed)}
            </span>
            <button onClick={handleSignOut} style={{ color: C.muted, fontSize: 12 }} className="hover:underline">
              {t.signOut}
            </button>
          </div>
        </header>

        {/* ── Returning-user banner ─────────────────────────────────────── */}
        {prefilledFields.length > 0 && (
          <div style={{ background: C.greenLt, borderBottom: `1px solid ${C.green}30`, flexShrink: 0, padding: "8px 16px" }}>
            <div className="flex items-center gap-2 flex-wrap">
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: C.green, flexShrink: 0 }} />
              <span style={{ fontSize: 11, fontWeight: 600, color: C.green }}>Welcome back —</span>
              <span style={{ fontSize: 11, color: C.green }}>
                {prefilledFields.length} fields loaded from your last survey:
              </span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {prefilledFields.slice(0, 8).map(f => (
                  <span key={f} style={{ fontFamily: MONO, fontSize: 9, fontWeight: 600, background: C.white, border: `1px solid ${C.green}40`, color: C.green, padding: "1px 7px", borderRadius: 99 }}>
                    {PREFILL_LABELS[f] || f}
                  </span>
                ))}
                {prefilledFields.length > 8 && (
                  <span style={{ fontFamily: MONO, fontSize: 9, color: C.green }}>+{prefilledFields.length - 8} more</span>
                )}
              </div>
              <span style={{ fontSize: 11, color: C.green, marginLeft: 4 }}>Just confirm or update any changes.</span>
            </div>
          </div>
        )}

        {/* ── Progress strip ────────────────────────────────────────────── */}
        {surveyProgress && surveyProgress.total > 2 && !completed && (
          <div style={{ background: C.white, borderBottom: `1px solid ${C.border}`, flexShrink: 0 }}
               className="px-4 py-2">
            <div className="max-w-3xl mx-auto flex items-center gap-3">
              <span style={{ color: C.green, fontSize: 11, fontWeight: 600, minWidth: 120 }}>
                {curSection
                  ? (SECTIONS.find(s => s.id === curSection)?.[isAr ? "ar" : "en"] ?? "Survey")
                  : "Survey"}
              </span>
              <div className="flex-1 rounded-full" style={{ height: 5, background: C.border }}>
                <div className="rounded-full" style={{ height: 5, background: C.green, width: `${surveyProgress.pct}%`, transition: "width 0.5s" }} />
              </div>
              <span style={{ fontFamily: MONO, color: C.green, fontSize: 11, fontWeight: 700 }}>{surveyProgress.pct}%</span>
            </div>
          </div>
        )}

        {/* ── LLM processing strip ─────────────────────────────────────── */}
        {sending && (
          <div style={{ background: C.amberBg, borderBottom: `1px solid ${C.amberBd}`, flexShrink: 0 }}
               className="px-4 py-1.5 flex items-center gap-2">
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: C.amber, flexShrink: 0 }} className="animate-pulse" />
            <span style={{ color: "#92400e", fontSize: 11, fontWeight: 500 }}>
              LLM Processing — {LLM_ROLES[fsmState] ?? "Generating response…"}
            </span>
          </div>
        )}

        {/* ── Body: chat + right panel ──────────────────────────────────── */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

          {/* ── Chat column ─────────────────────────────────────────────── */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

            {/* Messages */}
            <main style={{ flex: 1, overflowY: "auto", padding: "16px 16px 8px", background: "#f8fafc" }}>

              {/* Initialising */}
              {initialising && (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ width: 32, height: 32, border: `2px solid ${C.green}`, borderTopColor: "transparent", borderRadius: "50%", margin: "0 auto 12px" }} className="animate-spin" />
                    <p style={{ color: C.muted, fontSize: 13 }}>{t.startingSession}</p>
                  </div>
                </div>
              )}

              {/* Page error */}
              {pageError && (
                <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", color: "#dc2626", borderRadius: 10, padding: "10px 14px", marginBottom: 12, fontSize: 13 }}>
                  {pageError}
                </div>
              )}

              {/* Message list */}
              {!initialising && messages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} lang={lang} t={t} />
              ))}

              {/* Typing indicator */}
              {sending && !initialising && messages.length > 0 && (
                <div style={{ display: "flex", marginBottom: 8 }}>
                  <div style={{ background: C.white, border: `1px solid ${C.border}`, borderLeft: `3px solid ${C.green}`, borderRadius: "0 12px 12px 12px", padding: "10px 14px" }}>
                    <TypingDots />
                  </div>
                </div>
              )}

              {/* Completion */}
              {completed && (
                <div style={{ background: "#f0fdf4", border: "1px solid #86efac", borderRadius: 12, padding: "20px 16px", textAlign: "center", marginTop: 8 }}>
                  <div style={{ fontSize: 28, marginBottom: 8 }}>✓</div>
                  <p style={{ color: "#166534", fontWeight: 600, fontSize: 14 }}>{t.completedBanner}</p>
                  <p style={{ color: C.muted, fontSize: 12, marginTop: 4 }}>Redirecting to report…</p>
                </div>
              )}

              <div ref={messagesEndRef} />
            </main>

            {/* Quick replies */}
            {!completed && !sending && nextField && QUICK_OPTIONS[nextField] && (
              <div style={{ background: C.white, borderTop: `1px solid ${C.border}`, padding: "10px 16px 8px", flexShrink: 0 }}>
                <div className="max-w-3xl mx-auto">
                  <div className={`flex flex-wrap gap-2 ${isAr ? "justify-end" : "justify-start"}`}>
                    {(QUICK_OPTIONS[nextField][getLangKey(lang)] || QUICK_OPTIONS[nextField].en).map(opt => (
                      <button key={opt} onClick={() => handleQuickReply(opt)}
                        style={{ border: `1px solid ${C.green}`, color: C.green, borderRadius: 99, padding: "5px 12px", fontSize: 12, fontWeight: 500, background: C.white, cursor: "pointer", transition: "all 0.15s" }}
                        onMouseEnter={e => { e.target.style.background = C.green; e.target.style.color = "#fff"; }}
                        onMouseLeave={e => { e.target.style.background = C.white; e.target.style.color = C.green; }}>
                        {opt}
                      </button>
                    ))}
                  </div>
                  <p style={{ color: C.faint, fontSize: 11, textAlign: "center", marginTop: 6 }}>{t.orType}</p>
                </div>
              </div>
            )}

            {/* Input bar */}
            <footer style={{ background: C.white, borderTop: `1px solid ${C.border}`, padding: "12px 16px", flexShrink: 0 }}>
              <form onSubmit={handleSend} style={{ display: "flex", gap: 8, alignItems: "flex-end", maxWidth: 768, margin: "0 auto" }}>
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={completed ? "" : t.inputPlaceholder}
                  disabled={completed || sending || initialising || !!pageError}
                  rows={1}
                  style={{
                    flex: 1, fontFamily: SANS, fontSize: 14, padding: "10px 14px",
                    background: "#f9fafb", border: `1px solid ${C.border}`, borderRadius: 12,
                    color: C.text, resize: "none", maxHeight: 120, overflowY: "auto",
                    outline: "none", lineHeight: 1.5,
                    opacity: (completed || sending || initialising || pageError) ? 0.4 : 1,
                  }}
                />
                <button type="submit"
                  disabled={!input.trim() || sending || completed || initialising || !!pageError}
                  style={{
                    background: C.green, color: "#fff", border: "none", borderRadius: 12,
                    padding: "10px 18px", fontFamily: SANS, fontSize: 13, fontWeight: 600,
                    cursor: "pointer", flexShrink: 0, opacity: (!input.trim() || sending || completed || initialising || pageError) ? 0.4 : 1,
                  }}>
                  {sending ? "…" : t.sendButton}
                </button>
              </form>
            </footer>
          </div>

          {/* ── Right panel ─────────────────────────────────────────────── */}
          <aside style={{ width: 296, background: "#ffffff", borderLeft: `1px solid ${C.border}`, flexDirection: "column", overflowY: "auto", flexShrink: 0 }}
                 className="hidden lg:flex">

            {/* ── LIVE QUESTION TRACKER ─────────────────────────────────── */}
            <div style={{ borderBottom: `1px solid ${C.border}`, padding: "14px 16px" }}>
              <p style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>
                Live Question Tracker
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 10 }}>
                <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px", textAlign: "center" }}>
                  <p style={{ fontFamily: MONO, color: C.green, fontSize: 28, fontWeight: 700, lineHeight: 1 }}>{asked}</p>
                  <p style={{ color: C.faint, fontSize: 9, marginTop: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Asked</p>
                </div>
                <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px", textAlign: "center" }}>
                  <p style={{ fontFamily: MONO, color: C.amber, fontSize: 28, fontWeight: 700, lineHeight: 1 }}>{skipped}</p>
                  <p style={{ color: C.faint, fontSize: 9, marginTop: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Skipped</p>
                </div>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ textAlign: "center" }}>
                  <p style={{ fontFamily: MONO, color: "#6366f1", fontSize: 18, fontWeight: 700, lineHeight: 1 }}>{remaining}</p>
                  <p style={{ color: C.faint, fontSize: 9, marginTop: 2 }}>Remaining</p>
                </div>
                <div style={{ width: 1, height: 32, background: C.border }} />
                <div style={{ textAlign: "center" }}>
                  <p style={{ fontFamily: MONO, color: "#ec4899", fontSize: 18, fontWeight: 700, lineHeight: 1 }}>{reductionPct}%</p>
                  <p style={{ color: C.faint, fontSize: 9, marginTop: 2 }}>Reduction</p>
                </div>
                <div style={{ width: 1, height: 32, background: C.border }} />
                <div style={{ textAlign: "center" }}>
                  <p style={{ fontFamily: MONO, color: C.green, fontSize: 18, fontWeight: 700, lineHeight: 1 }}>{skipped}</p>
                  <p style={{ color: C.faint, fontSize: 9, marginTop: 2 }}>Saved</p>
                </div>
              </div>
            </div>

            {/* ── TIME — AI VS TRADITIONAL ──────────────────────────────── */}
            <div style={{ borderBottom: `1px solid ${C.border}`, padding: "14px 16px" }}>
              <p style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>
                Time — AI vs Traditional
              </p>
              <div style={{ marginBottom: 6 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#b91c1c", marginBottom: 3 }}>
                  <span style={{ fontWeight: 500 }}>Traditional</span><span style={{ fontFamily: MONO }}>45–60 min</span>
                </div>
                <div style={{ height: 7, borderRadius: 99, background: "#fee2e2", overflow: "hidden" }}>
                  <div style={{ height: 7, borderRadius: 99, background: "#ef4444", width: "100%" }} />
                </div>
              </div>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: C.green, marginBottom: 3 }}>
                  <span style={{ fontWeight: 500 }}>AI Survey</span>
                  <span style={{ fontFamily: MONO }}>{fmtTime(elapsed)}</span>
                </div>
                <div style={{ height: 7, borderRadius: 99, background: C.greenLt, overflow: "hidden" }}>
                  <div style={{ height: 7, borderRadius: 99, background: C.green, width: `${Math.min(100, (elapsed / 480) * 100)}%`, transition: "width 1s" }} />
                </div>
              </div>
              <div style={{ marginTop: 8, background: C.greenLt, border: `1px solid ${C.green}30`, borderRadius: 6, padding: "5px 10px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 10, color: C.green, fontWeight: 500 }}>You save ~{Math.max(0, 45 - Math.floor(elapsed / 60))} min</span>
                <span style={{ fontFamily: MONO, fontSize: 9, color: C.green }}>{Math.round(Math.min(100, (elapsed / 480) * 100))}% of 8 min</span>
              </div>
            </div>

            {/* ── SECTION STATUS ────────────────────────────────────────── */}
            <div style={{ borderBottom: `1px solid ${C.border}`, padding: "14px 16px" }}>
              <p style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>
                Section Status
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                {SECTIONS.map(sec => {
                  const status = getSectionStatus(sec.id);
                  return (
                    <div key={sec.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 6px", borderRadius: 6, background: status === "active" ? "#f0fdf4" : "transparent" }}>
                      <div style={{
                        fontFamily: MONO, width: 20, height: 20, borderRadius: 4, flexShrink: 0,
                        display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700,
                        background: status === "done" ? C.green : status === "active" ? C.greenLt : C.panel,
                        color: status === "done" ? "#fff" : status === "active" ? C.green : C.faint,
                        border: status === "active" ? `1px solid ${C.green}60` : "none",
                      }}>
                        {status === "done" ? "✓" : sec.id}
                      </div>
                      <span style={{ fontSize: 11, color: status === "done" ? C.muted : status === "active" ? C.text : C.faint, fontWeight: status === "active" ? 600 : 400, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {isAr ? sec.ar : sec.en}
                      </span>
                      {status === "active" && <span style={{ width: 6, height: 6, borderRadius: "50%", background: C.green, flexShrink: 0 }} className="animate-pulse" />}
                      {status === "done"   && <span style={{ width: 6, height: 6, borderRadius: "50%", background: C.faint, flexShrink: 0 }} />}
                    </div>
                  );
                })}
              </div>
              {skipped > 0 && (
                <div style={{ marginTop: 10 }}>
                  <span style={{ background: C.greenLt, border: `1px solid ${C.green}40`, color: C.green, borderRadius: 99, padding: "3px 10px", fontSize: 10, fontWeight: 600 }}>
                    {skipped} questions saved
                  </span>
                </div>
              )}
            </div>

            {/* ── LLM ROLE IN THIS STEP ─────────────────────────────────── */}
            <div style={{ borderBottom: `1px solid ${C.border}`, padding: "14px 16px" }}>
              <p style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
                LLM Role in This Step
              </p>
              <div style={{ background: C.amberBg, border: `1px solid ${C.amberBd}`, borderRadius: 8, padding: "10px 12px" }}>
                <p style={{ fontSize: 9, fontWeight: 700, color: C.amber, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>
                  What Claude is doing now
                </p>
                <p style={{ color: "#92400e", fontSize: 11, lineHeight: 1.5, fontWeight: 500 }}>
                  {LLM_ROLES[fsmState] ?? "Awaiting input"}
                </p>
                {sending && (
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8, paddingTop: 8, borderTop: `1px solid ${C.amberBd}` }}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: C.amber, flexShrink: 0 }} className="animate-pulse" />
                    <span style={{ color: C.amber, fontSize: 10, fontFamily: MONO }}>Processing…</span>
                  </div>
                )}
              </div>
            </div>

            {/* ── AGENT ACTIVATION ──────────────────────────────────────── */}
            <div style={{ padding: "14px 16px" }}>
              <p style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>
                Agent Activation
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {AGENTS.map(ag => {
                  const isActive = activeAgentIds.has(ag.id);
                  const wasDone  = !sending && lastMeta && (
                    ag.id === "A1" ||
                    (ag.id === "A2" && lastMeta.isco?.length > 0) ||
                    (ag.id === "A3" && lastMeta.isic) ||
                    (ag.id === "A4" && lastMeta.isced) ||
                    (ag.id === "A5" && lastMeta.isco?.length > 0) ||
                    ag.id === "A9"
                  );
                  const status = isActive ? "active" : wasDone ? "done" : "idle";
                  return (
                    <div key={ag.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 6px", borderRadius: 6, background: isActive ? "#f0fdf4" : "transparent" }}>
                      <span style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, width: 24, flexShrink: 0,
                        color: status === "active" ? C.green : status === "done" ? C.muted : "#d1d5db" }}>
                        {ag.id}
                      </span>
                      <span style={{ fontSize: 11, flex: 1,
                        color: status === "active" ? C.text : status === "done" ? C.muted : "#d1d5db",
                        fontWeight: status === "active" ? 600 : 400 }}>
                        {ag.label}
                      </span>
                      <span style={{
                        fontFamily: MONO, fontSize: 8, fontWeight: 700, padding: "2px 6px", borderRadius: 99,
                        background: status === "active" ? C.green : status === "done" ? "#f3f4f6" : "#f3f4f6",
                        color:      status === "active" ? "#ffffff" : status === "done" ? C.faint : "#d1d5db",
                        flexShrink: 0,
                      }}
                      className={status === "active" ? "animate-pulse" : ""}>
                        {status === "active" ? "ACTIVE" : status === "done" ? "done" : "idle"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </aside>
        </div>
      </div>
    </>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// ── Validation-summary parsing ──────────────────────────────────────────────
// The VALIDATING-state reply is a bullet list ("• Label: value • Label: value
// ...") followed by a yes/no question. Plain-text chat bubbles collapse the
// backend's line breaks into one dense paragraph, which is unreadable once a
// respondent has 30+ answered fields. Parse the known bullet pattern out and
// render it as an actual table instead; falls back to plain text (returns
// null) for anything that doesn't match, so an LLM-phrased reply (non-
// FAST_MODE) or any other message is never mangled.
const _VALIDATION_QUESTION_ANCHORS = [
  "Is everything correct",
  "هل جميع المعلومات صحيحة",
];

function parseValidationSummary(text) {
  if (!text || !text.includes("•")) return null;

  let anchorIdx = -1;
  let anchor = null;
  for (const a of _VALIDATION_QUESTION_ANCHORS) {
    const idx = text.indexOf(a);
    if (idx !== -1) { anchorIdx = idx; anchor = a; break; }
  }
  if (anchorIdx === -1) return null;

  const body = text.slice(0, anchorIdx).trim();
  const question = text.slice(anchorIdx).trim();

  const firstBullet = body.indexOf("•");
  if (firstBullet === -1) return null;
  const intro = body.slice(0, firstBullet).trim();
  const bulletsText = body.slice(firstBullet);

  const rows = bulletsText
    .split("•")
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => {
      const colonIdx = chunk.indexOf(":");
      if (colonIdx === -1) return null;
      return {
        label: chunk.slice(0, colonIdx).trim(),
        value: chunk.slice(colonIdx + 1).trim(),
      };
    })
    .filter(Boolean);

  if (rows.length === 0) return null;
  return { intro, rows, question };
}

function ValidationSummaryTable({ parsed, isAr }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {parsed.intro && <p style={{ margin: 0 }}>{parsed.intro}</p>}
      <div style={{ border: `1px solid ${C.border}`, borderRadius: 8, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
          <tbody>
            {parsed.rows.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? C.white : C.panel }}>
                <td style={{
                  padding: "6px 10px", color: C.muted, fontWeight: 500,
                  width: "44%", borderTop: i > 0 ? `1px solid ${C.border}` : "none",
                  verticalAlign: "top",
                }}>
                  {row.label}
                </td>
                <td style={{
                  padding: "6px 10px", color: C.text, fontWeight: 600,
                  borderTop: i > 0 ? `1px solid ${C.border}` : "none",
                  borderInlineStart: `1px solid ${C.border}`,
                  verticalAlign: "top", wordBreak: "break-word",
                }}>
                  {row.value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: 0, fontWeight: 500 }}>{parsed.question}</p>
    </div>
  );
}

function buildMeta(res) {
  return {
    state:         res.state,
    detectedLang:  res.detected_language,
    isCodeSwitched:res.is_code_switched,
    entities:      res.entities || [],
    isco:          res.isco_classifications || [],
    isic:          res.isic_classification  || null,
    isced:         res.isced_classification || null,
    nationality:   res.nationality_classification || null,
    latencyMs:     res.latency_ms || null,
    hitl_required: res.isco_classifications?.[0]?.hitl_required || false,
  };
}

// ── ISCO lookup maps ──────────────────────────────────────────────────────────

const ISCO_MAJOR = {
  "0":"Armed Forces","1":"Managers","2":"Professionals","3":"Technicians",
  "4":"Clerical","5":"Service & Sales","6":"Agriculture","7":"Craft & Trade",
  "8":"Operators","9":"Elementary",
};
const ISCO_SUBMAJOR = {
  "01":"Commissioned Armed Forces Officers","02":"Non-commissioned Armed Forces Officers","03":"Armed Forces (Other)",
  "11":"Chief Executives & Senior Officials","12":"Administrative & Commercial Managers",
  "13":"Production & Specialised Services Managers","14":"Hospitality, Retail & Other Services Managers",
  "21":"Science & Engineering Professionals","22":"Health Professionals","23":"Teaching Professionals",
  "24":"Business & Administration Professionals","25":"ICT Professionals","26":"Legal, Social & Cultural Professionals",
  "31":"Science & Engineering Associate Professionals","32":"Health Associate Professionals",
  "33":"Business & Administration Associate Professionals","34":"Legal, Social & Cultural Associate Professionals",
  "35":"ICT Technicians",
  "41":"General & Keyboard Clerks","42":"Customer Services Clerks","43":"Numerical & Material Recording Clerks","44":"Other Clerical Support Workers",
  "51":"Personal Service Workers","52":"Sales Workers","53":"Personal Care Workers","54":"Protective Services Workers",
  "61":"Skilled Agricultural Workers","62":"Skilled Forestry, Fishery & Hunting Workers","63":"Subsistence Farmers, Fishers & Hunters",
  "71":"Building & Related Trades Workers","72":"Metal & Machinery Trades Workers","73":"Handicraft & Printing Workers",
  "74":"Electrical & Electronic Trades Workers","75":"Food Processing, Woodworking & Garment Workers",
  "81":"Stationary Plant & Machine Operators","82":"Assemblers","83":"Drivers & Mobile Plant Operators",
  "91":"Cleaners & Helpers","92":"Agricultural & Fishery Labourers","93":"Mining, Construction & Transport Labourers",
  "94":"Food Preparation Assistants","95":"Street & Related Sales Workers","96":"Refuse Workers & Other Elementary Workers",
};
const ISCO_MINOR = {
  "011":"Commissioned Armed Forces Officers","021":"Non-commissioned Armed Forces Officers","031":"Armed Forces (Other)",
  "111":"Legislators & Senior Officials","112":"Managing Directors & CEOs","121":"Business Services Managers",
  "122":"Sales, Marketing & Development Managers","131":"Agricultural Managers","132":"Manufacturing Managers",
  "133":"ICT Service Managers","134":"Professional Services Managers","141":"Hotel & Restaurant Managers",
  "142":"Retail & Wholesale Trade Managers","143":"Other Services Managers",
  "211":"Physical & Earth Science Professionals","212":"Mathematicians, Actuaries & Statisticians",
  "213":"Life Science Professionals","214":"Engineering Professionals (excl. Electrotechnology)",
  "215":"Electrotechnology Engineers","216":"Architects, Planners, Surveyors & Designers",
  "221":"Medical Doctors","222":"Nursing & Midwifery Professionals","223":"Traditional Medicine Professionals",
  "224":"Paramedical Practitioners","225":"Veterinarians","226":"Other Health Professionals",
  "231":"University & Higher Education Teachers","232":"Vocational Education Teachers",
  "233":"Secondary Education Teachers","234":"Primary School & Early Childhood Teachers","235":"Other Teaching Professionals",
  "241":"Finance Professionals","242":"Administration Professionals","243":"Sales, Marketing & PR Professionals",
  "251":"Software & Applications Developers & Analysts","252":"Database & Network Professionals",
  "261":"Legal Professionals","262":"Librarians, Archivists & Curators","263":"Social & Religious Professionals",
  "264":"Authors, Journalists & Linguists","265":"Creative & Performing Arts Professionals",
  "311":"Physical & Engineering Science Technicians","312":"Mining, Manufacturing & Construction Supervisors",
  "313":"Process Control Technicians","314":"Life Science Technicians","315":"Ship & Aircraft Controllers & Technicians",
  "321":"Medical Imaging & Therapeutic Equipment Technicians","322":"Medical & Pharmaceutical Technicians",
  "323":"Traditional Medicine Associate Professionals","324":"Veterinary Technicians","325":"Other Health Associate Professionals",
  "331":"Financial & Mathematical Associate Professionals","332":"Sales & Purchasing Agents & Brokers",
  "333":"Business Services Agents","334":"Administrative & Executive Secretaries","335":"Regulatory Government Associate Professionals",
  "341":"Legal, Social & Religious Associate Professionals","342":"Sports & Fitness Workers","343":"Artistic, Cultural & Culinary Associate Professionals",
  "351":"ICT Operations & User Support Technicians","352":"Telecommunications & Broadcasting Technicians",
  "411":"General Office Clerks","412":"Secretaries (General)","413":"Keyboard Operators",
  "421":"Tellers, Money Collectors & Related Clerks","422":"Client Information Workers",
  "431":"Numerical Clerks","432":"Material-recording & Transport Clerks","441":"Other Clerical Support Workers",
  "511":"Travel Attendants","512":"Cooks","513":"Waiters & Bartenders","514":"Hairdressers & Beauty Workers",
  "515":"Building Caretakers & Housekeeping","516":"Other Personal Services Workers",
  "521":"Street & Market Salespersons","522":"Shop Salespersons","523":"Cashiers & Ticket Clerks","524":"Other Sales Workers",
  "531":"Child Care Workers & Teachers Aides","532":"Personal Care Workers in Health Services",
  "541":"Protective Services Workers",
  "611":"Market Gardeners & Crop Growers","612":"Animal Producers","613":"Mixed Crop & Animal Producers",
  "621":"Forestry & Related Workers","622":"Fishery Workers, Hunters & Trappers",
  "631":"Subsistence Crop Farmers","632":"Subsistence Livestock Farmers","633":"Subsistence Mixed Farmers","634":"Subsistence Fishers",
  "711":"Building Frame & Related Trades Workers","712":"Building Finishers & Related Trades Workers",
  "713":"Painters & Building Cleaners","721":"Sheet Metal Workers, Moulders & Welders",
  "722":"Blacksmiths, Toolmakers & Related Trades Workers","723":"Machinery Mechanics & Repairers",
  "731":"Handicraft Workers","732":"Printing Trades Workers",
  "741":"Electrical Equipment Installers & Repairers","742":"Electronics & Telecommunications Installers",
  "751":"Food Processing Workers","752":"Wood Treaters & Cabinet Makers","753":"Garment Workers","754":"Other Craft Workers",
  "811":"Mining & Mineral Processing Plant Operators","812":"Metal Processing Plant Operators",
  "815":"Chemical & Photographic Products Plant Operators","816":"Power Production Operators",
  "817":"Automated Assembly Line Operators","818":"Other Stationary Plant Operators",
  "821":"Assemblers","831":"Locomotive Engine Drivers","832":"Car, Taxi & Van Drivers",
  "833":"Bus & Tram Drivers","834":"Heavy Truck & Lorry Drivers","835":"Mobile Plant Operators",
  "911":"Domestic Cleaners & Helpers","912":"Vehicle & Window Cleaners",
  "921":"Agricultural Labourers","931":"Mining & Construction Labourers","932":"Manufacturing Labourers",
  "933":"Transport & Storage Labourers","941":"Food Preparation Assistants",
  "951":"Street & Related Service Workers","961":"Refuse Workers","962":"Other Elementary Workers",
};

const METHOD_LABELS = {
  hierarchical_llm:      { label:"Hierarchical + LLM",   color:"#7c3aed" },
  hierarchical_semantic: { label:"Hierarchical Semantic", color:"#2563eb" },
  flat_llm:              { label:"Flat RAG + LLM",        color:"#d97706" },
  flat_semantic:         { label:"Flat RAG",              color:"#6b7280" },
  cached:                { label:"Cached",                color:"#6b7280" },
};

const LANG_DISPLAY = {
  en:       { flag:"🇬🇧", name:"English" },
  ar:       { flag:"🇦🇪", name:"Arabic (MSA)" },
  "ar-gulf":{ flag:"🇦🇪", name:"Arabic (Gulf)" },
  ur:       { flag:"🇵🇰", name:"Urdu" },
  hi:       { flag:"🇮🇳", name:"Hindi" },
  tl:       { flag:"🇵🇭", name:"Filipino" },
  other:    { flag:"🌐",  name:"Other" },
};

// ── FsmPill ───────────────────────────────────────────────────────────────────

const FSM_COLORS = {
  greeting:       { bg:"#f3f4f6", text:"#6b7280",  dot:"#9ca3af" },
  collecting_info:{ bg:"#eff6ff", text:"#1d4ed8",  dot:"#3b82f6" },
  clarifying:     { bg:"#fffbeb", text:"#d97706",  dot:"#f59e0b" },
  validating:     { bg:"#f5f3ff", text:"#7c3aed",  dot:"#8b5cf6" },
  completing:     { bg:"#f0fdf4", text:"#16a34a",  dot:"#22c55e" },
};

function FsmPill({ state }) {
  if (!state) return null;
  const c = FSM_COLORS[state] || FSM_COLORS.greeting;
  return (
    <span style={{ background: c.bg, color: c.text, border: `1px solid ${c.dot}40`, borderRadius: 99, padding: "2px 8px", fontSize: 10, fontWeight: 600, display: "inline-flex", alignItems: "center", gap: 4 }}
          className="hidden sm:inline-flex">
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: c.dot }} className="animate-pulse" />
      {state.replace(/_/g, " ").toUpperCase()}
    </span>
  );
}

// ── MessageBubble ─────────────────────────────────────────────────────────────

function MessageBubble({ msg, lang, t }) {
  const isUser  = msg.role === "user";
  const isError = msg.role === "error";
  const [panelOpen, setPanelOpen] = useState(false);

  if (isError) {
    return (
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 8 }}>
        <p style={{ background: "#fef2f2", border: "1px solid #fca5a5", color: "#dc2626", borderRadius: 99, padding: "4px 12px", fontSize: 12 }}>
          {msg.text}
        </p>
      </div>
    );
  }

  const hasAiData = !isUser && msg.meta != null;
  const isAr = RTL_LANGS.has(lang);
  const validationSummary =
    !isUser && msg.meta?.state === "validating" ? parseValidationSummary(msg.text) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: isUser ? "flex-end" : "flex-start", gap: 4, marginBottom: 12 }}>

      {/* Bubble */}
      <div style={{
        maxWidth: validationSummary ? "92%" : "78%", padding: "10px 14px", fontSize: 13, lineHeight: 1.55, borderRadius: 14,
        whiteSpace: validationSummary ? "normal" : "pre-wrap",
        ...(isUser
          ? { background: C.green, color: "#fff", borderBottomRightRadius: 4 }
          : { background: C.white, border: `1px solid ${C.border}`, borderLeft: `3px solid ${C.green}`, color: C.text, borderBottomLeftRadius: 4 }),
      }} dir={isAr ? "rtl" : "ltr"}>
        {validationSummary
          ? <ValidationSummaryTable parsed={validationSummary} isAr={isAr} />
          : msg.text}
      </div>

      {/* AI Analysis panel */}
      {hasAiData && (
        <div style={{ width: "100%", maxWidth: "92%" }}>
          <button onClick={() => setPanelOpen(v => !v)}
            style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 10px", background: "#f9fafb", border: `1px solid ${C.border}`, borderRadius: panelOpen ? "8px 8px 0 0" : 8, cursor: "pointer", fontSize: 10 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: C.purple }} className="animate-pulse" />
              <span style={{ color: C.text, fontWeight: 600 }}>AI Analysis Pipeline</span>
              {msg.meta.detectedLang && (
                <span style={{ background: "#f3f4f6", border: `1px solid ${C.border}`, color: C.muted, borderRadius: 99, padding: "1px 7px", fontSize: 9 }}>
                  {LANG_DISPLAY[msg.meta.detectedLang]?.flag} {LANG_DISPLAY[msg.meta.detectedLang]?.name}
                </span>
              )}
              <FsmPill state={msg.meta.state} />
              {msg.meta.isCodeSwitched && (
                <span style={{ background: "#fffbeb", border: "1px solid #fde68a", color: "#d97706", borderRadius: 99, padding: "1px 7px", fontSize: 9 }}>⇄ code-switched</span>
              )}
              {msg.meta.latencyMs && (
                <span style={{ color: C.faint, fontFamily: MONO, fontSize: 9 }}>{msg.meta.latencyMs}ms</span>
              )}
            </span>
            <span style={{ color: C.faint, fontSize: 10 }}>{panelOpen ? "▲" : "▼"}</span>
          </button>

          {panelOpen && (
            <div style={{ border: `1px solid ${C.border}`, borderTop: "none", borderRadius: "0 0 8px 8px", background: C.white, overflow: "hidden" }}>

              {/* NER entities */}
              <div style={{ padding: "10px 12px", borderBottom: `1px solid #f3f4f6` }}>
                <p style={{ fontSize: 9, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>Named Entities (NER)</p>
                {msg.meta.entities?.length > 0 ? (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {msg.meta.entities.map((e, i) => <EntityChip key={i} entity={e} />)}
                  </div>
                ) : (
                  <p style={{ fontSize: 11, color: C.faint, fontStyle: "italic" }}>Short answer — NER skipped</p>
                )}
              </div>

              {/* ISCO */}
              {msg.meta.isco?.length > 0
                ? msg.meta.isco.map((clf, i) => <IscoPanel key={i} clf={clf} lang={lang} />)
                : (
                  <div style={{ padding: "10px 12px", borderBottom: `1px solid #f3f4f6` }}>
                    <p style={{ fontSize: 9, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>ISCO-08 Classification</p>
                    <p style={{ fontSize: 11, color: C.faint, fontStyle: "italic" }}>No job title detected yet — ISCO pipeline will run when you describe your occupation.</p>
                  </div>
                )
              }

              {/* ISIC + ISCED + Nationality */}
              {(msg.meta.isic || msg.meta.isced || msg.meta.nationality) && (
                <div style={{ padding: "10px 12px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
                  {msg.meta.isic       && <IsicCard   isic={msg.meta.isic} />}
                  {msg.meta.isced      && <IscedCard  isced={msg.meta.isced} />}
                  {msg.meta.nationality && <NationalityCard nat={msg.meta.nationality} />}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── IscoPanel ─────────────────────────────────────────────────────────────────

function IscoPanel({ clf, lang }) {
  const title  = lang === "ar" ? clf.primary_title_ar : clf.primary_title_en;
  const pct    = Math.round((clf.confidence || 0) * 100);
  const method = METHOD_LABELS[clf.method] || METHOD_LABELS.flat_semantic;
  const path   = clf.hierarchy_path || [];
  const stages = clf.stage_confidences || {};

  const stageRows = [
    { key:"stage1", code:path[0], label: ISCO_MAJOR[path[0]]    || "Major Group" },
    { key:"stage2", code:path[1], label: ISCO_SUBMAJOR[path[1]] || "Sub-major" },
    { key:"stage3", code:path[2], label: ISCO_MINOR[path[2]]    || "Minor Group" },
    { key:"stage4", code:path[3] || clf.primary_code, label: clf.primary_title_en || "Unit Group" },
  ];

  return (
    <div style={{ padding: "10px 12px", borderBottom: `1px solid #f3f4f6` }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
        <div>
          <p style={{ fontSize: 9, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em" }}>Occupation — ISCO-08</p>
          <p style={{ fontSize: 11, color: C.muted, marginTop: 1 }}>Job title: <span style={{ color: C.text, fontWeight: 600 }}>{clf.job_title}</span></p>
        </div>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
          <span style={{ background: `${method.color}18`, border: `1px solid ${method.color}40`, color: method.color, borderRadius: 99, padding: "2px 8px", fontSize: 9, fontWeight: 600 }}>
            {method.label}
          </span>
          {clf.hitl_required && (
            <span style={{ background: "#fef2f2", border: "1px solid #fca5a5", color: "#dc2626", borderRadius: 99, padding: "2px 8px", fontSize: 9, fontWeight: 600 }}>
              ⚠ HITL Review
            </span>
          )}
        </div>
      </div>

      {/* 4-stage pipeline */}
      {path.length > 0 && stages.stage4 > 0 ? (
        <div style={{ marginBottom: 8 }}>
          <p style={{ fontSize: 9, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>4-Stage Hierarchical Pipeline</p>
          {stageRows.map((row, idx) => {
            const score    = stages[row.key];
            const isUnit   = idx === 3;
            const scorePct = score > 0 ? Math.round(score * 100) : null;
            return (
              <div key={row.key} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 14, flexShrink: 0 }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: isUnit ? C.green : "#d1d5db" }} />
                  {idx < 3 && <div style={{ width: 1, height: 10, background: "#e5e7eb" }} />}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, flex: 1, minWidth: 0 }}>
                  {row.code && <span style={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: isUnit ? C.green : C.faint, flexShrink: 0 }}>{row.code}</span>}
                  <span style={{ fontSize: 11, color: isUnit ? C.text : C.muted, fontWeight: isUnit ? 600 : 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{row.label}</span>
                  {scorePct != null && (
                    <div style={{ display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
                      <div style={{ width: 48, height: 3, background: "#e5e7eb", borderRadius: 99, overflow: "hidden" }}>
                        <div style={{ height: 3, background: isUnit ? C.green : "#d1d5db", width: `${scorePct}%` }} />
                      </div>
                      <span style={{ fontFamily: MONO, fontSize: 9, color: isUnit ? C.green : C.faint, width: 24, textAlign: "right" }}>{scorePct}%</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 6, padding: "6px 10px", marginBottom: 8 }}>
          <p style={{ fontSize: 9, fontWeight: 700, color: "#d97706", textTransform: "uppercase" }}>Flat RAG Fallback</p>
          <p style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>Hierarchical collections not populated. Run <code>python -m backend.rag.load_full_isco</code> to enable.</p>
        </div>
      )}

      {/* Confidence bar */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
          <span style={{ fontSize: 10, color: C.muted }}>Overall confidence</span>
          <span style={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: pct >= 70 ? C.green : C.amber }}>{pct}%</span>
        </div>
        <div style={{ height: 5, background: "#e5e7eb", borderRadius: 99, overflow: "hidden" }}>
          <div style={{ height: 5, background: pct >= 70 ? C.green : C.amber, width: `${pct}%`, borderRadius: 99 }} />
        </div>
      </div>

      {/* Primary result */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, background: C.greenLt, border: `1px solid ${C.green}30`, borderRadius: 8, padding: "7px 10px" }}>
        <span style={{ fontFamily: MONO, fontWeight: 700, color: C.green, fontSize: 14 }}>{clf.primary_code}</span>
        <span style={{ fontSize: 12, color: C.text, fontWeight: 500 }}>{title}</span>
      </div>

      {/* Alternatives */}
      {clf.alternatives?.length > 0 && (
        <div style={{ marginTop: 6 }}>
          <p style={{ fontSize: 9, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>Alternatives</p>
          {clf.alternatives.slice(0, 3).map((alt, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "#f9fafb", border: `1px solid ${C.border}`, borderRadius: 6, padding: "4px 8px", marginBottom: 3 }}>
              <div style={{ display: "flex", gap: 6, minWidth: 0 }}>
                <span style={{ fontFamily: MONO, fontSize: 11, color: C.muted, flexShrink: 0 }}>{alt.code}</span>
                <span style={{ fontSize: 11, color: C.muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {(lang === "ar" ? alt.title_ar : alt.title_en) || ""}
                </span>
              </div>
              <span style={{ fontFamily: MONO, fontSize: 9, color: C.faint, flexShrink: 0 }}>{Math.round(alt.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── IsicCard ──────────────────────────────────────────────────────────────────

function IsicCard({ isic }) {
  return (
    <div>
      <p style={{ fontSize: 9, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 5 }}>ISIC Rev.4 Industry</p>
      <div style={{ background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: 8, padding: "8px 10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 2 }}>
          <span style={{ fontFamily: MONO, fontWeight: 700, color: "#ea580c", fontSize: 12 }}>§ {isic.section}</span>
          <span style={{ fontSize: 11, color: "#c2410c", fontWeight: 500 }}>{isic.section_title}</span>
        </div>
        <div style={{ paddingLeft: 8, borderLeft: "2px solid #fed7aa" }}>
          <p style={{ fontSize: 10, color: "#92400e" }}>Div {isic.division_code} — {isic.division_title}</p>
          {isic.group_code && <p style={{ fontSize: 10, color: "#78350f" }}>Grp {isic.group_code} — {isic.group_title}</p>}
          {isic.class_code && <p style={{ fontSize: 11, fontWeight: 700, color: "#ea580c", marginTop: 2 }}>{isic.class_code} {isic.class_title}</p>}
        </div>
        <ConfBar pct={Math.round((isic.confidence || 0) * 100)} color="#ea580c" />
      </div>
    </div>
  );
}

// ── IscedCard ─────────────────────────────────────────────────────────────────

function IscedCard({ isced }) {
  return (
    <div>
      <p style={{ fontSize: 9, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 5 }}>ISCED Education</p>
      <div style={{ background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 8, padding: "8px 10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 4 }}>
          <span style={{ fontFamily: MONO, fontWeight: 700, color: "#2563eb", fontSize: 12 }}>Level {isced.level}</span>
          <span style={{ fontSize: 11, color: "#1d4ed8", fontWeight: 500 }}>{isced.level_title}</span>
        </div>
        {isced.broad_code && (
          <div style={{ paddingLeft: 8, borderLeft: "2px solid #bfdbfe" }}>
            <p style={{ fontSize: 10, color: "#3730a3" }}>{isced.broad_code} {isced.broad_title}</p>
            {isced.narrow_code && <p style={{ fontSize: 10, color: "#4338ca" }}>{isced.narrow_code} {isced.narrow_title}</p>}
            {isced.detailed_code && <p style={{ fontSize: 11, fontWeight: 700, color: "#2563eb", marginTop: 2 }}>{isced.detailed_code} {isced.detailed_title}</p>}
          </div>
        )}
        <ConfBar pct={Math.round((isced.confidence || 0) * 100)} color="#2563eb" />
      </div>
    </div>
  );
}

// ── NationalityCard ───────────────────────────────────────────────────────────

function NationalityCard({ nat }) {
  return (
    <div>
      <p style={{ fontSize: 9, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 5 }}>UN M49 Nationality</p>
      <div style={{ background: C.greenLt, border: `1px solid ${C.green}40`, borderRadius: 8, padding: "8px 10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ fontFamily: MONO, fontWeight: 700, color: C.green, fontSize: 13 }}>{nat.iso_alpha3}</span>
          <span style={{ fontSize: 10, color: C.muted }}>M49:{nat.m49_code}</span>
        </div>
        <p style={{ fontSize: 12, color: C.text, fontWeight: 500, marginTop: 2 }}>{nat.country_en}</p>
        <p style={{ fontSize: 10, color: C.muted }}>{nat.region_en}</p>
        <ConfBar pct={Math.round((nat.confidence || 0) * 100)} color={C.green} />
      </div>
    </div>
  );
}

// ── EntityChip ────────────────────────────────────────────────────────────────

const ENTITY_COLORS = {
  JOB_TITLE:         { bg:"#f5f3ff", bd:"#e9d5ff", text:"#7c3aed" },
  ORGANIZATION:      { bg:"#fefce8", bd:"#fde68a", text:"#d97706" },
  LOCATION:          { bg:C.greenLt, bd:`${C.green}40`, text:C.green },
  INDUSTRY:          { bg:"#fff7ed", bd:"#fed7aa", text:"#ea580c" },
  EMPLOYMENT_STATUS: { bg:"#f0fdfa", bd:"#99f6e4", text:"#0d9488" },
  DURATION:          { bg:"#fdf4ff", bd:"#f0abfc", text:"#a21caf" },
  HOURS:             { bg:"#eef2ff", bd:"#c7d2fe", text:"#4338ca" },
  PERSON:            { bg:"#f9fafb", bd:C.border,  text:C.muted   },
};

function EntityChip({ entity }) {
  const c = ENTITY_COLORS[entity.label] || ENTITY_COLORS.PERSON;
  return (
    <span style={{ background: c.bg, border: `1px solid ${c.bd}`, color: c.text, borderRadius: 99, padding: "2px 8px", fontSize: 11, display: "inline-flex", alignItems: "center", gap: 4 }}>
      <span style={{ fontWeight: 500 }}>{entity.text}</span>
      <span style={{ opacity: 0.5, fontSize: 9, textTransform: "uppercase", letterSpacing: "0.05em" }}>{entity.label}</span>
    </span>
  );
}

// ── ConfBar ───────────────────────────────────────────────────────────────────

function ConfBar({ pct, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
      <div style={{ flex: 1, height: 3, background: "#e5e7eb", borderRadius: 99, overflow: "hidden" }}>
        <div style={{ height: 3, background: color, width: `${pct}%` }} />
      </div>
      <span style={{ fontFamily: MONO, fontSize: 9, color, fontWeight: 700 }}>{pct}%</span>
    </div>
  );
}

// ── TypingDots ────────────────────────────────────────────────────────────────

function TypingDots() {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4, height: 16 }}>
      {[0, 150, 300].map(d => (
        <span key={d} style={{ width: 6, height: 6, borderRadius: "50%", background: C.green, display: "inline-block" }}
              className="animate-bounce"
              /* inline delay via style is not supported in tailwind animate; use a wrapper trick */
              />
      ))}
    </span>
  );
}
