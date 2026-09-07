/**
 * Full Questionnaire Reference — all 56 real fields, in all 5 supported
 * languages. Content is a condensed single-line paraphrase (question +
 * parenthetical options) of the real, verbatim, multi-line question text in
 * backend/agents/conversation_manager.py's _EXACT_QUESTIONS_EN/AR/UR/HI/TL
 * dicts, and of _get_field_order()'s skip-logic (all checked directly
 * against the live source, 2026-08-29). The Urdu/Hindi/Tagalog wording here
 * is machine-drafted and has not been reviewed by a native speaker — same
 * caveat as the backend dicts it's condensed from.
 */

import Head from "next/head";
import Link from "next/link";
import { useState } from "react";

const C = {
  green:   "#007a62",
  greenLt: "#e6f4f1",
  greenDk: "#005a48",
  white:   "#ffffff",
  bg:      "#f8fafc",
  border:  "#e5e7eb",
  text:    "#1a1a2e",
  muted:   "#6b7280",
  faint:   "#9ca3af",
  amber:   "#d97706",
  amberBg: "#fefce8",
};
const MONO = "'DM Mono', monospace";
const SANS = "'Inter', sans-serif";

// Real question text, verbatim from backend/agents/conversation_manager.py's
// _EXACT_QUESTIONS_EN dict (56 keys, checked 2026-08-29). This is the FAST_MODE
// dev-stub wording — the production LLM path may phrase questions slightly
// differently in conversation, but asks for the same 56 fields.
const Q = {
  en: {
  employment_status: "What is your current employment status? (Employed / Unemployed / Not in the labour force)",
  education_level: "What is your highest level of education completed? (No formal education / Primary / Secondary / Diploma / Bachelor's degree / Master's degree / PhD or higher)",
  field_of_study: "What was your main field of study? (e.g. Engineering, Business, Medicine, Computer Science, Arts)",
  gender: "What is your gender? (Male / Female / Prefer not to say)",
  nationality: "What is your nationality or country of origin? (e.g. Indian, Pakistani, Filipino, Emirati, Egyptian, British, American)",
  marital_status: "What is your marital status? (Single / Married / Divorced / Widowed)",
  emirate: "Which Emirate do you currently reside in? (Abu Dhabi / Dubai / Sharjah / Ajman / Umm Al Quwain / Ras Al Khaimah / Fujairah)",
  uae_residence_duration: "How long have you been residing in the UAE? (Born in UAE / <1 year / 1–4 years / 5–9 years / 10–19 years / 20+ years)",
  vocational_training: "Have you completed any vocational training or professional certifications in the last 12 months? (Yes / No)",
  employment_nature: "What best describes your employment arrangement? (Paid employee / Employer / Self-employed / Contributing family worker)",
  employment_sector: "Which sector does your employer belong to? (Government / Private / Semi-government / Non-profit / NGO)",
  job_title: "What is your job title?",
  job_duties: "Briefly describe your main tasks and duties in that role.",
  industry: "Which industry or sector do you work in? (e.g. technology, healthcare, airlines, government, finance, education)",
  actual_hours_worked: "How many hours did you actually work in your main job during the past week?",
  hours_per_week: "How many hours do you usually work per week?",
  secondary_job: "Do you have any other job or business in addition to your main job? (Yes / No)",
  secondary_job_hours: "How many hours did you work in your secondary job(s) last week?",
  underemployment: "Do you want to work more hours than you currently do and are you available for additional work? (Yes — I want more hours / No / Already overemployed)",
  employment_type: "What is your employment arrangement? (Full-time / Part-time / Seasonal / Casual)",
  contract_type: "Is your employment contract permanent or temporary? (Permanent / Fixed-term <1yr / Fixed-term 1–3yrs / Probation / No written contract)",
  remote_work: "Do you primarily work from home (remote/telework)? (Always / Mostly / Partially / Never)",
  monthly_wage_range: "What is your approximate monthly salary range? (in AED) (<5,000 / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / >50,000 / Prefer not to say)",
  salary_allowances: "Does your salary include any of the following allowances? (Housing / Transport / Food / Schooling / None — choose all that apply)",
  bonuses: "Did you receive any bonuses or incentives in the past 12 months? (Annual bonus / Performance bonus / Other / No)",
  health_insurance: "Does your employer provide health insurance? (Full coverage / Partial coverage / No / I pay for my own)",
  pension_scheme: "Are you enrolled in a pension fund or end-of-service gratuity scheme? (GPSSA / DIFC-ADGM scheme / Employer private scheme / No / Not sure)",
  qualification_match: "Do you feel your qualifications match your current job requirements? (Overqualified / Well matched / Underqualified)",
  job_search_active: "Over the past four weeks, have you been actively looking for a job? (Yes / No)",
  job_search_methods: "What methods have you used to search for employment? (please describe)",
  available_for_work: "If a suitable job were offered today, would you be available to start work within the next two weeks? (Yes / No)",
  unemployment_duration: "How long have you been looking for work?",
  desired_job_type: "What type of work are you looking for? (Same as previous occupation / A different occupation / This will be my first job)",
  ever_worked: "Have you ever worked before? (Yes — last job in the UAE / Yes — last job outside the UAE / No — never worked)",
  last_job_title: "What was your most recent job title? (If never worked, say 'never worked'.)",
  last_job_sector: "Which sector was your last job in? (Government / Private / Semi-government / Non-profit / NGO / Self-employed)",
  reason_left_job: "Why did you leave or lose your last job? (Made redundant / Resigned / Business closed / Contract ended / Other)",
  highest_previous_salary: "What was the highest monthly salary you received in your previous employment? (in AED)",
  outside_lf_reason: "What is the main reason you are not currently seeking work? (Retired / Student / Homemaker / Discouraged worker / Illness or disability / Other)",
  main_skills: "What are your main work-related skills? (choose up to 3 — Digital/IT / Management / Finance / Engineering / Healthcare / Education / Trades / Hospitality / Sales / Other)",
  training_participation: "Did you participate in any vocational or professional training in the past 12 months? (Employer-funded / Self-funded / Government program / No)",
  emiratization_program: "Are you registered in any Emiratization program (NAFIS, Tawteen, Absher)? (Yes — NAFIS / Yes — other / No)",
  labour_market_barriers: "What are the main barriers you face in the labour market? (choose all that apply)",
  platform_work: "Do you work through digital platforms (e.g. Uber, Careem, Talabat, Freelancer, Upwork)? (Primary income / Supplementary income / No)",
  platform_names: "Which platform(s) do you work through? (Ride-hailing / Food delivery / Freelance / Professional services / E-commerce / Other)",
  platform_hours: "How many hours per week do you spend working through these platforms?",
  online_business: "Do you own or manage any online business or e-commerce activity? (Registered business / Informal / No)",
  job_satisfaction: "Overall, how satisfied are you with your current job? (Rate 1–5)",
  work_safety: "Do you work in a safe and healthy work environment? (Always / Mostly / Sometimes / Rarely / Never)",
  workplace_issues: "Have you experienced any of the following at your workplace in the past 12 months? (Harassment / Discrimination / Wage theft / Contract violation / None)",
  work_life_balance: "Do you feel you have an appropriate work-life balance? (Yes / Somewhat / No)",
  question_clarity: "How would you rate the clarity of the questions in this interview? (Rate 1–5)",
  difficulty_answering: "Did you encounter difficulty answering any specific questions? (No / Yes — please specify)",
  ai_preference: "Do you prefer conducting surveys through an AI assistant like this one compared to a human interviewer? (Prefer AI / Prefer human / No preference)",
  data_confidence: "How confident are you that your data is kept private and confidential in this survey? (Very / Somewhat / Not confident)",
  survey_comments: "Do you have any comments or suggestions to improve this survey? (optional)",
  },
  ar: {
  employment_status: "ما هي حالتك الوظيفية الحالية؟ (موظف / عاطل عن العمل / خارج القوى العاملة)",
  education_level: "ما أعلى مستوى تعليمي أتممته؟ (بدون تعليم رسمي / ابتدائي / ثانوي / دبلوم / بكالوريوس / ماجستير / دكتوراه أو أعلى)",
  field_of_study: "ما كان مجال تخصصك الدراسي الرئيسي؟ (مثال: هندسة، إدارة أعمال، طب، علوم حاسوب، آداب)",
  gender: "ما هو جنسك؟ (ذكر / أنثى / أفضل عدم الإفصاح)",
  nationality: "ما هي جنسيتك أو بلدك الأصلي؟ (مثال: هندي، باكستاني، فلبيني، إماراتي، مصري، بريطاني، أمريكي)",
  marital_status: "ما هي حالتك الاجتماعية؟ (أعزب / متزوج / مطلق / أرمل)",
  emirate: "في أي إمارة تقيم حاليًا؟ (أبوظبي / دبي / الشارقة / عجمان / أم القيوين / رأس الخيمة / الفجيرة)",
  uae_residence_duration: "منذ متى وأنت مقيم في الإمارات؟ (مولود في الإمارات / أقل من سنة / 1–4 سنوات / 5–9 سنوات / 10–19 سنة / 20+ سنة)",
  vocational_training: "هل أكملت أي تدريب مهني أو شهادات احترافية في آخر 12 شهرًا؟ (نعم / لا)",
  employment_nature: "ما الذي يصف ترتيب عملك بشكل أفضل؟ (موظف بأجر / صاحب عمل / عمل حر / عامل أسري مساهم)",
  employment_sector: "إلى أي قطاع ينتمي صاحب عملك؟ (حكومي / خاص / شبه حكومي / غير ربحي / منظمة غير حكومية)",
  job_title: "ما هو مسماك الوظيفي؟",
  job_duties: "صف بإيجاز مهامك وواجباتك الرئيسية في هذا الدور.",
  industry: "في أي صناعة أو قطاع تعمل؟ (مثال: تكنولوجيا، رعاية صحية، طيران، حكومة، مالية، تعليم)",
  actual_hours_worked: "كم ساعة عملت فعليًا في وظيفتك الرئيسية خلال الأسبوع الماضي؟",
  hours_per_week: "كم ساعة تعمل عادةً في الأسبوع؟",
  secondary_job: "هل لديك وظيفة أو عمل آخر إضافي إلى وظيفتك الرئيسية؟ (نعم / لا)",
  secondary_job_hours: "كم ساعة عملت في وظيفتك (وظائفك) الثانوية الأسبوع الماضي؟",
  underemployment: "هل ترغب في العمل ساعات أكثر مما تعمل حاليًا وهل أنت متاح لعمل إضافي؟ (نعم — أريد ساعات أكثر / لا / أعمل بالفعل ساعات زائدة)",
  employment_type: "ما هو ترتيب عملك؟ (دوام كامل / دوام جزئي / موسمي / عرضي)",
  contract_type: "هل عقد عملك دائم أم مؤقت؟ (دائم / محدد المدة أقل من سنة / محدد المدة 1–3 سنوات / فترة تجربة / بدون عقد مكتوب)",
  remote_work: "هل تعمل بشكل أساسي من المنزل (عن بُعد)؟ (دائمًا / غالبًا / جزئيًا / أبدًا)",
  monthly_wage_range: "ما هو نطاق راتبك الشهري التقريبي؟ (بالدرهم) (أقل من 5,000 / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / أكثر من 50,000 / أفضل عدم الإفصاح)",
  salary_allowances: "هل يتضمن راتبك أيًا من البدلات التالية؟ (سكن / مواصلات / طعام / تعليم / لا شيء — اختر كل ما ينطبق)",
  bonuses: "هل حصلت على أي مكافآت أو حوافز في آخر 12 شهرًا؟ (مكافأة سنوية / مكافأة أداء / أخرى / لا)",
  health_insurance: "هل يوفر صاحب عملك تأمينًا صحيًا؟ (تغطية كاملة / تغطية جزئية / لا / أدفع بنفسي)",
  pension_scheme: "هل أنت مسجل في صندوق تقاعد أو نظام مكافأة نهاية الخدمة؟ (GPSSA / نظام DIFC-ADGM / نظام خاص من صاحب العمل / لا / غير متأكد)",
  qualification_match: "هل تشعر أن مؤهلاتك تتطابق مع متطلبات وظيفتك الحالية؟ (مؤهل أعلى / متطابق جيدًا / مؤهل أقل)",
  job_search_active: "خلال الأسابيع الأربعة الماضية، هل كنت تبحث بنشاط عن عمل؟ (نعم / لا)",
  job_search_methods: "ما هي الطرق التي استخدمتها للبحث عن عمل؟ (يرجى الوصف)",
  available_for_work: "إذا عُرضت عليك وظيفة مناسبة اليوم، هل ستكون متاحًا لبدء العمل خلال الأسبوعين القادمين؟ (نعم / لا)",
  unemployment_duration: "منذ متى وأنت تبحث عن عمل؟",
  desired_job_type: "ما نوع العمل الذي تبحث عنه؟ (نفس مهنتي السابقة / مهنة مختلفة / ستكون هذه أول وظيفة لي)",
  ever_worked: "هل سبق لك العمل من قبل؟ (نعم — آخر وظيفة في الإمارات / نعم — آخر وظيفة خارج الإمارات / لا — لم أعمل من قبل)",
  last_job_title: "ما كان مسماك الوظيفي الأخير؟ (إذا لم تعمل من قبل، قل 'لم أعمل من قبل'.)",
  last_job_sector: "في أي قطاع كانت وظيفتك الأخيرة؟ (حكومي / خاص / شبه حكومي / غير ربحي / منظمة غير حكومية / عمل حر)",
  reason_left_job: "لماذا تركت أو فقدت وظيفتك الأخيرة؟ (تسريح / استقالة / إغلاق العمل / انتهاء العقد / أخرى)",
  highest_previous_salary: "ما هو أعلى راتب شهري تلقيته في وظيفتك السابقة؟ (بالدرهم)",
  outside_lf_reason: "ما هو السبب الرئيسي لعدم بحثك عن عمل حاليًا؟ (متقاعد / طالب / ربة/رب منزل / عامل محبط / مرض أو إعاقة / أخرى)",
  main_skills: "ما هي مهاراتك الأساسية المتعلقة بالعمل؟ (اختر حتى 3 — رقمي/تقنية معلومات / إدارة / مالية / هندسة / رعاية صحية / تعليم / حرف / ضيافة / مبيعات / أخرى)",
  training_participation: "هل شاركت في أي تدريب مهني أو احترافي في آخر 12 شهرًا؟ (ممول من صاحب العمل / ممول ذاتيًا / برنامج حكومي / لا)",
  emiratization_program: "هل أنت مسجل في أي برنامج توطين (نافس، توطين، أبشر)؟ (نعم — نافس / نعم — أخرى / لا)",
  labour_market_barriers: "ما هي أهم العوائق التي تواجهها في سوق العمل؟ (اختر كل ما ينطبق)",
  platform_work: "هل تعمل عبر منصات رقمية (مثل أوبر، كريم، طلبات، فريلانسر، أب وورك)؟ (دخل رئيسي / دخل إضافي / لا)",
  platform_names: "ما هي المنصة (المنصات) التي تعمل من خلالها؟ (توصيل ركاب / توصيل طعام / عمل حر / خدمات احترافية / تجارة إلكترونية / أخرى)",
  platform_hours: "كم ساعة أسبوعيًا تقضيها في العمل عبر هذه المنصات؟",
  online_business: "هل تمتلك أو تدير أي نشاط تجاري إلكتروني؟ (نشاط مسجل / غير رسمي / لا)",
  job_satisfaction: "بشكل عام، ما مدى رضاك عن وظيفتك الحالية؟ (قيّم من 1–5)",
  work_safety: "هل تعمل في بيئة عمل آمنة وصحية؟ (دائمًا / غالبًا / أحيانًا / نادرًا / أبدًا)",
  workplace_issues: "هل واجهت أيًا مما يلي في مكان عملك خلال آخر 12 شهرًا؟ (تحرش / تمييز / سرقة أجر / خرق العقد / لا شيء)",
  work_life_balance: "هل تشعر أن لديك توازنًا مناسبًا بين العمل والحياة؟ (نعم / نوعًا ما / لا)",
  question_clarity: "كيف تقيّم وضوح الأسئلة في هذه المقابلة؟ (قيّم من 1–5)",
  difficulty_answering: "هل واجهت صعوبة في الإجابة عن أي أسئلة معينة؟ (لا / نعم — يرجى التحديد)",
  ai_preference: "هل تفضل إجراء الاستطلاعات عبر مساعد ذكاء اصطناعي مثل هذا مقارنة بمحاور بشري؟ (أفضل الذكاء الاصطناعي / أفضل الإنسان / لا تفضيل)",
  data_confidence: "ما مدى ثقتك بأن بياناتك ستبقى خاصة وسرية في هذا الاستطلاع؟ (واثق جدًا / نوعًا ما / غير واثق)",
  survey_comments: "هل لديك أي تعليقات أو اقتراحات لتحسين هذا الاستطلاع؟ (اختياري)",
  },
  ur: {
  employment_status: "آپ کی موجودہ ملازمت کی صورتحال کیا ہے؟ (ملازم / بے روزگار / لیبر فورس سے باہر)",
  education_level: "آپ کی حاصل کردہ تعلیم کی اعلیٰ ترین سطح کیا ہے؟ (کوئی باضابطہ تعلیم نہیں / ابتدائی / ثانوی / ڈپلومہ / بیچلر ڈگری / ماسٹر ڈگری / پی ایچ ڈی یا اس سے اعلیٰ)",
  field_of_study: "آپ کا بنیادی مضمون تعلیم کیا تھا؟ (مثلاً انجینئرنگ، بزنس، طب، کمپیوٹر سائنس، آرٹس)",
  gender: "آپ کی صنف کیا ہے؟ (مرد / عورت / بتانا نہیں چاہتے)",
  nationality: "آپ کی قومیت یا اصل ملک کیا ہے؟ (مثلاً بھارتی، پاکستانی، فلپائنی، اماراتی، مصری، برطانوی، امریکی)",
  marital_status: "آپ کی ازدواجی حیثیت کیا ہے؟ (غیر شادی شدہ / شادی شدہ / طلاق یافتہ / بیوہ)",
  emirate: "آپ فی الحال کس امارات میں رہائش پذیر ہیں؟ (ابوظہبی / دبئی / شارجہ / عجمان / ام القوین / رأس الخیمہ / فجیرہ)",
  uae_residence_duration: "آپ کتنے عرصے سے متحدہ عرب امارات میں مقیم ہیں؟ (امارات میں پیدا ہوئے / 1 سال سے کم / 1–4 سال / 5–9 سال / 10–19 سال / 20 سال یا زیادہ)",
  vocational_training: "کیا آپ نے گزشتہ 12 مہینوں میں کوئی پیشہ ورانہ تربیت یا سرٹیفیکیشن مکمل کی ہے؟ (ہاں / نہیں)",
  employment_nature: "آپ کے روزگار کے انتظام کو بہترین طور پر کیا بیان کرتا ہے؟ (تنخواہ دار ملازم / آجر / خود روزگار / خاندانی کاروبار میں معاون کارکن)",
  employment_sector: "آپ کا آجر کس شعبے سے تعلق رکھتا ہے؟ (سرکاری / نجی / نیم سرکاری / غیر منافع بخش / این جی او)",
  job_title: "آپ کا جاب ٹائٹل کیا ہے؟",
  job_duties: "براہ کرم مختصراً اپنے اس کردار میں اپنے بنیادی کام اور فرائض بیان کریں۔",
  industry: "آپ کس صنعت یا شعبے میں کام کرتے ہیں؟ (مثلاً ٹیکنالوجی، صحت کی دیکھ بھال، ایئر لائنز، حکومت، مالیات، تعلیم)",
  actual_hours_worked: "گزشتہ ہفتے آپ نے اپنی بنیادی نوکری میں اصل میں کتنے گھنٹے کام کیا؟",
  hours_per_week: "آپ عام طور پر ہفتے میں کتنے گھنٹے کام کرتے ہیں؟",
  secondary_job: "کیا آپ کی اپنی بنیادی نوکری کے علاوہ کوئی اور نوکری یا کاروبار ہے؟ (ہاں / نہیں)",
  secondary_job_hours: "گزشتہ ہفتے آپ نے اپنی ثانوی نوکری/نوکریوں میں کتنے گھنٹے کام کیا؟",
  underemployment: "کیا آپ اپنے موجودہ گھنٹوں سے زیادہ کام کرنا چاہتے ہیں اور کیا آپ اضافی کام کے لیے دستیاب ہیں؟ (ہاں — مجھے زیادہ گھنٹے چاہئیں / نہیں / پہلے ہی ضرورت سے زیادہ کام کر رہا ہوں)",
  employment_type: "آپ کا روزگار کا انتظام کیا ہے؟ (کل وقتی / جز وقتی / موسمی / عارضی)",
  contract_type: "کیا آپ کا ملازمتی معاہدہ مستقل ہے یا عارضی؟ (مستقل / مقررہ مدت 1 سال سے کم / مقررہ مدت 1–3 سال / آزمائشی مدت / کوئی تحریری معاہدہ نہیں)",
  remote_work: "کیا آپ بنیادی طور پر گھر سے کام کرتے ہیں؟ (ہمیشہ / زیادہ تر / جزوی طور پر / کبھی نہیں)",
  monthly_wage_range: "آپ کی تقریباً ماہانہ تنخواہ کی حد کیا ہے؟ (درہم میں) (5,000 سے کم / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / 50,000 سے زیادہ / بتانا نہیں چاہتے)",
  salary_allowances: "کیا آپ کی تنخواہ میں مندرجہ ذیل میں سے کوئی الاؤنس شامل ہے؟ (رہائش / آمد و رفت / کھانا / تعلیم / کوئی نہیں — جو لاگو ہو منتخب کریں)",
  bonuses: "کیا آپ کو گزشتہ 12 مہینوں میں کوئی بونس یا مراعات ملی؟ (سالانہ بونس / کارکردگی بونس / دیگر / نہیں)",
  health_insurance: "کیا آپ کا آجر صحت کی انشورنس فراہم کرتا ہے؟ (مکمل کوریج / جزوی کوریج / نہیں / میں خود ادائیگی کرتا ہوں)",
  pension_scheme: "کیا آپ کسی پنشن فنڈ یا گریجویٹی اسکیم میں شامل ہیں؟ (GPSSA / DIFC-ADGM اسکیم / آجر کی نجی اسکیم / نہیں / یقین نہیں)",
  qualification_match: "کیا آپ محسوس کرتے ہیں کہ آپ کی قابلیت آپ کی موجودہ نوکری کی ضروریات سے مماثل ہے؟ (زیادہ قابل / اچھی طرح مماثل / کم قابل)",
  job_search_active: "گزشتہ چار ہفتوں کے دوران، کیا آپ فعال طور پر نوکری تلاش کر رہے ہیں؟ (ہاں / نہیں)",
  job_search_methods: "آپ نے روزگار تلاش کرنے کے لیے کون سے طریقے استعمال کیے ہیں؟ (براہ کرم بیان کریں)",
  available_for_work: "اگر آج کوئی مناسب نوکری پیش کی جائے تو کیا آپ اگلے دو ہفتوں میں کام شروع کرنے کے لیے دستیاب ہوں گے؟ (ہاں / نہیں)",
  unemployment_duration: "آپ کتنے عرصے سے کام تلاش کر رہے ہیں؟",
  desired_job_type: "آپ کس قسم کا کام تلاش کر رہے ہیں؟ (میرے پچھلے پیشے جیسا / ایک مختلف پیشہ / یہ میری پہلی نوکری ہوگی)",
  ever_worked: "کیا آپ نے کبھی کام کیا ہے؟ (ہاں — آخری نوکری متحدہ عرب امارات میں / ہاں — آخری نوکری متحدہ عرب امارات سے باہر / نہیں — کبھی کام نہیں کیا)",
  last_job_title: "آپ کی سب سے حالیہ جاب ٹائٹل کیا تھی؟ (اگر کبھی کام نہیں کیا تو 'کبھی کام نہیں کیا' کہیں۔)",
  last_job_sector: "آپ کی آخری نوکری کس شعبے میں تھی؟ (سرکاری / نجی / نیم سرکاری / غیر منافع بخش / این جی او / خود روزگار)",
  reason_left_job: "آپ نے اپنی آخری نوکری کیوں چھوڑی یا کھو دی؟ (نوکری ختم کر دی گئی / استعفیٰ دیا / کاروبار بند ہو گیا / معاہدہ ختم ہوا / دیگر)",
  highest_previous_salary: "آپ کی پچھلی ملازمت میں سب سے زیادہ ماہانہ تنخواہ کیا تھی؟ (درہم میں)",
  outside_lf_reason: "آپ فی الحال کام تلاش نہ کرنے کی بنیادی وجہ کیا ہے؟ (ریٹائرڈ / طالب علم / گھریلو خاتون/خاوند / دلبرداشتہ کارکن / بیماری یا معذوری / دیگر)",
  main_skills: "آپ کی بنیادی کام سے متعلق مہارتیں کیا ہیں؟ (زیادہ سے زیادہ 3 منتخب کریں — ڈیجیٹل/آئی ٹی / انتظام / مالیات / انجینئرنگ / صحت کی دیکھ بھال / تعلیم / دستکاری / مہمان نوازی / سیلز / دیگر)",
  training_participation: "کیا آپ نے گزشتہ 12 مہینوں میں کسی پیشہ ورانہ تربیت میں حصہ لیا؟ (آجر کی طرف سے فنڈڈ / خود فنڈڈ / حکومتی پروگرام / نہیں)",
  emiratization_program: "کیا آپ کسی ایمرٹائزیشن پروگرام میں رجسٹرڈ ہیں؟ (ہاں — NAFIS / ہاں — دیگر / نہیں)",
  labour_market_barriers: "لیبر مارکیٹ میں آپ کو کن اہم رکاوٹوں کا سامنا ہے؟ (جو لاگو ہوں منتخب کریں)",
  platform_work: "کیا آپ ڈیجیٹل پلیٹ فارمز کے ذریعے کام کرتے ہیں (مثلاً Uber، Careem، Talabat)؟ (بنیادی آمدنی / اضافی آمدنی / نہیں)",
  platform_names: "آپ کن پلیٹ فارمز کے ذریعے کام کرتے ہیں؟ (رائیڈ ہیلنگ / فوڈ ڈیلیوری / فری لانس / پیشہ ورانہ خدمات / ای کامرس / دیگر)",
  platform_hours: "آپ ہفتے میں ان پلیٹ فارمز کے ذریعے کام کرنے میں کتنے گھنٹے صرف کرتے ہیں؟",
  online_business: "کیا آپ کے پاس کوئی آن لائن کاروبار ہے؟ (رجسٹرڈ کاروبار / غیر رسمی / نہیں)",
  job_satisfaction: "مجموعی طور پر، آپ اپنی موجودہ نوکری سے کتنے مطمئن ہیں؟ (1–5 درجہ بندی)",
  work_safety: "کیا آپ ایک محفوظ اور صحت مند کام کے ماحول میں کام کرتے ہیں؟ (ہمیشہ / زیادہ تر / کبھی کبھار / شاذ و نادر / کبھی نہیں)",
  workplace_issues: "کیا آپ نے گزشتہ 12 مہینوں میں اپنے کام کی جگہ پر مندرجہ ذیل میں سے کسی کا سامنا کیا؟ (ہراسانی / امتیازی سلوک / اجرت کی چوری / معاہدے کی خلاف ورزی / کوئی نہیں)",
  work_life_balance: "کیا آپ محسوس کرتے ہیں کہ آپ کا کام اور زندگی میں مناسب توازن ہے؟ (ہاں / کسی حد تک / نہیں)",
  question_clarity: "آپ اس انٹرویو کے سوالات کی وضاحت کو کیسے درجہ دیں گے؟ (1–5 درجہ بندی)",
  difficulty_answering: "کیا آپ کو کسی مخصوص سوال کا جواب دینے میں دشواری ہوئی؟ (نہیں / ہاں — براہ کرم بتائیں)",
  ai_preference: "کیا آپ اس طرح کے AI اسسٹنٹ کے ذریعے سروے کروانے کو انسانی انٹرویو لینے والے کے مقابلے میں ترجیح دیتے ہیں؟ (AI کو ترجیح / انسان کو ترجیح / کوئی ترجیح نہیں)",
  data_confidence: "آپ کو کتنا یقین ہے کہ اس سروے میں آپ کا ڈیٹا نجی اور خفیہ رکھا جائے گا؟ (بہت پراعتماد / کسی حد تک / پراعتماد نہیں)",
  survey_comments: "کیا آپ کے پاس اس سروے کو بہتر بنانے کے لیے کوئی تبصرے یا تجاویز ہیں؟ (اختیاری)",
  },
  hi: {
  employment_status: "आपकी वर्तमान रोजगार स्थिति क्या है? (नियोजित / बेरोजगार / श्रम शक्ति से बाहर)",
  education_level: "आपकी पूरी की गई उच्चतम शिक्षा का स्तर क्या है? (कोई औपचारिक शिक्षा नहीं / प्राथमिक / माध्यमिक / डिप्लोमा / स्नातक डिग्री / स्नातकोत्तर डिग्री / पीएचडी या उच्चतर)",
  field_of_study: "आपका मुख्य अध्ययन क्षेत्र क्या था? (जैसे इंजीनियरिंग, व्यवसाय, चिकित्सा, कंप्यूटर विज्ञान, कला)",
  gender: "आपका लिंग क्या है? (पुरुष / महिला / बताना नहीं चाहते)",
  nationality: "आपकी राष्ट्रीयता या मूल देश क्या है? (जैसे भारतीय, पाकिस्तानी, फिलिपिनो, अमीराती, मिस्री, ब्रिटिश, अमेरिकी)",
  marital_status: "आपकी वैवाहिक स्थिति क्या है? (अविवाहित / विवाहित / तलाकशुदा / विधवा/विधुर)",
  emirate: "आप वर्तमान में किस अमीरात में रहते हैं? (अबू धाबी / दुबई / शारजाह / अजमान / उम्म अल क़ुवैन / रास अल खैमाह / फुजैराह)",
  uae_residence_duration: "आप कब से यूएई में रह रहे हैं? (यूएई में जन्मे / 1 वर्ष से कम / 1–4 वर्ष / 5–9 वर्ष / 10–19 वर्ष / 20+ वर्ष)",
  vocational_training: "क्या आपने पिछले 12 महीनों में कोई व्यावसायिक प्रशिक्षण या प्रमाणन पूरा किया है? (हाँ / नहीं)",
  employment_nature: "आपकी रोजगार व्यवस्था का सबसे अच्छा वर्णन क्या है? (वेतनभोगी कर्मचारी / नियोक्ता / स्वरोजगार / पारिवारिक व्यवसाय में सहयोगी कार्यकर्ता)",
  employment_sector: "आपका नियोक्ता किस क्षेत्र से संबंधित है? (सरकारी / निजी / अर्ध-सरकारी / गैर-लाभकारी / एनजीओ)",
  job_title: "आपका पद (जॉब टाइटल) क्या है?",
  job_duties: "कृपया अपनी इस भूमिका में अपने मुख्य कार्यों और कर्तव्यों का संक्षेप में वर्णन करें।",
  industry: "आप किस उद्योग या क्षेत्र में काम करते हैं? (जैसे प्रौद्योगिकी, स्वास्थ्य सेवा, एयरलाइंस, सरकार, वित्त, शिक्षा)",
  actual_hours_worked: "पिछले सप्ताह आपने अपनी मुख्य नौकरी में वास्तव में कितने घंटे काम किया?",
  hours_per_week: "आप आमतौर पर प्रति सप्ताह कितने घंटे काम करते हैं?",
  secondary_job: "क्या आपकी मुख्य नौकरी के अलावा कोई अन्य नौकरी या व्यवसाय है? (हाँ / नहीं)",
  secondary_job_hours: "पिछले सप्ताह आपने अपनी द्वितीयक नौकरी(यों) में कितने घंटे काम किया?",
  underemployment: "क्या आप अपने वर्तमान घंटों से अधिक काम करना चाहते हैं? (हाँ — मुझे अधिक घंटे चाहिए / नहीं / पहले से ही अधिक काम कर रहे हैं)",
  employment_type: "आपकी रोजगार व्यवस्था क्या है? (पूर्णकालिक / अंशकालिक / मौसमी / अस्थायी)",
  contract_type: "क्या आपका रोजगार अनुबंध स्थायी है या अस्थायी? (स्थायी / निश्चित अवधि <1 वर्ष / निश्चित अवधि 1–3 वर्ष / परिवीक्षा अवधि / कोई लिखित अनुबंध नहीं)",
  remote_work: "क्या आप मुख्य रूप से घर से काम करते हैं? (हमेशा / अधिकतर / आंशिक रूप से / कभी नहीं)",
  monthly_wage_range: "आपकी लगभग मासिक वेतन सीमा क्या है? (AED में) (5,000 से कम / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / 50,000 से अधिक / बताना नहीं चाहते)",
  salary_allowances: "क्या आपके वेतन में निम्नलिखित में से कोई भत्ता शामिल है? (आवास / परिवहन / भोजन / शिक्षा / कोई नहीं — जो लागू हों चुनें)",
  bonuses: "क्या आपको पिछले 12 महीनों में कोई बोनस या प्रोत्साहन मिला? (वार्षिक बोनस / प्रदर्शन बोनस / अन्य / नहीं)",
  health_insurance: "क्या आपका नियोक्ता स्वास्थ्य बीमा प्रदान करता है? (पूर्ण कवरेज / आंशिक कवरेज / नहीं / मैं स्वयं भुगतान करता हूँ)",
  pension_scheme: "क्या आप किसी पेंशन फंड या उपदान योजना में नामांकित हैं? (GPSSA / DIFC-ADGM योजना / नियोक्ता की निजी योजना / नहीं / निश्चित नहीं)",
  qualification_match: "क्या आपको लगता है कि आपकी योग्यताएं आपकी वर्तमान नौकरी से मेल खाती हैं? (अधिक योग्य / अच्छी तरह मेल खाता है / कम योग्य)",
  job_search_active: "पिछले चार हफ्तों में, क्या आप सक्रिय रूप से नौकरी की तलाश कर रहे हैं? (हाँ / नहीं)",
  job_search_methods: "आपने रोजगार खोजने के लिए किन तरीकों का उपयोग किया है? (कृपया वर्णन करें)",
  available_for_work: "यदि आज कोई उपयुक्त नौकरी मिले, तो क्या आप अगले दो हफ्तों में काम शुरू करने के लिए उपलब्ध होंगे? (हाँ / नहीं)",
  unemployment_duration: "आप कब से काम की तलाश कर रहे हैं?",
  desired_job_type: "आप किस प्रकार का काम खोज रहे हैं? (मेरे पिछले पेशे जैसा / एक अलग पेशा / यह मेरी पहली नौकरी होगी)",
  ever_worked: "क्या आपने कभी काम किया है? (हाँ — यूएई में / हाँ — यूएई के बाहर / नहीं — कभी काम नहीं किया)",
  last_job_title: "आपकी सबसे हाल की नौकरी का पद क्या था? (यदि कभी काम नहीं किया, तो 'कभी काम नहीं किया' कहें।)",
  last_job_sector: "आपकी पिछली नौकरी किस क्षेत्र में थी? (सरकारी / निजी / अर्ध-सरकारी / गैर-लाभकारी / एनजीओ / स्वरोजगार)",
  reason_left_job: "आपने अपनी पिछली नौकरी क्यों छोड़ी या खोई? (छंटनी हुई / इस्तीफा दिया / व्यवसाय बंद हो गया / अनुबंध समाप्त हुआ / अन्य)",
  highest_previous_salary: "आपकी पिछली नौकरी में सबसे अधिक मासिक वेतन क्या था? (AED में)",
  outside_lf_reason: "आप वर्तमान में काम की तलाश न करने का मुख्य कारण क्या है? (सेवानिवृत्त / छात्र / गृहिणी/गृहस्थ / निराश कार्यकर्ता / बीमारी या विकलांगता / अन्य)",
  main_skills: "आपकी मुख्य कार्य-संबंधी कौशल क्या हैं? (अधिकतम 3 चुनें — डिजिटल/आईटी / प्रबंधन / वित्त / इंजीनियरिंग / स्वास्थ्य सेवा / शिक्षा / व्यापार / आतिथ्य / बिक्री / अन्य)",
  training_participation: "क्या आपने पिछले 12 महीनों में किसी व्यावसायिक प्रशिक्षण में भाग लिया? (नियोक्ता द्वारा वित्त पोषित / स्व-वित्त पोषित / सरकारी कार्यक्रम / नहीं)",
  emiratization_program: "क्या आप किसी एमिराटाइजेशन कार्यक्रम में पंजीकृत हैं? (हाँ — NAFIS / हाँ — अन्य / नहीं)",
  labour_market_barriers: "श्रम बाजार में आपको किन मुख्य बाधाओं का सामना करना पड़ता है? (जो लागू हों चुनें)",
  platform_work: "क्या आप डिजिटल प्लेटफॉर्म के माध्यम से काम करते हैं (जैसे Uber, Careem, Talabat)? (मुख्य आय / अतिरिक्त आय / नहीं)",
  platform_names: "आप किन प्लेटफॉर्म के माध्यम से काम करते हैं? (राइड-हेलिंग / फूड डिलीवरी / फ्रीलांस / पेशेवर सेवाएं / ई-कॉमर्स / अन्य)",
  platform_hours: "आप प्रति सप्ताह इन प्लेटफॉर्म के माध्यम से काम करने में कितने घंटे बिताते हैं?",
  online_business: "क्या आपके पास कोई ऑनलाइन व्यवसाय है? (पंजीकृत व्यवसाय / अनौपचारिक / नहीं)",
  job_satisfaction: "कुल मिलाकर, आप अपनी वर्तमान नौकरी से कितने संतुष्ट हैं? (1–5 रेटिंग)",
  work_safety: "क्या आप एक सुरक्षित और स्वस्थ कार्य वातावरण में काम करते हैं? (हमेशा / अधिकतर / कभी-कभी / शायद ही कभी / कभी नहीं)",
  workplace_issues: "क्या आपने पिछले 12 महीनों में अपने कार्यस्थल पर निम्नलिखित में से किसी का अनुभव किया? (उत्पीड़न / भेदभाव / वेतन की चोरी / अनुबंध का उल्लंघन / कोई नहीं)",
  work_life_balance: "क्या आपको लगता है कि आपके पास कार्य-जीवन का उचित संतुलन है? (हाँ / कुछ हद तक / नहीं)",
  question_clarity: "आप इस साक्षात्कार में प्रश्नों की स्पष्टता को कैसे आंकेंगे? (1–5 रेटिंग)",
  difficulty_answering: "क्या आपको किसी विशिष्ट प्रश्न का उत्तर देने में कठिनाई हुई? (नहीं / हाँ — कृपया बताएं)",
  ai_preference: "क्या आप इस तरह के AI सहायक के माध्यम से सर्वेक्षण कराना पसंद करते हैं? (AI पसंद है / मानव पसंद है / कोई प्राथमिकता नहीं)",
  data_confidence: "आपको कितना विश्वास है कि इस सर्वेक्षण में आपका डेटा निजी रखा जाएगा? (बहुत आश्वस्त / कुछ हद तक / आश्वस्त नहीं)",
  survey_comments: "क्या इस सर्वेक्षण को बेहतर बनाने के लिए आपके पास कोई टिप्पणी है? (वैकल्पिक)",
  },
  tl: {
  employment_status: "Ano ang kasalukuyan mong katayuan sa trabaho? (Nagtatrabaho / Walang trabaho / Wala sa labor force)",
  education_level: "Ano ang pinakamataas na antas ng edukasyon na natapos mo? (Walang pormal na edukasyon / Elementarya / Sekondarya / Diploma / Bachelor's degree / Master's degree / PhD o mas mataas)",
  field_of_study: "Ano ang iyong pangunahing larangan ng pag-aaral? (hal. Engineering, Business, Medisina, Computer Science, Arts)",
  gender: "Ano ang iyong kasarian? (Lalaki / Babae / Mas gustong hindi sabihin)",
  nationality: "Ano ang iyong nasyonalidad o bansang pinagmulan? (hal. Indian, Pakistani, Pilipino, Emirati, Egyptian, British, American)",
  marital_status: "Ano ang iyong katayuan sa kasal? (Walang asawa / May asawa / Diborsyado / Balo)",
  emirate: "Aling Emirate ka kasalukuyang naninirahan? (Abu Dhabi / Dubai / Sharjah / Ajman / Umm Al Quwain / Ras Al Khaimah / Fujairah)",
  uae_residence_duration: "Gaano katagal ka nang naninirahan sa UAE? (Ipinanganak sa UAE / <1 taon / 1–4 taon / 5–9 taon / 10–19 taon / 20+ taon)",
  vocational_training: "Nakumpleto mo ba ang anumang bokasyonal na pagsasanay o sertipikasyon sa nakaraang 12 buwan? (Oo / Hindi)",
  employment_nature: "Ano ang pinakamahusay na naglalarawan sa iyong kaayusan sa trabaho? (Sinasahurang empleyado / Employer / Nagsasariling negosyo / Kasapi sa negosyo ng pamilya)",
  employment_sector: "Anong sektor kabilang ang iyong employer? (Gobyerno / Pribado / Semi-gobyerno / Non-profit / NGO)",
  job_title: "Ano ang iyong job title?",
  job_duties: "Maikling ilarawan ang iyong mga pangunahing gawain at tungkulin sa papel na iyon.",
  industry: "Anong industriya o sektor ka nagtatrabaho? (hal. teknolohiya, healthcare, mga airline, gobyerno, pananalapi, edukasyon)",
  actual_hours_worked: "Ilang oras ka talagang nagtrabaho sa iyong pangunahing trabaho noong nakaraang linggo?",
  hours_per_week: "Ilang oras ka karaniwang nagtatrabaho bawat linggo?",
  secondary_job: "May iba ka bang trabaho o negosyo bukod sa iyong pangunahing trabaho? (Oo / Hindi)",
  secondary_job_hours: "Ilang oras ka nagtrabaho sa iyong pangalawang trabaho noong nakaraang linggo?",
  underemployment: "Gusto mo bang magtrabaho nang mas mahabang oras? (Oo — gusto ko ng mas maraming oras / Hindi / Sobra na ang trabaho ko)",
  employment_type: "Ano ang iyong kaayusan sa trabaho? (Full-time / Part-time / Pana-panahon / Kaswal)",
  contract_type: "Permanente ba o pansamantala ang iyong kontrata? (Permanente / May takdang panahon <1 taon / May takdang panahon 1–3 taon / Panahon ng probasyon / Walang nakasulat na kontrata)",
  remote_work: "Pangunahin ka bang nagtatrabaho mula sa bahay? (Palagi / Kadalasan / Bahagya / Hindi kailanman)",
  monthly_wage_range: "Ano ang tinatayang buwanang saklaw ng suweldo mo? (sa AED) (Mas mababa sa 5,000 / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / Higit sa 50,000 / Mas gustong hindi sabihin)",
  salary_allowances: "Kasama ba sa iyong suweldo ang alinman sa mga sumusunod na allowance? (Pabahay / Transportasyon / Pagkain / Edukasyon / Wala — pumili ng lahat na naaangkop)",
  bonuses: "Nakatanggap ka ba ng anumang bonus o insentibo sa nakaraang 12 buwan? (Taunang bonus / Bonus sa performance / Iba pa / Hindi)",
  health_insurance: "Nagbibigay ba ang iyong employer ng health insurance? (Buong coverage / Bahagyang coverage / Hindi / Ako mismo ang nagbabayad)",
  pension_scheme: "Ikaw ba ay naka-enroll sa isang pension fund o gratuity scheme? (GPSSA / DIFC-ADGM scheme / Pribadong scheme ng employer / Hindi / Hindi sigurado)",
  qualification_match: "Sa tingin mo ba ang iyong mga kwalipikasyon ay tumutugma sa iyong kasalukuyang trabaho? (Overqualified / Tumutugmang mabuti / Underqualified)",
  job_search_active: "Sa nakaraang apat na linggo, aktibo ka bang naghahanap ng trabaho? (Oo / Hindi)",
  job_search_methods: "Anong mga paraan ang ginamit mo sa paghahanap ng trabaho? (mangyaring ilarawan)",
  available_for_work: "Kung may angkop na trabahong ialok ngayon, magagawa mo bang magsimula sa loob ng susunod na dalawang linggo? (Oo / Hindi)",
  unemployment_duration: "Gaano katagal ka nang naghahanap ng trabaho?",
  desired_job_type: "Anong uri ng trabaho ang hinahanap mo? (Kapareho ng aking dating trabaho / Ibang trabaho / Ito ang magiging unang trabaho ko)",
  ever_worked: "Nagtrabaho ka na ba kailanman? (Oo — sa UAE / Oo — sa labas ng UAE / Hindi — hindi pa kailanman)",
  last_job_title: "Ano ang iyong pinakahuling job title? (Kung hindi ka pa kailanman nagtrabaho, sabihin 'hindi pa kailanman nagtrabaho'.)",
  last_job_sector: "Anong sektor ang iyong huling trabaho? (Gobyerno / Pribado / Semi-gobyerno / Non-profit / NGO / Nagsasariling negosyo)",
  reason_left_job: "Bakit mo iniwan o nawala ang iyong huling trabaho? (Na-redundant / Nagbitiw / Nagsara ang negosyo / Natapos ang kontrata / Iba pa)",
  highest_previous_salary: "Ano ang pinakamataas na buwanang suweldo na natanggap mo sa iyong nakaraang trabaho? (sa AED)",
  outside_lf_reason: "Ano ang pangunahing dahilan kung bakit hindi ka naghahanap ng trabaho? (Retirado / Estudyante / Nasa bahay / Nawalan ng pag-asa / Sakit o kapansanan / Iba pa)",
  main_skills: "Ano ang iyong mga pangunahing kasanayan? (pumili ng hanggang 3 — Digital/IT / Pamamahala / Pananalapi / Engineering / Healthcare / Edukasyon / Kalakal / Hospitality / Sales / Iba pa)",
  training_participation: "Nakisali ka ba sa anumang bokasyonal na pagsasanay sa nakaraang 12 buwan? (Pinondohan ng employer / Sariling pondo / Programa ng gobyerno / Hindi)",
  emiratization_program: "Naka-rehistro ka ba sa anumang Emiratization program? (Oo — NAFIS / Oo — ibang programa / Hindi)",
  labour_market_barriers: "Ano ang mga pangunahing hadlang na kinakaharap mo sa labor market? (pumili ng lahat na naaangkop)",
  platform_work: "Nagtatrabaho ka ba sa pamamagitan ng mga digital platform (hal. Uber, Careem, Talabat)? (Pangunahing kita / Karagdagang kita / Hindi)",
  platform_names: "Anong mga platform ang ginagamit mo sa pagtatrabaho? (Ride-hailing / Paghahatid ng pagkain / Freelance / Propesyonal na serbisyo / E-commerce / Iba pa)",
  platform_hours: "Ilang oras kada linggo ang ginugugol mo sa pagtatrabaho sa mga platform na ito?",
  online_business: "May online business ka ba? (Rehistradong negosyo / Impormal / Hindi)",
  job_satisfaction: "Sa kabuuan, gaano ka nasisiyahan sa iyong kasalukuyang trabaho? (Mag-rate 1–5)",
  work_safety: "Nagtatrabaho ka ba sa ligtas at malusog na kapaligiran? (Palagi / Kadalasan / Minsan / Bihira / Hindi kailanman)",
  workplace_issues: "Naranasan mo ba ang alinman sa mga sumusunod sa iyong trabaho sa nakaraang 12 buwan? (Panliligalig / Diskriminasyon / Pagnanakaw sa sahod / Paglabag sa kontrata / Wala)",
  work_life_balance: "Sa tingin mo ba mayroon kang angkop na balanse sa trabaho at buhay? (Oo / Medyo / Hindi)",
  question_clarity: "Paano mo ira-rate ang kalinawan ng mga tanong sa interview na ito? (Mag-rate 1–5)",
  difficulty_answering: "Naranasan mo ba ang kahirapan sa pagsagot ng anumang tanong? (Hindi / Oo — mangyaring tukuyin)",
  ai_preference: "Mas gusto mo bang gawin ang survey sa pamamagitan ng isang AI assistant? (Mas gusto ang AI / Mas gusto ang tao / Walang preference)",
  data_confidence: "Gaano ka kumpiyansa na ang iyong data ay itatago nang pribado sa survey na ito? (Lubos na kumpiyansa / Medyo kumpiyansa / Hindi kumpiyansa)",
  survey_comments: "Mayroon ka bang mga komento upang mapabuti ang survey na ito? (opsyonal)",
  },
};

// Field groupings, matching _get_field_order()'s real section structure and
// gating conditions exactly. "gate" is shown verbatim, not paraphrased.
const SECTIONS = [
  {
    id: "base", title: "Core (asked first, all paths)", code: "—",
    fields: [
      { key: "employment_status" },
      { key: "education_level" },
      { key: "field_of_study", gate: "only if education_level is bachelor / master / phd" },
    ],
  },
  {
    id: "demographics", title: "Demographics", code: "Section B", paths: "all paths",
    fields: [
      { key: "gender" }, { key: "nationality" }, { key: "marital_status" },
      { key: "emirate" }, { key: "uae_residence_duration" }, { key: "vocational_training" },
    ],
  },
  {
    id: "employment-details", title: "Employment Details", code: "Section C", paths: "Employed path only",
    fields: [
      { key: "employment_nature" }, { key: "employment_sector" }, { key: "job_title" },
      { key: "job_duties" }, { key: "industry" },
    ],
  },
  {
    id: "hours-conditions", title: "Hours & Conditions", code: "Section D", paths: "Employed path only",
    fields: [
      { key: "actual_hours_worked" }, { key: "hours_per_week" }, { key: "secondary_job" },
      { key: "secondary_job_hours", gate: "only if secondary_job = yes" },
      { key: "underemployment" }, { key: "employment_type" },
      { key: "contract_type", gate: "only if employment_nature = paid_employee" },
      { key: "remote_work" },
    ],
  },
  {
    id: "wages-benefits", title: "Wages & Benefits", code: "Section E", paths: "Employed path only",
    fields: [
      { key: "monthly_wage_range", gate: "only if employment_nature = paid_employee" },
      { key: "salary_allowances" }, { key: "bonuses" }, { key: "health_insurance" },
      { key: "pension_scheme" }, { key: "qualification_match" },
    ],
  },
  {
    id: "job-search", title: "Unemployment & Job Search", code: "Section F", paths: "Unemployed path only",
    fields: [
      { key: "job_search_active" },
      { key: "job_search_methods", gate: "only if job_search_active = yes" },
      { key: "available_for_work" }, { key: "unemployment_duration" },
      { key: "desired_job_type" }, { key: "ever_worked" },
    ],
  },
  {
    id: "outside-lf", title: "Outside Labour Force", code: "Section F", paths: "Not-in-labour-force path only",
    fields: [ { key: "outside_lf_reason" }, { key: "ever_worked" } ],
  },
  {
    id: "previous-employment", title: "Previous Employment", code: "Section G",
    paths: "Unemployed / Not-in-labour-force paths, only if ever_worked ≠ never_worked",
    fields: [
      { key: "last_job_title" }, { key: "last_job_sector" },
      { key: "reason_left_job", gate: "Unemployed path only" },
      { key: "highest_previous_salary" },
    ],
  },
  {
    id: "skills-training", title: "Skills & Training", code: "Section H", paths: "all paths",
    fields: [
      { key: "main_skills" }, { key: "training_participation" },
      { key: "emiratization_program", gate: "only if nationality = UAE national" },
      { key: "labour_market_barriers" },
    ],
  },
  {
    id: "digital-work", title: "Digital Work", code: "Section I", paths: "all paths",
    fields: [
      { key: "platform_work" },
      { key: "platform_names", gate: "only if platform_work = yes (primary or supplementary)" },
      { key: "platform_hours", gate: "only if platform_work = yes (primary or supplementary)" },
      { key: "online_business" },
    ],
  },
  {
    id: "quality-of-work", title: "Quality of Work", code: "Section J", paths: "Employed path only",
    fields: [
      { key: "job_satisfaction" }, { key: "work_safety" },
      { key: "workplace_issues" }, { key: "work_life_balance" },
    ],
  },
  {
    id: "feedback", title: "Feedback", code: "Section K", paths: "all paths",
    fields: [
      { key: "question_clarity" }, { key: "difficulty_answering" },
      { key: "ai_preference" }, { key: "data_confidence" },
      { key: "survey_comments", gate: "optional" },
    ],
  },
];

const TOTAL_FIELDS = SECTIONS.reduce((n, s) => n + s.fields.length, 0);

const LANG_PILLS = [
  { code: "en", label: "EN" },
  { code: "ar", label: "AR" },
  { code: "ur", label: "UR" },
  { code: "hi", label: "HI" },
  { code: "tl", label: "TL" },
];
const RTL_LANGS = new Set(["ar", "ur"]);

export default function QuestionnairePage() {
  const [openSection, setOpenSection] = useState(SECTIONS.map(s => s.id));
  const [qLang, setQLang] = useState("en");
  const dir = RTL_LANGS.has(qLang) ? "rtl" : "ltr";

  function toggle(id) {
    setOpenSection(o => o.includes(id) ? o.filter(x => x !== id) : [...o, id]);
  }

  return (
    <>
      <Head>
        <title>Full Questionnaire — LFS AI Platform</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div style={{ fontFamily: SANS, background: C.bg, minHeight: "100vh", color: C.text }}>
        <header style={{ background: C.white, borderBottom: `1px solid ${C.border}`, position: "sticky", top: 0, zIndex: 10 }}>
          <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 24px", height: 56, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ background: C.green, fontFamily: MONO, width: 32, height: 32, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 11, fontWeight: 700 }}>
                LFS
              </div>
              <span style={{ fontFamily: MONO, color: C.text, fontSize: 13, fontWeight: 600 }}>Full Questionnaire Reference</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 11, color: C.muted, fontFamily: MONO }}>LANG:</span>
                {LANG_PILLS.map(lp => (
                  <button key={lp.code} onClick={() => setQLang(lp.code)}
                    style={{
                      fontFamily: MONO, fontSize: 10, fontWeight: 700, padding: "3px 10px", borderRadius: 99,
                      border: `1px solid ${qLang === lp.code ? C.green : C.border}`,
                      background: qLang === lp.code ? C.green : "transparent",
                      color: qLang === lp.code ? "#fff" : C.muted,
                      cursor: "pointer",
                    }}>
                    {lp.label}
                  </button>
                ))}
              </div>
              <Link href="/" style={{ fontSize: 12, color: C.green, textDecoration: "underline" }}>← Back to home</Link>
            </div>
          </div>
        </header>

        <div style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px 64px" }}>

          <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginBottom: 28 }}>
            <p style={{ fontSize: 13, color: C.text, lineHeight: 1.6, marginBottom: 8 }}>
              <strong>{TOTAL_FIELDS} real fields</strong>, sourced directly from the live backend
              (<code style={{ fontFamily: MONO, fontSize: 12, background: C.bg, padding: "1px 5px", borderRadius: 4 }}>
                conversation_manager.py
              </code>'s question text and skip-logic — not a reconstruction). No single respondent
              answers all {TOTAL_FIELDS}: the actual path taken depends on employment status and a
              handful of conditional gates, shown inline below.
            </p>
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6 }}>
              <strong>Employed path:</strong> Core + Demographics + Employment Details + Hours &
              Conditions + Wages & Benefits + Skills & Training + Digital Work + Quality of Work + Feedback.{" "}
              <strong>Unemployed path:</strong> Core + Demographics + Unemployment & Job Search +
              (Previous Employment, if ever worked) + Skills & Training + Digital Work + Feedback.{" "}
              <strong>Not-in-labour-force path:</strong> Core + Demographics + Outside Labour Force +
              (Previous Employment, if ever worked) + Skills & Training + Digital Work + Feedback.
            </p>
            <p style={{ fontSize: 11, color: C.amber, background: C.amberBg, border: "1px solid #fde68a", borderRadius: 8, padding: "8px 12px", marginTop: 12 }}>
              Question wording below is the exact English text used by the system's fast-mode
              (no-LLM) path. As of 2026-08-29, all 5 languages (English, Arabic, Urdu, Hindi,
              Tagalog) have their own translated wording for the full conversation in fast mode —
              greeting, follow-up questions, clarification prompts, the final answer-review
              summary, and the closing message. Arabic and English acknowledgments are
              field-specific; Urdu/Hindi/Tagalog acknowledgments use a shorter generic phrasing
              instead (by design, not a bug — see the code comments in conversation_manager.py).
              The Urdu/Hindi/Tagalog translations are machine-drafted and have not yet been
              reviewed by a native speaker — treat them as a first draft pending review, not
              verified final wording. The full LLM-based production path handles all five
              languages independently of this fast-mode wording.
            </p>
          </div>

          {SECTIONS.map(sec => (
            <div key={sec.id} style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 12, marginBottom: 14, overflow: "hidden" }}>
              <button onClick={() => toggle(sec.id)} style={{
                width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "14px 18px", background: "none", border: "none", cursor: "pointer", textAlign: "left",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: C.green, background: C.greenLt, border: `1px solid ${C.green}30`, borderRadius: 6, padding: "2px 8px" }}>
                    {sec.code}
                  </span>
                  <h2 style={{ fontSize: 14, fontWeight: 700, color: C.text, margin: 0 }}>{sec.title}</h2>
                  {sec.paths && <span style={{ fontSize: 11, color: C.faint }}>· {sec.paths}</span>}
                </div>
                <span style={{ color: C.faint, fontSize: 12 }}>{openSection.includes(sec.id) ? "▾" : "▸"} {sec.fields.length}</span>
              </button>

              {openSection.includes(sec.id) && (
                <div style={{ borderTop: `1px solid ${C.border}` }}>
                  {sec.fields.map((f, i) => (
                    <div key={f.key} style={{ padding: "12px 18px", borderTop: i > 0 ? `1px solid ${C.border}` : "none" }}>
                      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
                        <code style={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: C.greenDk }}>{f.key}</code>
                        {f.gate && (
                          <span style={{ fontSize: 10, color: C.amber, background: C.amberBg, border: "1px solid #fde68a", borderRadius: 5, padding: "1px 6px" }}>
                            {f.gate}
                          </span>
                        )}
                      </div>
                      <p dir={dir} style={{ fontSize: 13, color: C.text, lineHeight: 1.5, margin: 0, textAlign: dir === "rtl" ? "right" : "left" }}>{Q[qLang][f.key]}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}

          <p style={{ fontSize: 11, color: C.faint, textAlign: "center", marginTop: 24 }}>
            Nationality classification separately maps free text to UN M49 country codes
            (backend/agents/nationality_classifier.py, verified real). Occupation, industry, and
            education-field free-text answers (job_title, industry, field_of_study) are separately
            classified to ISCO-08 / ISIC Rev.4 / ISCED-F — see the report page after completing a session.
          </p>
        </div>
      </div>
    </>
  );
}
