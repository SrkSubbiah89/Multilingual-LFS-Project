import evidence from "../data/parent-document-comparison-2026-10-06.json";
import historical from "../data/historical-intended-e5large-reference.json";

const T = {
  en: {
    title: "Active local retrieval — Parent-document RAG",
    subtitle: "Frozen comparison on reused WISCO v3 cases, with the same catalogue and E5-small encoder in both methods.",
    heldout: "Historical held-out split (reused)", validation: "Validation split (reused)",
    dense: "Dense flat retrieval", parent: "Parent-document RAG",
    method: "Method", correct: "Exact matches / cases", accuracy: "Exact-code accuracy", ci: "95% Wilson interval",
    perLanguage: "Exact-code results by input language", language: "Language",
    languages: { ar: "Arabic", en: "English", hi: "Hindi", tl: "Tagalog", ur: "Urdu" },
    encoder: "Encoder", catalogue: "Catalogue", reranking: "LLM reranking", off: "Disabled",
    limitation: "These benchmark splits were used historically, and catalogue enrichment was informed by held-out errors. This is not an untouched test or Labour Force Survey field validation.",
    intervalNote: "Intervals describe case-level results. Multilingual titles share source groups and are not independent respondents.",
    comparisonNote: "Compare the two methods within this card. The historical experiments below are separate references, including the intended E5-large profile with unresolved encoder provenance. This result does not exceed the historical 40.95%.",
    provenance: "Sources and frozen configuration", source: "Source report", parameters: "Frozen parameters",
    active: "Active local method", baseline: "Same-encoder comparison baseline",
    localSetup: "Local setup verified on 6 October 2026. These percentages describe benchmark results, not the confidence of an individual survey classification.",
    historicalTitle: "Historical offline reference — intended E5-large profile",
    historicalDate: "Run date", intendedEncoder: "Intended encoder", executedEncoder: "Executed encoder", unresolved: "Unresolved",
    historicalWarning: "The recorded encoder conflicts with the intended E5-large profile. The provenance correction leaves the executed encoder unknown; 40.95% remains a historical offline result, not verified E5-large or live survey accuracy.",
    historicalScope: "This historical run uses the same 18,747 WISCO v3 cases. Its 40.95% exceeds the current 38.83% reference result, but it cannot establish a comparison between verified encoders. These reused cases do not validate Labour Force Survey field accuracy.",
    correction: "Encoder provenance correction",
  },
  ar: {
    title: "الاسترجاع المحلي النشط — RAG بالمستندات الأصلية",
    subtitle: "مقارنة بإعدادات ثابتة على حالات WISCO v3 المُعاد استخدامها، بالكتالوج نفسه ومُرمّز E5-small نفسه للطريقتين.",
    heldout: "مجموعة الاختبار المحجوزة تاريخيًا (مُعاد استخدامها)", validation: "مجموعة التحقق (مُعاد استخدامها)",
    dense: "الاسترجاع المسطح الكثيف", parent: "RAG بالمستندات الأصلية",
    method: "الطريقة", correct: "التطابقات الكاملة / الحالات", accuracy: "دقة الرمز الكامل", ci: "فاصل ويلسون 95%",
    perLanguage: "نتائج الرمز الكامل حسب لغة الإدخال", language: "اللغة",
    languages: { ar: "العربية", en: "الإنجليزية", hi: "الهندية", tl: "التاغالوغية", ur: "الأردية" },
    encoder: "المُرمّز", catalogue: "الكتالوج", reranking: "إعادة الترتيب بنموذج لغوي", off: "مُعطّلة",
    limitation: "استُخدمت مجموعات هذا المعيار سابقًا، واستند إثراء الكتالوج إلى أخطاء الاختبار المحجوز. هذا ليس اختبارًا لم يُستخدم من قبل أو تحققًا ميدانيًا لمسح القوى العاملة.",
    intervalNote: "تصف الفواصل النتائج على مستوى الحالات. تشترك العناوين متعددة اللغات في مجموعات المصدر ولا تمثل مستجيبين مستقلين.",
    comparisonNote: "قارن الطريقتين داخل هذه البطاقة. التجارب التاريخية أدناه مراجع منفصلة، بما فيها ملف E5-large المقصود ذو مصدر هوية مُرمّز غير محسوم. لا تتجاوز هذه النتيجة نسبة 40.95% التاريخية.",
    provenance: "المصادر والإعدادات الثابتة", source: "تقرير المصدر", parameters: "المعاملات الثابتة",
    active: "الطريقة المحلية النشطة", baseline: "خط المقارنة بالمُرمّز نفسه",
    localSetup: "تم التحقق من الإعداد المحلي في 6 أكتوبر 2026. تصف هذه النسب نتائج المعيار، وليست درجة الثقة في تصنيف استبيان فردي.",
    historicalTitle: "مرجع تاريخي غير مستخدم في التشغيل الحالي — ملف E5-large المقصود",
    historicalDate: "تاريخ التشغيل", intendedEncoder: "المُرمّز المقصود", executedEncoder: "المُرمّز المنفّذ", unresolved: "غير محسوم",
    historicalWarning: "يتعارض المُرمّز المسجّل مع ملف E5-large المقصود. يترك تصحيح المصدر هوية المُرمّز المنفّذ غير معروفة؛ تبقى 40.95% نتيجة تاريخية غير مستخدمة في التشغيل الحالي، وليست دقة E5-large متحققًا منه أو دقة الاستبيان الجاري.",
    historicalScope: "يستخدم هذا التشغيل التاريخي حالات WISCO v3 نفسها وعددها 18,747. تتجاوز نتيجته 40.95% النتيجة المرجعية الحالية 38.83%، لكنه لا يثبت مقارنة بين مُرمّزات متحقق منها. هذه الحالات المُعاد استخدامها لا تتحقق من الدقة الميدانية لمسح القوى العاملة.",
    correction: "تصحيح مصدر هوية المُرمّز",
  },
  ur: {
    title: "فعال مقامی بازیافت — اصل دستاویز پر مبنی RAG",
    subtitle: "دوبارہ استعمال ہونے والے WISCO v3 کیسز پر مقررہ ترتیب کا موازنہ، دونوں طریقوں میں ایک ہی کیٹلاگ اور E5-small انکوڈر کے ساتھ۔",
    heldout: "تاریخی مخصوص ٹیسٹ اسپلٹ (دوبارہ استعمال شدہ)", validation: "توثیقی اسپلٹ (دوبارہ استعمال شدہ)",
    dense: "ڈینس فلیٹ بازیافت", parent: "اصل دستاویز پر مبنی RAG",
    method: "طریقہ", correct: "درست مماثلتیں / کیسز", accuracy: "مکمل کوڈ کی درستگی", ci: "95% ولسن وقفہ",
    perLanguage: "ان پٹ زبان کے لحاظ سے مکمل کوڈ کے نتائج", language: "زبان",
    languages: { ar: "عربی", en: "انگریزی", hi: "ہندی", tl: "ٹیگالوگ", ur: "اردو" },
    encoder: "انکوڈر", catalogue: "کیٹلاگ", reranking: "LLM دوبارہ ترتیب", off: "غیر فعال",
    limitation: "یہ بینچ مارک اسپلٹس پہلے استعمال ہو چکے ہیں، اور کیٹلاگ میں بہتری تاریخی ٹیسٹ کی غلطیوں سے متاثر تھی۔ یہ پہلے کبھی نہ دیکھا گیا ٹیسٹ یا لیبر فورس سروے کی فیلڈ توثیق نہیں ہے۔",
    intervalNote: "وقفے کیس کی سطح کے نتائج کو بیان کرتے ہیں۔ کثیر لسانی عنوانات مشترک ماخذ گروہوں سے تعلق رکھتے ہیں اور آزاد جواب دہندگان نہیں ہیں۔",
    comparisonNote: "اس کارڈ میں موجود دونوں طریقوں کا موازنہ کریں۔ نیچے تاریخی تجربات الگ حوالے ہیں، جن میں مطلوبہ E5-large پروفائل بھی ہے جس کے انکوڈر کا ماخذ غیر طے شدہ ہے۔ یہ نتیجہ تاریخی 40.95% سے زیادہ نہیں ہے۔",
    provenance: "ماخذ اور مقررہ ترتیب", source: "ماخذ رپورٹ", parameters: "مقررہ پیرامیٹرز",
    active: "فعال مقامی طریقہ", baseline: "ایک ہی انکوڈر کا تقابلی بنیادی طریقہ",
    localSetup: "مقامی ترتیب کی 6 اکتوبر 2026 کو تصدیق ہوئی۔ یہ فیصد بینچ مارک کے نتائج ہیں، کسی انفرادی سروے کی درجہ بندی کا اعتماد نہیں۔",
    historicalTitle: "تاریخی آف لائن حوالہ — مطلوبہ E5-large پروفائل",
    historicalDate: "رن کی تاریخ", intendedEncoder: "مطلوبہ انکوڈر", executedEncoder: "استعمال شدہ انکوڈر", unresolved: "غیر طے شدہ",
    historicalWarning: "ریکارڈ شدہ انکوڈر مطلوبہ E5-large پروفائل سے مختلف ہے۔ ماخذ کی تصحیح میں اصل استعمال شدہ انکوڈر نامعلوم ہے؛ 40.95% ایک تاریخی آف لائن نتیجہ ہے، تصدیق شدہ E5-large یا موجودہ سروے کی درستگی نہیں۔",
    historicalScope: "اس تاریخی رن میں وہی 18,747 WISCO v3 کیسز ہیں۔ اس کا 40.95% موجودہ 38.83% حوالہ نتیجے سے زیادہ ہے، لیکن یہ تصدیق شدہ انکوڈرز کا موازنہ ثابت نہیں کرتا۔ دوبارہ استعمال شدہ کیسز لیبر فورس سروے کی فیلڈ درستگی کی توثیق نہیں کرتے۔",
    correction: "انکوڈر کے ماخذ کی تصحیح",
  },
  hi: {
    title: "सक्रिय स्थानीय पुनर्प्राप्ति — मूल दस्तावेज़ आधारित RAG",
    subtitle: "दोबारा इस्तेमाल किए गए WISCO v3 मामलों पर निश्चित विन्यास की तुलना, दोनों विधियों में समान कैटलॉग और E5-small एन्कोडर के साथ।",
    heldout: "ऐतिहासिक आरक्षित परीक्षण विभाजन (पुनः उपयोग)", validation: "सत्यापन विभाजन (पुनः उपयोग)",
    dense: "डेंस फ्लैट पुनर्प्राप्ति", parent: "मूल दस्तावेज़ आधारित RAG",
    method: "विधि", correct: "सटीक मिलान / मामले", accuracy: "सटीक कोड की शुद्धता", ci: "95% विल्सन अंतराल",
    perLanguage: "इनपुट भाषा के अनुसार सटीक कोड परिणाम", language: "भाषा",
    languages: { ar: "अरबी", en: "अंग्रेज़ी", hi: "हिंदी", tl: "टैगालोग", ur: "उर्दू" },
    encoder: "एन्कोडर", catalogue: "कैटलॉग", reranking: "LLM पुनः रैंकिंग", off: "अक्षम",
    limitation: "ये बेंचमार्क विभाजन पहले इस्तेमाल किए गए हैं, और कैटलॉग का विस्तार पुराने आरक्षित परीक्षण की त्रुटियों से प्रभावित था। यह अछूता परीक्षण या श्रम बल सर्वेक्षण का क्षेत्र सत्यापन नहीं है।",
    intervalNote: "अंतराल मामले के स्तर के परिणाम बताते हैं। बहुभाषी शीर्षक समान स्रोत समूह साझा करते हैं और स्वतंत्र उत्तरदाता नहीं हैं।",
    comparisonNote: "इस कार्ड की दोनों विधियों की तुलना करें। नीचे ऐतिहासिक प्रयोग अलग संदर्भ हैं, जिनमें अनिश्चित एन्कोडर स्रोत वाली अभिप्रेत E5-large प्रोफ़ाइल भी शामिल है। यह परिणाम ऐतिहासिक 40.95% से अधिक नहीं है।",
    provenance: "स्रोत और निश्चित विन्यास", source: "स्रोत रिपोर्ट", parameters: "निश्चित पैरामीटर",
    active: "सक्रिय स्थानीय विधि", baseline: "समान एन्कोडर की तुलना आधार विधि",
    localSetup: "स्थानीय विन्यास का सत्यापन 6 अक्टूबर 2026 को हुआ। ये प्रतिशत बेंचमार्क परिणाम हैं, किसी व्यक्तिगत सर्वेक्षण वर्गीकरण का विश्वास स्तर नहीं।",
    historicalTitle: "ऐतिहासिक ऑफलाइन संदर्भ — अभिप्रेत E5-large प्रोफ़ाइल",
    historicalDate: "रन की तारीख", intendedEncoder: "अभिप्रेत एन्कोडर", executedEncoder: "प्रयुक्त एन्कोडर", unresolved: "अनिश्चित",
    historicalWarning: "दर्ज एन्कोडर अभिप्रेत E5-large प्रोफ़ाइल से मेल नहीं खाता। स्रोत सुधार में वास्तविक प्रयुक्त एन्कोडर अज्ञात है; 40.95% एक ऐतिहासिक ऑफलाइन परिणाम है, सत्यापित E5-large या वर्तमान सर्वेक्षण की शुद्धता नहीं।",
    historicalScope: "इस ऐतिहासिक रन में वही 18,747 WISCO v3 मामले हैं। इसका 40.95% वर्तमान 38.83% संदर्भ परिणाम से अधिक है, लेकिन यह सत्यापित एन्कोडरों की तुलना स्थापित नहीं करता। दोबारा इस्तेमाल किए गए मामले श्रम बल सर्वेक्षण की क्षेत्रीय शुद्धता का सत्यापन नहीं करते।",
    correction: "एन्कोडर स्रोत सुधार",
  },
  tl: {
    title: "Aktibong lokal na retrieval — Parent-document RAG",
    subtitle: "Paghahambing gamit ang nakapirming konpigurasyon sa muling ginamit na WISCO v3 cases, na may parehong catalogue at E5-small encoder para sa dalawang paraan.",
    heldout: "Makasaysayang held-out split (muling ginamit)", validation: "Validation split (muling ginamit)",
    dense: "Dense flat retrieval", parent: "Parent-document RAG",
    method: "Paraan", correct: "Eksaktong tugma / kaso", accuracy: "Katumpakan ng eksaktong code", ci: "95% Wilson interval",
    perLanguage: "Eksaktong-code na resulta ayon sa wika ng input", language: "Wika",
    languages: { ar: "Arabic", en: "English", hi: "Hindi", tl: "Tagalog", ur: "Urdu" },
    encoder: "Encoder", catalogue: "Catalogue", reranking: "LLM reranking", off: "Naka-disable",
    limitation: "Nagamit na dati ang mga benchmark split na ito, at naimpluwensiyahan ng mga dating held-out error ang pagpapalawak ng catalogue. Hindi ito hindi pa nagamit na pagsusulit o field validation ng Labour Force Survey.",
    intervalNote: "Inilalarawan ng mga interval ang resulta sa antas ng kaso. May magkakaparehong source group ang mga multilingguwal na pamagat at hindi sila mga independiyenteng respondent.",
    comparisonNote: "Ihambing ang dalawang paraan sa card na ito. Hiwalay na reference ang mga makasaysayang eksperimento sa ibaba, kasama ang nilalayong E5-large profile na hindi pa matiyak ang encoder provenance. Hindi lumampas ang resultang ito sa makasaysayang 40.95%.",
    provenance: "Mga source at nakapirming konpigurasyon", source: "Source report", parameters: "Nakapirming parameter",
    active: "Aktibong lokal na paraan", baseline: "Baseline ng paghahambing gamit ang parehong encoder",
    localSetup: "Na-verify ang lokal na setup noong 6 Oktubre 2026. Inilalarawan ng mga porsiyento ang benchmark results, hindi ang confidence ng isang survey classification.",
    historicalTitle: "Makasaysayang offline reference — nilalayong E5-large profile",
    historicalDate: "Petsa ng run", intendedEncoder: "Nilalayong encoder", executedEncoder: "Aktuwal na encoder", unresolved: "Hindi pa matiyak",
    historicalWarning: "Magkaiba ang naitalang encoder at ang nilalayong E5-large profile. Hindi matiyak sa provenance correction ang aktuwal na encoder; nananatiling makasaysayang offline result ang 40.95%, hindi na-verify na E5-large o live survey accuracy.",
    historicalScope: "Ginamit ng makasaysayang run ang parehong 18,747 WISCO v3 cases. Mas mataas ang 40.95% nito kaysa sa kasalukuyang 38.83% reference result, ngunit hindi nito itinatag ang paghahambing ng mga na-verify na encoder. Hindi field validation ng Labour Force Survey ang muling ginamit na mga kasong ito.",
    correction: "Pagwawasto sa provenance ng encoder",
  },
};

const METHODS = ["parent_document_rag", "dense_flat"];
const percent = value => `${(value * 100).toFixed(2)}%`;
const count = value => value.toLocaleString();
const matches = metric => `${count(metric.top1_correct)}/${count(metric.n)}`;

export default function ParentDocumentEvidence({ language = "en" }) {
  const t = T[language === "ar-gulf" ? "ar" : language] || T.en;
  const methodName = method => method === "dense_flat" ? t.dense : t.parent;
  const heldout = evidence.comparisons.find(comparison => comparison.split === "heldout");
  return (
    <>
    <section aria-label={t.title} className="bg-white border border-emerald-200 rounded-xl overflow-hidden shadow-sm">
      <div className="px-4 py-2.5 border-b border-emerald-100 bg-emerald-50">
        <h3 className="text-xs font-semibold text-emerald-800 uppercase tracking-widest">{t.title}</h3>
      </div>
      <div className="px-4 py-4 space-y-5">
        <p className="text-xs text-gray-600">{t.subtitle}</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {METHODS.map(method => <div key={method} data-evidence-role={method === "parent_document_rag" ? "active" : "baseline"} className={method === "parent_document_rag" ? "rounded-lg border border-emerald-300 bg-emerald-50 p-3" : "rounded-lg border border-gray-200 bg-gray-50 p-3"}>
            <p className="text-[11px] font-semibold text-gray-700">{method === "parent_document_rag" ? t.active : t.baseline}</p>
            <p className="text-xs text-gray-600 mt-1">{methodName(method)}</p>
            <p className="text-2xl font-semibold tabular-nums text-gray-900 mt-1">{percent(heldout.methods[method].top1_accuracy)}</p>
            <p className="text-[11px] text-gray-600">{matches(heldout.methods[method])} · {t.heldout}</p>
          </div>)}
        </div>
        <p className="text-[11px] text-gray-600">{t.localSetup}</p>
        <dl className="flex flex-wrap gap-x-5 gap-y-2 text-[11px] text-gray-600">
          <div><dt className="inline font-semibold">{t.encoder}: </dt><dd className="inline font-mono break-all">{evidence.embedding_model}</dd></div>
          <div><dt className="inline font-semibold">{t.catalogue}: </dt><dd className="inline font-mono break-all">{evidence.catalogue_profile}</dd></div>
          <div><dt className="inline font-semibold">{t.reranking}: </dt><dd className="inline">{t.off}</dd></div>
        </dl>
        <p className="text-xs text-amber-900 bg-amber-50 border border-amber-200 rounded-lg p-3">{t.limitation}</p>
        {evidence.comparisons.map(comparison => (
          <div key={comparison.split} className="space-y-3">
            <h4 className="text-xs font-semibold text-gray-800">{t[comparison.split]} · n={count(comparison.n)}</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead><tr className="border-b border-gray-200">
                  <th scope="col" className="text-start py-2 pe-3 text-gray-500">{t.method}</th>
                  <th scope="col" className="text-center py-2 px-2 text-gray-500">{t.correct}</th>
                  <th scope="col" className="text-center py-2 px-2 text-gray-500">{t.accuracy}</th>
                  <th scope="col" className="text-center py-2 ps-2 text-gray-500">{t.ci}</th>
                </tr></thead>
                <tbody>{METHODS.map(method => {
                  const metric = comparison.methods[method];
                  return <tr key={method} className="border-b border-gray-100">
                    <th scope="row" className="text-start py-2 pe-3 font-medium text-gray-800">{methodName(method)}<span className="block text-[10px] text-gray-500">{method === "parent_document_rag" ? t.active : t.baseline}</span></th>
                    <td className="text-center py-2 px-2 font-mono tabular-nums">{matches(metric)}</td>
                    <td className="text-center py-2 px-2 font-mono tabular-nums">{percent(metric.top1_accuracy)}</td>
                    <td className="text-center py-2 ps-2 font-mono tabular-nums">{`[${metric.top1_wilson95_case_level.map(percent).join(", ")}]`}</td>
                  </tr>;
                })}</tbody>
              </table>
            </div>
            <details className="text-xs">
              <summary className="cursor-pointer font-medium text-gray-700">{t.perLanguage}</summary>
              <div className="overflow-x-auto mt-2">
                <table className="w-full text-xs border-collapse">
                  <thead><tr className="border-b border-gray-200">
                    <th scope="col" className="text-start py-2 pe-3 text-gray-500">{t.language}</th>
                    {METHODS.map(method => <th scope="col" key={method} className="text-center py-2 px-2 text-gray-500">{methodName(method)}</th>)}
                  </tr></thead>
                  <tbody>{Object.entries(comparison.per_language).map(([languageCode, metrics]) => (
                    <tr key={languageCode} className="border-b border-gray-100">
                      <th scope="row" className="text-start py-2 pe-3 font-medium">{t.languages[languageCode]}</th>
                      {METHODS.map(method => <td key={method} className="text-center py-2 px-2 font-mono tabular-nums">{`${matches(metrics[method])} · ${percent(metrics[method].top1_accuracy)}`}</td>)}
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </details>
          </div>
        ))}
        <p className="text-[11px] text-gray-600">{t.intervalNote}</p>
        <p className="text-[11px] text-gray-600">{t.comparisonNote}</p>
        <details className="text-[10px] text-gray-500">
          <summary className="cursor-pointer">{t.provenance}</summary>
          <div className="mt-2 space-y-2 break-all">
            <p>{t.parameters}: <code>{`child_weight=${evidence.selected.child_weight}; aggregation=${evidence.selected.aggregation}`}</code></p>
            <p>{t.catalogue} SHA-256: <code>{evidence.source_catalogue_sha256}</code></p>
            {evidence.comparisons.map(comparison => <p key={comparison.split}>{t.source} ({t[comparison.split]}): <code>{comparison.source_report}</code> · SHA-256: <code>{comparison.source_report_sha256}</code></p>)}
          </div>
        </details>
      </div>
    </section>
    <section aria-label={t.historicalTitle} className="bg-white border border-amber-200 rounded-xl overflow-hidden shadow-sm">
      <div className="px-4 py-2.5 border-b border-amber-100 bg-amber-50">
        <h3 className="text-xs font-semibold text-amber-900 uppercase tracking-widest">{t.historicalTitle}</h3>
      </div>
      <div className="px-4 py-4 space-y-3">
        <p className="text-2xl font-semibold tabular-nums text-gray-900">{percent(historical.metrics.top1_accuracy)}</p>
        <p className="text-xs text-gray-600">{t.correct}: {matches(historical.metrics)} · {t.heldout}</p>
        <dl className="flex flex-wrap gap-x-5 gap-y-2 text-[11px] text-gray-600">
          <div><dt className="inline font-semibold">{t.historicalDate}: </dt><dd className="inline">{historical.run_date}</dd></div>
          <div><dt className="inline font-semibold">{t.intendedEncoder}: </dt><dd className="inline font-mono break-all">{historical.intended_embedding_model}</dd></div>
          <div><dt className="inline font-semibold">{t.executedEncoder}: </dt><dd className="inline font-semibold text-amber-900">{t.unresolved}</dd></div>
          <div><dt className="inline font-semibold">{t.reranking}: </dt><dd className="inline">{t.off}</dd></div>
        </dl>
        <p className="text-xs text-amber-900 bg-amber-50 border border-amber-200 rounded-lg p-3">{t.historicalWarning}</p>
        <p className="text-[11px] text-gray-600">{t.historicalScope}</p>
        <details className="text-xs">
          <summary className="cursor-pointer font-medium text-gray-700">{t.perLanguage}</summary>
          <div className="overflow-x-auto mt-2">
            <table className="w-full text-xs border-collapse">
              <thead><tr className="border-b border-gray-200"><th scope="col" className="text-start py-2 pe-3 text-gray-500">{t.language}</th><th scope="col" className="text-center py-2 px-2 text-gray-500">{t.correct}</th><th scope="col" className="text-center py-2 ps-2 text-gray-500">{t.accuracy}</th></tr></thead>
              <tbody>{Object.entries(historical.per_language).map(([languageCode, metric]) => <tr key={languageCode} className="border-b border-gray-100">
                <th scope="row" className="text-start py-2 pe-3 font-medium">{t.languages[languageCode]}</th>
                <td className="text-center py-2 px-2 font-mono tabular-nums">{matches(metric)}</td>
                <td className="text-center py-2 ps-2 font-mono tabular-nums">{percent(metric.top1_accuracy)}</td>
              </tr>)}</tbody>
            </table>
          </div>
        </details>
        <details className="text-[10px] text-gray-500">
          <summary className="cursor-pointer">{t.provenance}</summary>
          <div className="mt-2 space-y-2 break-all">
            <p>{t.catalogue}: <code>{historical.intended_catalogue_profile}</code></p>
            <p>{t.source}: <code>{historical.source_csv}</code> · SHA-256: <code>{historical.source_csv_sha256}</code></p>
            <p>{t.correction}: <code>{historical.provenance_correction}</code> · SHA-256: <code>{historical.provenance_correction_sha256}</code></p>
          </div>
        </details>
      </div>
    </section>
    </>
  );
}
