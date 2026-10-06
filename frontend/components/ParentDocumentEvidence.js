import evidence from "../data/parent-document-comparison-2026-10-06.json";

const T = {
  en: {
    title: "Current retrieval comparison — Parent-document RAG",
    subtitle: "Frozen comparison on reused WISCO v3 cases, with the same catalogue and E5-small encoder in both methods.",
    heldout: "Historical held-out split (reused)", validation: "Validation split (reused)",
    dense: "Dense flat retrieval", parent: "Parent-document RAG",
    method: "Method", correct: "Exact matches / cases", accuracy: "Exact-code accuracy", ci: "95% Wilson interval",
    perLanguage: "Exact-code results by input language", language: "Language",
    languages: { ar: "Arabic", en: "English", hi: "Hindi", tl: "Tagalog", ur: "Urdu" },
    encoder: "Encoder", catalogue: "Catalogue", reranking: "LLM reranking", off: "Disabled",
    limitation: "These benchmark splits were used historically, and catalogue enrichment was informed by held-out errors. This is not an untouched test or Labour Force Survey field validation.",
    intervalNote: "Intervals describe case-level results. Multilingual titles share source groups and are not independent respondents.",
    comparisonNote: "Compare the two methods within this card. The historical benchmark below is a separate experiment. No improvement over another catalogue or encoder, including older E5-large results, is established here.",
    provenance: "Sources and frozen configuration", source: "Source report", parameters: "Frozen parameters",
  },
  ar: {
    title: "مقارنة الاسترجاع الحالية — RAG بالمستندات الأصلية",
    subtitle: "مقارنة بإعدادات ثابتة على حالات WISCO v3 المُعاد استخدامها، بالكتالوج نفسه ومُرمّز E5-small نفسه للطريقتين.",
    heldout: "مجموعة الاختبار المحجوزة تاريخيًا (مُعاد استخدامها)", validation: "مجموعة التحقق (مُعاد استخدامها)",
    dense: "الاسترجاع المسطح الكثيف", parent: "RAG بالمستندات الأصلية",
    method: "الطريقة", correct: "التطابقات الكاملة / الحالات", accuracy: "دقة الرمز الكامل", ci: "فاصل ويلسون 95%",
    perLanguage: "نتائج الرمز الكامل حسب لغة الإدخال", language: "اللغة",
    languages: { ar: "العربية", en: "الإنجليزية", hi: "الهندية", tl: "التاغالوغية", ur: "الأردية" },
    encoder: "المُرمّز", catalogue: "الكتالوج", reranking: "إعادة الترتيب بنموذج لغوي", off: "مُعطّلة",
    limitation: "استُخدمت مجموعات هذا المعيار سابقًا، واستند إثراء الكتالوج إلى أخطاء الاختبار المحجوز. هذا ليس اختبارًا لم يُستخدم من قبل أو تحققًا ميدانيًا لمسح القوى العاملة.",
    intervalNote: "تصف الفواصل النتائج على مستوى الحالات. تشترك العناوين متعددة اللغات في مجموعات المصدر ولا تمثل مستجيبين مستقلين.",
    comparisonNote: "قارن الطريقتين داخل هذه البطاقة. المعيار التاريخي أدناه تجربة منفصلة. لا تثبت هذه النتائج تحسنًا على كتالوج أو مُرمّز آخر، بما في ذلك نتائج E5-large السابقة.",
    provenance: "المصادر والإعدادات الثابتة", source: "تقرير المصدر", parameters: "المعاملات الثابتة",
  },
  ur: {
    title: "موجودہ بازیافت کا موازنہ — اصل دستاویز پر مبنی RAG",
    subtitle: "دوبارہ استعمال ہونے والے WISCO v3 کیسز پر مقررہ ترتیب کا موازنہ، دونوں طریقوں میں ایک ہی کیٹلاگ اور E5-small انکوڈر کے ساتھ۔",
    heldout: "تاریخی مخصوص ٹیسٹ اسپلٹ (دوبارہ استعمال شدہ)", validation: "توثیقی اسپلٹ (دوبارہ استعمال شدہ)",
    dense: "ڈینس فلیٹ بازیافت", parent: "اصل دستاویز پر مبنی RAG",
    method: "طریقہ", correct: "درست مماثلتیں / کیسز", accuracy: "مکمل کوڈ کی درستگی", ci: "95% ولسن وقفہ",
    perLanguage: "ان پٹ زبان کے لحاظ سے مکمل کوڈ کے نتائج", language: "زبان",
    languages: { ar: "عربی", en: "انگریزی", hi: "ہندی", tl: "ٹیگالوگ", ur: "اردو" },
    encoder: "انکوڈر", catalogue: "کیٹلاگ", reranking: "LLM دوبارہ ترتیب", off: "غیر فعال",
    limitation: "یہ بینچ مارک اسپلٹس پہلے استعمال ہو چکے ہیں، اور کیٹلاگ میں بہتری تاریخی ٹیسٹ کی غلطیوں سے متاثر تھی۔ یہ پہلے کبھی نہ دیکھا گیا ٹیسٹ یا لیبر فورس سروے کی فیلڈ توثیق نہیں ہے۔",
    intervalNote: "وقفے کیس کی سطح کے نتائج کو بیان کرتے ہیں۔ کثیر لسانی عنوانات مشترک ماخذ گروہوں سے تعلق رکھتے ہیں اور آزاد جواب دہندگان نہیں ہیں۔",
    comparisonNote: "اس کارڈ میں موجود دونوں طریقوں کا موازنہ کریں۔ نیچے تاریخی بینچ مارک ایک الگ تجربہ ہے۔ یہ نتائج کسی دوسرے کیٹلاگ یا انکوڈر، بشمول پرانے E5-large نتائج، پر بہتری ثابت نہیں کرتے۔",
    provenance: "ماخذ اور مقررہ ترتیب", source: "ماخذ رپورٹ", parameters: "مقررہ پیرامیٹرز",
  },
  hi: {
    title: "वर्तमान पुनर्प्राप्ति तुलना — मूल दस्तावेज़ आधारित RAG",
    subtitle: "दोबारा इस्तेमाल किए गए WISCO v3 मामलों पर निश्चित विन्यास की तुलना, दोनों विधियों में समान कैटलॉग और E5-small एन्कोडर के साथ।",
    heldout: "ऐतिहासिक आरक्षित परीक्षण विभाजन (पुनः उपयोग)", validation: "सत्यापन विभाजन (पुनः उपयोग)",
    dense: "डेंस फ्लैट पुनर्प्राप्ति", parent: "मूल दस्तावेज़ आधारित RAG",
    method: "विधि", correct: "सटीक मिलान / मामले", accuracy: "सटीक कोड की शुद्धता", ci: "95% विल्सन अंतराल",
    perLanguage: "इनपुट भाषा के अनुसार सटीक कोड परिणाम", language: "भाषा",
    languages: { ar: "अरबी", en: "अंग्रेज़ी", hi: "हिंदी", tl: "टैगालोग", ur: "उर्दू" },
    encoder: "एन्कोडर", catalogue: "कैटलॉग", reranking: "LLM पुनः रैंकिंग", off: "अक्षम",
    limitation: "ये बेंचमार्क विभाजन पहले इस्तेमाल किए गए हैं, और कैटलॉग का विस्तार पुराने आरक्षित परीक्षण की त्रुटियों से प्रभावित था। यह अछूता परीक्षण या श्रम बल सर्वेक्षण का क्षेत्र सत्यापन नहीं है।",
    intervalNote: "अंतराल मामले के स्तर के परिणाम बताते हैं। बहुभाषी शीर्षक समान स्रोत समूह साझा करते हैं और स्वतंत्र उत्तरदाता नहीं हैं।",
    comparisonNote: "इस कार्ड की दोनों विधियों की तुलना करें। नीचे ऐतिहासिक बेंचमार्क एक अलग प्रयोग है। यहाँ किसी दूसरे कैटलॉग या एन्कोडर, पुराने E5-large परिणामों सहित, से बेहतर प्रदर्शन सिद्ध नहीं होता।",
    provenance: "स्रोत और निश्चित विन्यास", source: "स्रोत रिपोर्ट", parameters: "निश्चित पैरामीटर",
  },
  tl: {
    title: "Kasalukuyang paghahambing ng retrieval — Parent-document RAG",
    subtitle: "Paghahambing gamit ang nakapirming konpigurasyon sa muling ginamit na WISCO v3 cases, na may parehong catalogue at E5-small encoder para sa dalawang paraan.",
    heldout: "Makasaysayang held-out split (muling ginamit)", validation: "Validation split (muling ginamit)",
    dense: "Dense flat retrieval", parent: "Parent-document RAG",
    method: "Paraan", correct: "Eksaktong tugma / kaso", accuracy: "Katumpakan ng eksaktong code", ci: "95% Wilson interval",
    perLanguage: "Eksaktong-code na resulta ayon sa wika ng input", language: "Wika",
    languages: { ar: "Arabic", en: "English", hi: "Hindi", tl: "Tagalog", ur: "Urdu" },
    encoder: "Encoder", catalogue: "Catalogue", reranking: "LLM reranking", off: "Naka-disable",
    limitation: "Nagamit na dati ang mga benchmark split na ito, at naimpluwensiyahan ng mga dating held-out error ang pagpapalawak ng catalogue. Hindi ito hindi pa nagamit na pagsusulit o field validation ng Labour Force Survey.",
    intervalNote: "Inilalarawan ng mga interval ang resulta sa antas ng kaso. May magkakaparehong source group ang mga multilingguwal na pamagat at hindi sila mga independiyenteng respondent.",
    comparisonNote: "Ihambing ang dalawang paraan sa card na ito. Hiwalay na eksperimento ang makasaysayang benchmark sa ibaba. Hindi pinatutunayan dito ang pagbuti kumpara sa ibang catalogue o encoder, kasama ang mga naunang E5-large result.",
    provenance: "Mga source at nakapirming konpigurasyon", source: "Source report", parameters: "Nakapirming parameter",
  },
};

const METHODS = ["dense_flat", "parent_document_rag"];
const percent = value => `${(value * 100).toFixed(2)}%`;
const count = value => value.toLocaleString();
const matches = metric => `${count(metric.top1_correct)}/${count(metric.n)}`;

export default function ParentDocumentEvidence({ language = "en" }) {
  const t = T[language === "ar-gulf" ? "ar" : language] || T.en;
  const methodName = method => method === "dense_flat" ? t.dense : t.parent;
  return (
    <section aria-label={t.title} className="bg-white border border-emerald-200 rounded-xl overflow-hidden shadow-sm">
      <div className="px-4 py-2.5 border-b border-emerald-100 bg-emerald-50">
        <h3 className="text-xs font-semibold text-emerald-800 uppercase tracking-widest">{t.title}</h3>
      </div>
      <div className="px-4 py-4 space-y-5">
        <p className="text-xs text-gray-600">{t.subtitle}</p>
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
                    <th scope="row" className="text-start py-2 pe-3 font-medium text-gray-800">{methodName(method)}</th>
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
  );
}
