/** Occupation display state comes from each server response, never cached UI codes. */
const MISSING_TITLES = new Set(["", "n/a", "na", "none", "null", "unknown", "not_applicable",
  "not applicable", "never_worked", "refused", "prefer_not_to_say", "prefer not to say"]);

export function getOccupationAnalysis(response = {}) {
  const fields = response.collected_data || {};
  const previous = ["unemployed", "not_in_labour_force"].includes(fields.employment_status);
  const field = previous ? "last_job_title" : "job_title";
  const raw = typeof fields[field] === "string" ? fields[field].trim() : "";
  const title = MISSING_TITLES.has(raw.toLowerCase()) ? "" : raw;
  const classifications = response.isco_classifications || [];
  const notApplicable = previous && (fields.ever_worked === "never_worked" || raw === "never_worked");
  const status = notApplicable ? "not_applicable"
    : classifications.some(item => item.primary_code) ? "classified"
    : title ? "unclassified"
    : raw || ["validating", "completing"].includes(response.state) ? "not_provided" : "not_collected";
  return { source: previous ? "previous" : "current", field, title, status };
}

const LABELS = {
  en: {
    current: "Current occupation — ISCO-08", previous: "Previous occupation — ISCO-08",
    jobTitle: "Job title", notApplicable: "ISCO does not apply because you have never worked.",
    unclassified: "No classification is available for this occupation yet. It needs classification or human review.",
    notProvided: "An occupation was not provided, so no ISCO classification is available.",
    notCollected: "An occupation has not been collected yet.",
    parent: "Parent-document RAG", duties: "Parent-document RAG with duties", saved: "Saved classification",
    hierarchy: "ISCO code hierarchy", stages: "Hierarchical retrieval stages", score: "Similarity score",
    confidence: "Reported confidence", scoreNote: "This score is not a calibrated probability of correctness.",
    review: "Human review required", alternatives: "Alternatives", retrieved: "Retrieved classification",
  },
  ar: {
    current: "المهنة الحالية — ISCO-08", previous: "المهنة السابقة — ISCO-08",
    jobTitle: "المسمى الوظيفي", notApplicable: "لا ينطبق تصنيف ISCO لأنك لم تعمل من قبل.",
    unclassified: "لا يتوفر تصنيف لهذه المهنة بعد. تحتاج إلى تصنيف أو مراجعة بشرية.",
    notProvided: "لم تُقدَّم مهنة، لذلك لا يتوفر تصنيف ISCO.",
    notCollected: "لم تُجمع معلومات المهنة بعد.",
    parent: "RAG بالمستندات الأصلية", duties: "RAG بالمستندات الأصلية مع المهام", saved: "التصنيف المحفوظ",
    hierarchy: "التسلسل الهرمي لرمز ISCO", stages: "مراحل الاسترجاع الهرمي", score: "درجة التشابه",
    confidence: "الثقة المُبلغ عنها", scoreNote: "هذه الدرجة ليست احتمالًا معايرًا لصحة التصنيف.",
    review: "المراجعة البشرية مطلوبة", alternatives: "البدائل", retrieved: "التصنيف المسترجع",
  },
  ur: {
    current: "موجودہ پیشہ — ISCO-08", previous: "سابقہ پیشہ — ISCO-08",
    jobTitle: "عہدہ", notApplicable: "ISCO لاگو نہیں ہوتا کیونکہ آپ نے کبھی کام نہیں کیا۔",
    unclassified: "اس پیشے کی درجہ بندی ابھی دستیاب نہیں۔ اسے درجہ بندی یا انسانی جائزے کی ضرورت ہے۔",
    notProvided: "پیشہ فراہم نہیں کیا گیا، اس لیے ISCO درجہ بندی دستیاب نہیں۔",
    notCollected: "پیشے کی معلومات ابھی جمع نہیں ہوئیں۔",
    parent: "اصل دستاویز پر مبنی RAG", duties: "فرائض کے ساتھ اصل دستاویز پر مبنی RAG", saved: "محفوظ درجہ بندی",
    hierarchy: "ISCO کوڈ کا درجہ وار ڈھانچہ", stages: "درجہ وار بازیافت کے مراحل", score: "مماثلت کا اسکور",
    confidence: "درج شدہ اعتماد", scoreNote: "یہ اسکور درستگی کا جانچا ہوا احتمالی پیمانہ نہیں ہے۔",
    review: "انسانی جائزہ درکار ہے", alternatives: "متبادل", retrieved: "حاصل کردہ درجہ بندی",
  },
  hi: {
    current: "वर्तमान पेशा — ISCO-08", previous: "पिछला पेशा — ISCO-08",
    jobTitle: "पद का नाम", notApplicable: "ISCO लागू नहीं होता क्योंकि आपने कभी काम नहीं किया है।",
    unclassified: "इस पेशे का वर्गीकरण अभी उपलब्ध नहीं है। इसे वर्गीकरण या मानव समीक्षा की आवश्यकता है।",
    notProvided: "पेशा नहीं दिया गया, इसलिए ISCO वर्गीकरण उपलब्ध नहीं है।",
    notCollected: "पेशे की जानकारी अभी एकत्र नहीं हुई है।",
    parent: "मूल दस्तावेज़ आधारित RAG", duties: "कार्य विवरण के साथ मूल दस्तावेज़ आधारित RAG", saved: "सहेजा गया वर्गीकरण",
    hierarchy: "ISCO कोड का पदानुक्रम", stages: "पदानुक्रमित पुनर्प्राप्ति चरण", score: "समानता स्कोर",
    confidence: "दर्ज विश्वास स्तर", scoreNote: "यह स्कोर सही होने की अंशांकित संभावना नहीं है।",
    review: "मानव समीक्षा आवश्यक", alternatives: "विकल्प", retrieved: "प्राप्त वर्गीकरण",
  },
  tl: {
    current: "Kasalukuyang trabaho — ISCO-08", previous: "Dating trabaho — ISCO-08",
    jobTitle: "Titulo ng trabaho", notApplicable: "Hindi naaangkop ang ISCO dahil hindi ka pa nagtrabaho.",
    unclassified: "Wala pang klasipikasyon para sa trabahong ito. Kailangan itong uriin o suriin ng tao.",
    notProvided: "Walang ibinigay na trabaho, kaya walang ISCO classification.",
    notCollected: "Hindi pa nakokolekta ang impormasyon tungkol sa trabaho.",
    parent: "Parent-document RAG", duties: "Parent-document RAG na may mga tungkulin", saved: "Naka-save na klasipikasyon",
    hierarchy: "Hirarkiya ng ISCO code", stages: "Mga yugto ng hierarchical retrieval", score: "Similarity score",
    confidence: "Naitalang confidence", scoreNote: "Ang score na ito ay hindi calibrated na posibilidad ng pagiging tama.",
    review: "Kailangan ng pagsusuri ng tao", alternatives: "Mga alternatibo", retrieved: "Nakuha na klasipikasyon",
  },
};

export function getOccupationLabels(language) {
  return LABELS[language === "ar-gulf" ? "ar" : language] || LABELS.en;
}

export function getIscoPresentation(classification, language) {
  const labels = getOccupationLabels(language);
  const method = classification.method;
  const parent = method === "isco_parent_document_rag" || method === "isco_parent_document_duties_rag";
  const saved = method === "cached";
  const hierarchical = ["hierarchical_llm", "hierarchical_semantic"].includes(method)
    && Number.isFinite(classification.stage_confidences?.stage4);
  return {
    label: saved ? labels.saved : parent ? (method.endsWith("_duties_rag") ? labels.duties : labels.parent) : null,
    showStageScores: hierarchical, similarity: parent || saved,
  };
}
