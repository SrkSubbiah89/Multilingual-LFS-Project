/**
 * Report page — bilingual LFS survey employment report.
 *
 * Loads on mount via GET /survey/sessions/{id}/report.
 * Displays the generated English and Arabic report side-by-side,
 * plus the employment profile and data-collector recommendations.
 *
 * URL param: ?session=<sessionId>
 */

import { useState, useEffect } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { getReport } from "../components/api";
import LanguageToggle from "../components/LanguageToggle";

// ── i18n strings ──────────────────────────────────────────────────────────────

const T = {
  en: {
    title: "Survey Report",
    loading: "Generating your report…",
    errorTitle: "Report unavailable",
    profile: "Employment Profile",
    reportHeading: "Summary",
    recommendationsHeading: "Data Collector Notes",
    qualityLabel: "Quality",
    generatedAt: "Generated",
    backToHome: "Back to home",
    fields: {
      employment_status: "Employment status",
      job_title: "Job title",
      isco_code: "ISCO-08 code",
      isco_confidence: "ISCO confidence",
      industry: "Industry / sector",
      hours_per_week: "Hours per week",
      employment_type: "Employment type",
    },
    qualityStatus: {
      pass: "Passed",
      fail: "Needs review",
      escalated: "Escalated",
      unknown: "Not assessed",
    },
    signOut: "Sign out",
  },
  ar: {
    title: "تقرير المسح",
    loading: "جارٍ إعداد تقريرك…",
    errorTitle: "التقرير غير متاح",
    profile: "الملف الوظيفي",
    reportHeading: "الملخص",
    recommendationsHeading: "ملاحظات جامع البيانات",
    qualityLabel: "الجودة",
    generatedAt: "تاريخ الإنشاء",
    backToHome: "العودة إلى الرئيسية",
    fields: {
      employment_status: "حالة التوظيف",
      job_title: "المسمى الوظيفي",
      isco_code: "رمز ISCO-08",
      isco_confidence: "ثقة ISCO",
      industry: "القطاع / الصناعة",
      hours_per_week: "ساعات العمل أسبوعيًا",
      employment_type: "نوع التوظيف",
    },
    qualityStatus: {
      pass: "اجتاز",
      fail: "يحتاج مراجعة",
      escalated: "محال للمراجعة",
      unknown: "لم يُقيَّم",
    },
    signOut: "تسجيل الخروج",
  },
};

const QUALITY_COLOURS = {
  pass:      "bg-green-50  border-green-200  text-green-800",
  fail:      "bg-amber-50  border-amber-200  text-amber-800",
  escalated: "bg-red-50    border-red-200    text-red-800",
  unknown:   "bg-gray-50   border-gray-200   text-gray-600",
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function ReportPage() {
  const router = useRouter();
  const { session: sessionId } = router.query;

  const [lang, setLang] = useState("en");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const t = T[lang];
  const dir = lang === "ar" ? "rtl" : "ltr";

  // ── Auth guard + load report ───────────────────────────────────────────────

  useEffect(() => {
    if (!sessionId) return;

    const token = localStorage.getItem("lfs_token");
    const storedLang = localStorage.getItem("lfs_lang") || "en";
    setLang(storedLang);

    if (!token) {
      router.replace("/");
      return;
    }

    getReport(token, sessionId)
      .then((data) => {
        setReport(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || "Failed to load report.");
        setLoading(false);
      });
  }, [sessionId, router]);

  function handleSignOut() {
    localStorage.removeItem("lfs_token");
    localStorage.removeItem("lfs_lang");
    router.push("/");
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  const reportText = lang === "ar" ? report?.report_ar : report?.report_en;
  const recText    = lang === "ar" ? report?.recommendations_ar : report?.recommendations_en;
  const qualityKey = report?.quality_status || "unknown";
  const qualityColour = QUALITY_COLOURS[qualityKey] || QUALITY_COLOURS.unknown;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <>
      <Head>
        <title>{t.title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </Head>

      <div dir={dir} className="min-h-screen bg-gray-50">

        {/* ── Header ──────────────────────────────────────────────────────── */}
        <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-3">
            <span className="text-xl">📋</span>
            <span className="font-semibold text-gray-800">{t.title}</span>
            {sessionId && (
              <span className="text-xs text-gray-400 hidden sm:inline">
                #{sessionId}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <LanguageToggle lang={lang} onToggle={setLang} />
            <button
              onClick={handleSignOut}
              className="text-sm text-gray-500 hover:text-gray-800 underline"
            >
              {t.signOut}
            </button>
          </div>
        </header>

        {/* ── Body ────────────────────────────────────────────────────────── */}
        <main className="max-w-3xl mx-auto px-4 py-8 space-y-6">

          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-24 gap-4 text-gray-500">
              <svg
                className="animate-spin h-8 w-8 text-blue-500"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              <p className="text-sm">{t.loading}</p>
            </div>
          )}

          {/* Error */}
          {!loading && error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-4 rounded-xl">
              <p className="font-semibold">{t.errorTitle}</p>
              <p className="text-sm mt-1">{error}</p>
              <button
                onClick={() => router.push("/")}
                className="mt-3 text-sm underline text-red-600 hover:text-red-800"
              >
                {t.backToHome}
              </button>
            </div>
          )}

          {/* Report */}
          {!loading && report && (
            <>
              {/* Quality badge */}
              <div className="flex items-center justify-between flex-wrap gap-2">
                <span
                  className={`inline-flex items-center gap-1.5 text-xs font-medium border px-3 py-1 rounded-full ${qualityColour}`}
                >
                  {t.qualityLabel}: {t.qualityStatus[qualityKey] || qualityKey}
                  {report.quality_score != null && (
                    <span className="opacity-60">
                      ({Math.round(report.quality_score * 100)}%)
                    </span>
                  )}
                </span>
                <span className="text-xs text-gray-400">
                  {t.generatedAt}: {new Date(report.generated_at).toLocaleString()}
                </span>
              </div>

              {/* Employment profile card */}
              <section className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
                <h2 className="text-sm font-semibold text-gray-700 bg-gray-50 border-b border-gray-200 px-4 py-2.5">
                  {t.profile}
                </h2>
                <dl className="divide-y divide-gray-100">
                  {Object.entries(t.fields).map(([key, label]) => {
                    const raw = key === "isco_confidence"
                      ? report.profile.isco_confidence != null
                        ? `${Math.round(report.profile.isco_confidence * 100)}%`
                        : null
                      : report.profile[key];
                    if (!raw) return null;
                    return (
                      <div key={key} className="flex gap-4 px-4 py-2.5 text-sm">
                        <dt className="w-44 flex-shrink-0 text-gray-500">{label}</dt>
                        <dd className="text-gray-800 font-medium">{raw}</dd>
                      </div>
                    );
                  })}
                </dl>
              </section>

              {/* Report narrative */}
              <section className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
                <h2 className="text-sm font-semibold text-gray-700 bg-gray-50 border-b border-gray-200 px-4 py-2.5">
                  {t.reportHeading}
                </h2>
                <div
                  className="px-4 py-4 text-sm text-gray-800 leading-relaxed whitespace-pre-wrap"
                  dir={dir}
                >
                  {reportText}
                </div>
              </section>

              {/* Recommendations */}
              <section className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
                <h2 className="text-sm font-semibold text-gray-700 bg-gray-50 border-b border-gray-200 px-4 py-2.5">
                  {t.recommendationsHeading}
                </h2>
                <div
                  className="px-4 py-4 text-sm text-gray-800 leading-relaxed whitespace-pre-wrap"
                  dir={dir}
                >
                  {recText}
                </div>
              </section>

              {/* Back link */}
              <div className="text-center pt-2">
                <button
                  onClick={() => router.push("/")}
                  className="text-sm text-blue-600 hover:text-blue-800 underline"
                >
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
