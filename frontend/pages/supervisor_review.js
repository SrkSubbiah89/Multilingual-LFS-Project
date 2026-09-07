/**
 * Supervisor Review Dashboard — HITL occupation coding queue.
 *
 * Loads pending items from GET /survey/hitl/queue, displays them in a
 * sortable table, and allows the reviewer to approve / correct / reject
 * each escalation via POST /survey/hitl/review.
 *
 * Supports English (LTR) and Arabic (RTL) via the shared LanguageToggle.
 */

import { useState, useEffect, useCallback } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { getHitlQueue, submitHitlReview } from "../components/api";
import LanguageToggle from "../components/LanguageToggle";

// ── i18n strings ──────────────────────────────────────────────────────────────

const T = {
  en: {
    title: "Supervisor Review — HITL Queue",
    heading: "Occupation Coding Review",
    filterAll: "All",
    filterPending: "Pending",
    filterReviewed: "Reviewed",
    filterRejected: "Rejected",
    colId: "ID",
    colSession: "Session",
    colRawText: "Raw Text",
    colAiCode: "AI Code",
    colConfidence: "Confidence",
    colPriority: "Priority",
    colStatus: "Status",
    colActions: "Actions",
    btnApprove: "Approve",
    btnCorrect: "Correct",
    btnReject: "Reject",
    btnSubmit: "Submit",
    btnCancel: "Cancel",
    correctCodeLabel: "Override ISCO code (4 digits)",
    notesLabel: "Notes (optional)",
    noItems: "No items in this queue.",
    loading: "Loading queue…",
    errorLoad: "Could not load the HITL queue.",
    errorSubmit: "Could not submit the review. Please try again.",
    successMsg: "Review submitted.",
    signOut: "Sign out",
    priorityHigh: "HIGH",
    priorityMedium: "MEDIUM",
    statusPending: "Pending",
    statusReviewed: "Reviewed",
    statusRejected: "Rejected",
    aiReasoning: "AI reasoning",
    hierarchyPath: "Hierarchy",
  },
  ar: {
    title: "لوحة مراجعة المشرف — قائمة HITL",
    heading: "مراجعة ترميز المهن",
    filterAll: "الكل",
    filterPending: "قيد الانتظار",
    filterReviewed: "تمت المراجعة",
    filterRejected: "مرفوض",
    colId: "المعرف",
    colSession: "الجلسة",
    colRawText: "النص الأصلي",
    colAiCode: "رمز الذكاء الاصطناعي",
    colConfidence: "الثقة",
    colPriority: "الأولوية",
    colStatus: "الحالة",
    colActions: "الإجراءات",
    btnApprove: "موافقة",
    btnCorrect: "تصحيح",
    btnReject: "رفض",
    btnSubmit: "إرسال",
    btnCancel: "إلغاء",
    correctCodeLabel: "رمز ISCO البديل (4 أرقام)",
    notesLabel: "ملاحظات (اختياري)",
    noItems: "لا توجد عناصر في هذه القائمة.",
    loading: "جارٍ تحميل القائمة…",
    errorLoad: "تعذّر تحميل قائمة HITL.",
    errorSubmit: "تعذّر إرسال المراجعة. يرجى المحاولة مرة أخرى.",
    successMsg: "تم إرسال المراجعة.",
    signOut: "تسجيل الخروج",
    priorityHigh: "عالية",
    priorityMedium: "متوسطة",
    statusPending: "قيد الانتظار",
    statusReviewed: "تمت المراجعة",
    statusRejected: "مرفوض",
    aiReasoning: "تفسير الذكاء الاصطناعي",
    hierarchyPath: "التسلسل الهرمي",
  },
};

const STATUS_COLOURS = {
  pending:  "bg-amber-100 text-amber-800",
  reviewed: "bg-green-100 text-green-800",
  rejected: "bg-red-100   text-red-800",
};

const PRIORITY_COLOURS = {
  HIGH:   "bg-red-100   text-red-800   font-semibold",
  MEDIUM: "bg-blue-100  text-blue-800",
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function SupervisorReview() {
  const router = useRouter();
  const [lang, setLang]           = useState("en");
  const [token, setToken]         = useState(null);
  const [items, setItems]         = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [statusFilter, setFilter] = useState("pending");

  // Inline-review state (one item at a time)
  const [activeId, setActiveId]     = useState(null);
  const [activeAction, setAction]   = useState(null); // "approve"|"correct"|"reject"
  const [codeInput, setCodeInput]   = useState("");
  const [notesInput, setNotes]      = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [flashMsg, setFlashMsg]     = useState(null);

  const t = T[lang];
  const isRtl = lang === "ar";

  // ── Auth ──────────────────────────────────────────────────────────────────

  useEffect(() => {
    const stored = localStorage.getItem("lfs_token");
    if (!stored) {
      router.replace("/");
      return;
    }
    setToken(stored);
  }, [router]);

  // ── Load queue ────────────────────────────────────────────────────────────

  const loadQueue = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getHitlQueue(token, statusFilter);
      setItems(data);
    } catch (e) {
      setError(t.errorLoad);
    } finally {
      setLoading(false);
    }
  }, [token, statusFilter, t.errorLoad]);

  useEffect(() => {
    loadQueue();
  }, [loadQueue]);

  // ── Sign out ──────────────────────────────────────────────────────────────

  function signOut() {
    localStorage.removeItem("lfs_token");
    router.replace("/");
  }

  // ── Inline review helpers ─────────────────────────────────────────────────

  function openReview(id, action) {
    setActiveId(id);
    setAction(action);
    setCodeInput("");
    setNotes("");
  }

  function cancelReview() {
    setActiveId(null);
    setAction(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (submitting) return;

    // Validate code for "correct"
    if (activeAction === "correct" && !/^\d{4}$/.test(codeInput.trim())) {
      alert(t.correctCodeLabel);
      return;
    }

    setSubmitting(true);
    try {
      await submitHitlReview(
        token,
        activeId,
        activeAction,
        activeAction === "correct" ? codeInput.trim() : null,
        notesInput.trim() || null,
      );
      setFlashMsg(t.successMsg);
      setTimeout(() => setFlashMsg(null), 3000);
      cancelReview();
      loadQueue();
    } catch {
      setError(t.errorSubmit);
    } finally {
      setSubmitting(false);
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <>
      <Head>
        <title>{t.title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div dir={isRtl ? "rtl" : "ltr"} className="min-h-screen bg-gray-50">
        {/* ── Header ─────────────────────────────────────────────────────── */}
        <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
          <h1 className="text-lg font-semibold text-gray-800">{t.heading}</h1>
          <div className="flex items-center gap-3">
            <LanguageToggle lang={lang} onToggle={setLang} languages={["en", "ar"]} />
            <button
              onClick={signOut}
              className="text-sm text-gray-500 hover:text-gray-700 underline"
            >
              {t.signOut}
            </button>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 py-6">
          {/* ── Flash message ──────────────────────────────────────────── */}
          {flashMsg && (
            <div className="mb-4 rounded-md bg-green-50 border border-green-200 px-4 py-2 text-green-800 text-sm">
              {flashMsg}
            </div>
          )}

          {/* ── Error banner ───────────────────────────────────────────── */}
          {error && (
            <div className="mb-4 rounded-md bg-red-50 border border-red-200 px-4 py-2 text-red-800 text-sm">
              {error}
            </div>
          )}

          {/* ── Filter tabs ────────────────────────────────────────────── */}
          <div className="flex gap-2 mb-4">
            {[
              { key: "pending",  label: t.filterPending },
              { key: "reviewed", label: t.filterReviewed },
              { key: "rejected", label: t.filterRejected },
              { key: "all",      label: t.filterAll },
            ].map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
                  statusFilter === key
                    ? "bg-blue-600 text-white"
                    : "bg-white border border-gray-300 text-gray-600 hover:bg-gray-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* ── Table ──────────────────────────────────────────────────── */}
          {loading ? (
            <p className="text-gray-500 text-sm">{t.loading}</p>
          ) : items.length === 0 ? (
            <p className="text-gray-500 text-sm">{t.noItems}</p>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white shadow-sm">
              <table className="w-full text-sm text-left">
                <thead className="bg-gray-100 text-gray-600 text-xs uppercase tracking-wide">
                  <tr>
                    {[
                      t.colId, t.colSession, t.colRawText,
                      t.colAiCode, t.colConfidence, t.colPriority,
                      t.colStatus, t.colActions,
                    ].map((h) => (
                      <th key={h} className="px-4 py-3 whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {items.map((item) => (
                    <>
                      <tr key={item.id} className="hover:bg-gray-50">
                        <td className="px-4 py-3 font-mono text-gray-500">{item.id}</td>
                        <td className="px-4 py-3 text-gray-500">{item.session_id ?? "—"}</td>
                        <td className="px-4 py-3 max-w-xs">
                          <div className="truncate" title={item.raw_text}>{item.raw_text}</div>
                          {item.hierarchy_path && (
                            <div className="text-xs text-gray-400 mt-0.5">{t.hierarchyPath}: {item.hierarchy_path}</div>
                          )}
                        </td>
                        <td className="px-4 py-3 font-mono font-semibold text-blue-700">
                          {item.ai_code ?? "—"}
                        </td>
                        <td className="px-4 py-3">
                          {item.confidence != null
                            ? `${(item.confidence * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-xs ${PRIORITY_COLOURS[item.priority] ?? ""}`}>
                            {item.priority === "HIGH" ? t.priorityHigh : t.priorityMedium}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-xs ${STATUS_COLOURS[item.status] ?? ""}`}>
                            {item.status === "pending"
                              ? t.statusPending
                              : item.status === "reviewed"
                              ? t.statusReviewed
                              : t.statusRejected}
                          </span>
                          {item.reviewer_code && (
                            <div className="text-xs text-gray-500 mt-0.5">→ {item.reviewer_code}</div>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {item.status === "pending" ? (
                            <div className="flex gap-1.5">
                              <button
                                onClick={() => openReview(item.id, "approve")}
                                className="px-2.5 py-1 rounded bg-green-500 hover:bg-green-600 text-white text-xs"
                              >
                                {t.btnApprove}
                              </button>
                              <button
                                onClick={() => openReview(item.id, "correct")}
                                className="px-2.5 py-1 rounded bg-blue-500 hover:bg-blue-600 text-white text-xs"
                              >
                                {t.btnCorrect}
                              </button>
                              <button
                                onClick={() => openReview(item.id, "reject")}
                                className="px-2.5 py-1 rounded bg-red-500 hover:bg-red-600 text-white text-xs"
                              >
                                {t.btnReject}
                              </button>
                            </div>
                          ) : (
                            <span className="text-gray-400 text-xs">—</span>
                          )}
                        </td>
                      </tr>

                      {/* ── Inline review form ── */}
                      {activeId === item.id && (
                        <tr key={`${item.id}-form`} className="bg-blue-50">
                          <td colSpan={8} className="px-6 py-4">
                            <form onSubmit={handleSubmit} className="flex flex-wrap gap-3 items-end">
                              <div className="text-sm font-medium text-blue-800 w-full">
                                {activeAction === "approve" && `${t.btnApprove} — ISCO ${item.ai_code}`}
                                {activeAction === "reject"  && t.btnReject}
                                {activeAction === "correct" && t.btnCorrect}
                              </div>

                              {activeAction === "correct" && (
                                <div className="flex flex-col gap-1">
                                  <label className="text-xs text-gray-600">{t.correctCodeLabel}</label>
                                  <input
                                    type="text"
                                    value={codeInput}
                                    onChange={(e) => setCodeInput(e.target.value)}
                                    placeholder="e.g. 2512"
                                    maxLength={4}
                                    pattern="\d{4}"
                                    required
                                    className="border border-gray-300 rounded px-2 py-1 text-sm w-28 font-mono"
                                  />
                                </div>
                              )}

                              <div className="flex flex-col gap-1">
                                <label className="text-xs text-gray-600">{t.notesLabel}</label>
                                <input
                                  type="text"
                                  value={notesInput}
                                  onChange={(e) => setNotes(e.target.value)}
                                  placeholder="…"
                                  className="border border-gray-300 rounded px-2 py-1 text-sm w-64"
                                />
                              </div>

                              <button
                                type="submit"
                                disabled={submitting}
                                className="px-4 py-1.5 rounded bg-blue-600 hover:bg-blue-700 text-white text-sm disabled:opacity-50"
                              >
                                {t.btnSubmit}
                              </button>
                              <button
                                type="button"
                                onClick={cancelReview}
                                className="px-4 py-1.5 rounded bg-gray-200 hover:bg-gray-300 text-gray-700 text-sm"
                              >
                                {t.btnCancel}
                              </button>
                            </form>

                            {/* AI reasoning (expandable) */}
                            {item.ai_reasoning && (
                              <details className="mt-3 text-xs text-gray-600">
                                <summary className="cursor-pointer text-blue-600 hover:underline">
                                  {t.aiReasoning}
                                </summary>
                                <p className="mt-1 bg-white rounded p-2 border border-gray-200">
                                  {item.ai_reasoning}
                                </p>
                              </details>
                            )}
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </main>
      </div>
    </>
  );
}
