/**
 * Chat page — LFS survey conversation.
 *
 * On mount: loads JWT from localStorage, creates a survey session, then drives
 * a multi-turn conversation through POST /survey/sessions/{id}/message.
 *
 * Each turn response contains:
 *   reply, state, detected_language, is_code_switched, entities,
 *   isco_classifications, session_completed
 *
 * When session_completed is true the input is locked and a completion banner
 * is shown.
 *
 * Supports English (LTR) and Arabic (RTL) with a toggle that also updates the
 * active session language preference.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { createSession, sendMessage } from "../components/api";
import LanguageToggle from "../components/LanguageToggle";

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
  },
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function ChatPage() {
  const router = useRouter();

  const [lang, setLang] = useState("en");
  const [token, setToken] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]); // { role, text, meta? }
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [pageError, setPageError] = useState("");
  const [initialising, setInitialising] = useState(true);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const t = T[lang];
  const dir = lang === "ar" ? "rtl" : "ltr";

  // ── Auth guard + session init ──────────────────────────────────────────────

  useEffect(() => {
    const storedToken = localStorage.getItem("lfs_token");
    const storedLang = localStorage.getItem("lfs_lang") || "en";

    if (!storedToken) {
      router.replace("/");
      return;
    }

    setToken(storedToken);
    setLang(storedLang);

    createSession(storedToken, storedLang)
      .then((session) => {
        setSessionId(session.id);
        setInitialising(false);
        // Focus input after session is ready
        setTimeout(() => inputRef.current?.focus(), 100);
      })
      .catch(() => {
        setPageError(T[storedLang].sessionError);
        setInitialising(false);
      });
  }, [router]);

  // ── Auto-scroll to latest message ─────────────────────────────────────────

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Send a message ─────────────────────────────────────────────────────────

  const handleSend = useCallback(
    async (e) => {
      e?.preventDefault();
      const text = input.trim();
      if (!text || sending || completed || !sessionId) return;

      // Optimistically add user message
      setMessages((prev) => [...prev, { role: "user", text }]);
      setInput("");
      setSending(true);

      try {
        const res = await sendMessage(token, sessionId, text);

        // Determine display language from backend detection
        if (res.detected_language === "ar" || res.detected_language === "en") {
          setLang(res.detected_language);
          localStorage.setItem("lfs_lang", res.detected_language);
        }

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            text: res.reply,
            meta: {
              state: res.state,
              detectedLang: res.detected_language,
              isCodeSwitched: res.is_code_switched,
              entities: res.entities || [],
              isco: res.isco_classifications || [],
            },
          },
        ]);

        if (res.session_completed) {
          setCompleted(true);
        }
      } catch {
        // Remove the optimistic user message on failure
        setMessages((prev) => prev.slice(0, -1));
        setInput(text);
        // Show inline error as a transient assistant message
        setMessages((prev) => [
          ...prev,
          { role: "error", text: T[lang].sendError },
        ]);
        setTimeout(
          () => setMessages((prev) => prev.filter((m) => m.role !== "error")),
          4000
        );
      } finally {
        setSending(false);
        inputRef.current?.focus();
      }
    },
    [input, sending, completed, sessionId, token, lang]
  );

  // Send on Enter (Shift+Enter inserts newline)
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Sign out ───────────────────────────────────────────────────────────────

  function handleSignOut() {
    localStorage.removeItem("lfs_token");
    localStorage.removeItem("lfs_lang");
    router.push("/");
  }

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

      <div
        dir={dir}
        className="h-screen flex flex-col bg-gray-50"
      >
        {/* ── Header ────────────────────────────────────────────────────── */}
        <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between shadow-sm flex-shrink-0">
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

        {/* ── Message area ──────────────────────────────────────────────── */}
        <main className="flex-1 overflow-y-auto px-4 py-4 space-y-4">

          {/* Initialising spinner */}
          {initialising && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-3 text-gray-500">
                <svg
                  className="animate-spin h-8 w-8 mx-auto text-blue-500"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <p className="text-sm">{t.startingSession}</p>
              </div>
            </div>
          )}

          {/* Page-level error (session creation failed) */}
          {pageError && (
            <div className="max-w-lg mx-auto bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">
              {pageError}
            </div>
          )}

          {/* Messages */}
          {!initialising && messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} lang={lang} t={t} />
          ))}

          {/* Sending indicator */}
          {sending && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-none px-4 py-2.5 shadow-sm max-w-[60%]">
                <TypingDots />
              </div>
            </div>
          )}

          {/* Completion banner */}
          {completed && (
            <div className="max-w-lg mx-auto bg-green-50 border border-green-200 text-green-800 text-sm text-center px-4 py-3 rounded-lg font-medium">
              ✓ {t.completedBanner}
            </div>
          )}

          <div ref={messagesEndRef} />
        </main>

        {/* ── Input bar ─────────────────────────────────────────────────── */}
        <footer className="bg-white border-t border-gray-200 px-4 py-3 flex-shrink-0">
          <form
            onSubmit={handleSend}
            className="flex gap-2 items-end max-w-3xl mx-auto"
          >
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={completed ? "" : t.inputPlaceholder}
              disabled={completed || sending || initialising || !!pageError}
              rows={1}
              className="flex-1 resize-none px-4 py-2.5 border border-gray-300 rounded-xl text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                         disabled:bg-gray-50 disabled:text-gray-400 placeholder:text-gray-400
                         max-h-32 overflow-y-auto"
              style={{ lineHeight: "1.5" }}
            />
            <button
              type="submit"
              disabled={!input.trim() || sending || completed || initialising || !!pageError}
              className="flex-shrink-0 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40
                         text-white font-medium rounded-xl text-sm transition-colors"
            >
              {sending ? "…" : t.sendButton}
            </button>
          </form>
        </footer>
      </div>
    </>
  );
}

// ── MessageBubble ─────────────────────────────────────────────────────────────

function MessageBubble({ msg, lang, t }) {
  const isUser = msg.role === "user";
  const isError = msg.role === "error";

  if (isError) {
    return (
      <div className="flex justify-center">
        <p className="text-xs text-red-500 bg-red-50 px-3 py-1 rounded-full">
          {msg.text}
        </p>
      </div>
    );
  }

  return (
    <div className={`flex flex-col gap-1 ${isUser ? "items-end" : "items-start"}`}>

      {/* Bubble */}
      <div
        className={`
          px-4 py-2.5 rounded-2xl text-sm leading-relaxed shadow-sm max-w-[75%]
          ${isUser
            ? "bg-blue-600 text-white rounded-br-none"
            : "bg-white border border-gray-200 text-gray-800 rounded-bl-none"
          }
        `}
        dir={lang === "ar" ? "rtl" : "ltr"}
      >
        {msg.text}
      </div>

      {/* Metadata (assistant messages only) */}
      {!isUser && msg.meta && (
        <div className="max-w-[75%] space-y-1">

          {/* Code-switch notice */}
          {msg.meta.isCodeSwitched && (
            <span className="inline-block text-xs text-amber-600 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full">
              ⇄ {t.codeSwitched}
            </span>
          )}

          {/* ISCO classifications */}
          {msg.meta.isco.length > 0 && (
            <div className="space-y-1">
              {msg.meta.isco.map((c, i) => (
                <IscoTag key={i} clf={c} lang={lang} t={t} />
              ))}
            </div>
          )}

          {/* Named entities */}
          {msg.meta.entities.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-0.5">
              {msg.meta.entities.map((e, i) => (
                <EntityChip key={i} entity={e} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── IscoTag ───────────────────────────────────────────────────────────────────

function IscoTag({ clf, lang, t }) {
  const title = lang === "ar" ? clf.primary_title_ar : clf.primary_title_en;
  const pct = Math.round(clf.confidence * 100);

  return (
    <div className="inline-flex items-center gap-2 bg-blue-50 border border-blue-200 text-blue-800 text-xs px-2.5 py-1 rounded-lg">
      <span className="font-mono font-semibold">{clf.primary_code}</span>
      <span>{title}</span>
      <span className="text-blue-500">{pct}%</span>
    </div>
  );
}

// ── EntityChip ────────────────────────────────────────────────────────────────

const LABEL_COLOURS = {
  JOB_TITLE:         "bg-purple-50 border-purple-200 text-purple-700",
  ORGANIZATION:      "bg-yellow-50 border-yellow-200 text-yellow-700",
  LOCATION:          "bg-green-50  border-green-200  text-green-700",
  INDUSTRY:          "bg-orange-50 border-orange-200 text-orange-700",
  EMPLOYMENT_STATUS: "bg-teal-50   border-teal-200   text-teal-700",
  DURATION:          "bg-pink-50   border-pink-200   text-pink-700",
  HOURS:             "bg-indigo-50 border-indigo-200 text-indigo-700",
  PERSON:            "bg-gray-50   border-gray-200   text-gray-700",
};

function EntityChip({ entity }) {
  const colours = LABEL_COLOURS[entity.label] || "bg-gray-50 border-gray-200 text-gray-700";
  return (
    <span className={`inline-flex items-center gap-1 text-xs border px-2 py-0.5 rounded-full ${colours}`}>
      <span className="font-medium">{entity.text}</span>
      <span className="opacity-60 uppercase tracking-wide text-[10px]">{entity.label}</span>
    </span>
  );
}

// ── TypingDots ────────────────────────────────────────────────────────────────

function TypingDots() {
  return (
    <span className="flex items-center gap-1 h-4">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </span>
  );
}
