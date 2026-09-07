/**
 * Landing + Login page — UAE Labour Force Survey AI Platform
 * White professional theme consistent with chat.js (#007a62 green, white bg)
 */

import { useState, useEffect } from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { requestOtp, verifyOtp, requestSmsOtp, verifySmsOtp } from "../components/api";

// ── Design tokens (mirrors chat.js) ──────────────────────────────────────────
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

// ── i18n ──────────────────────────────────────────────────────────────────────
const T = {
  en: {
    title: "Labour Force Survey",
    subtitle: "AI-Powered Employment Data Collection Platform",
    emailLabel: "Email address",
    emailPlaceholder: "you@example.com",
    emailButton: "Send verification code",
    otpLabel: "Verification code",
    otpPlaceholder: "Enter 6-digit code",
    otpSent: (e) => `Verification code sent to ${e}.`,
    otpButton: "Sign in",
    backLink: "Use a different email",
    resendLink: "Resend code",
    resendOk: "Code resent!",
    errorGeneric: "Something went wrong. Please try again.",
    loginHeading: "Sign in to begin",
    loginSub: "Your responses are confidential and protected.",
  },
  ar: {
    title: "مسح القوى العاملة",
    subtitle: "منصة جمع بيانات التوظيف بالذكاء الاصطناعي",
    emailLabel: "البريد الإلكتروني",
    emailPlaceholder: "example@email.com",
    emailButton: "إرسال رمز التحقق",
    otpLabel: "رمز التحقق",
    otpPlaceholder: "أدخل الرمز المكوّن من 6 أرقام",
    otpSent: (e) => `تم إرسال رمز التحقق إلى ${e}.`,
    otpButton: "تسجيل الدخول",
    backLink: "استخدام بريد إلكتروني مختلف",
    resendLink: "إعادة إرسال الرمز",
    resendOk: "تمت إعادة الإرسال!",
    errorGeneric: "حدث خطأ ما. يرجى المحاولة مرة أخرى.",
    loginHeading: "تسجيل الدخول للبدء",
    loginSub: "إجاباتك سرية ومحمية.",
  },
  ur: {
    title: "لیبر فورس سروے",
    subtitle: "AI سے چلنے والا ملازمت ڈیٹا جمع کرنے کا پلیٹ فارم",
    emailLabel: "ای میل ایڈریس",
    emailPlaceholder: "you@example.com",
    emailButton: "تصدیقی کوڈ بھیجیں",
    otpLabel: "تصدیقی کوڈ",
    otpPlaceholder: "6 ہندسوں کا کوڈ درج کریں",
    otpSent: (e) => `تصدیقی کوڈ ${e} پر بھیج دیا گیا۔`,
    otpButton: "سائن ان کریں",
    backLink: "مختلف ای میل استعمال کریں",
    resendLink: "کوڈ دوبارہ بھیجیں",
    resendOk: "کوڈ دوبارہ بھیج دیا گیا!",
    errorGeneric: "کچھ غلط ہو گیا۔ براہ کرم دوبارہ کوشش کریں۔",
    loginHeading: "شروع کرنے کے لیے سائن ان کریں",
    loginSub: "آپ کے جوابات خفیہ اور محفوظ ہیں۔",
  },
  hi: {
    title: "श्रम बल सर्वेक्षण",
    subtitle: "AI-संचालित रोजगार डेटा संग्रह प्लेटफ़ॉर्म",
    emailLabel: "ईमेल पता",
    emailPlaceholder: "you@example.com",
    emailButton: "सत्यापन कोड भेजें",
    otpLabel: "सत्यापन कोड",
    otpPlaceholder: "6-अंकीय कोड दर्ज करें",
    otpSent: (e) => `सत्यापन कोड ${e} पर भेजा गया।`,
    otpButton: "साइन इन करें",
    backLink: "एक अलग ईमेल का उपयोग करें",
    resendLink: "कोड पुनः भेजें",
    resendOk: "कोड पुनः भेज दिया गया!",
    errorGeneric: "कुछ गलत हो गया। कृपया पुनः प्रयास करें।",
    loginHeading: "शुरू करने के लिए साइन इन करें",
    loginSub: "आपके उत्तर गोपनीय और सुरक्षित हैं।",
  },
  tl: {
    title: "Labour Force Survey",
    subtitle: "AI-Powered na Platform sa Pangongolekta ng Datos ng Trabaho",
    emailLabel: "Email address",
    emailPlaceholder: "you@example.com",
    emailButton: "Ipadala ang verification code",
    otpLabel: "Verification code",
    otpPlaceholder: "Ilagay ang 6-digit na code",
    otpSent: (e) => `Naipadala ang verification code sa ${e}.`,
    otpButton: "Mag-sign in",
    backLink: "Gumamit ng ibang email",
    resendLink: "Ipadala ulit ang code",
    resendOk: "Naipadala ulit ang code!",
    errorGeneric: "May nagkamali. Pakisubukan muli.",
    loginHeading: "Mag-sign in para magsimula",
    loginSub: "Kumpidensyal at protektado ang iyong mga sagot.",
  },
};

const STATS = [
  { value: "436", label: "ISCO-08 Unit Groups" },
  { value: "5",   label: "Languages Supported" },
  { value: "4",   label: "Classification Standards" },
  { value: "4",   label: "Pipeline Stages" },
];

const FEATURES = [
  {
    id: "nlp",
    title: "Multilingual NLP",
    tag: null,
    points: ["English · Arabic · Gulf Arabic", "Urdu · Hindi · Filipino", "Named entity recognition", "Gulf dialect normalisation"],
  },
  {
    id: "rag",
    title: "ISCO-08 Retrieval (RAG)",
    tag: "Best-tested: flat + enrichment",
    points: ["Flat retrieval + official-text enrichment (headline result)", "Hierarchical 4-stage pipeline also implemented, for comparison", "Hierarchical measurably underperforms flat on real benchmark data", "Keyword major-group anchoring"],
  },
  {
    id: "ilo",
    title: "Classification Standards",
    tag: null,
    points: ["ISCO-08 occupation coding", "ISIC Rev.4 industry coding", "ISCED 2011 education level", "UN M49 nationality coding"],
  },
];

// Real, verified counts for all 3 implemented classification standards
// (checked directly against the live Qdrant collections and CLAUDE.md's
// own verified-numbers table, 2026-08-30). Each standard gets its own card
// below — ISIC Rev.4 and ISCED were previously missing from this page
// entirely, giving the false impression only ISCO-08 was implemented.
const STRUCTURE_SECTIONS = [
  {
    id: "isco",
    title: "ISCO-08 Catalogue Structure (4 Levels)",
    stages: [
      { stage: "Stage 1", label: "Major Group", count: "10" },
      { stage: "Stage 2", label: "Sub-major",   count: "43" },
      { stage: "Stage 3", label: "Minor Group", count: "130" },
      { stage: "Stage 4", label: "Unit Group",  count: "436" },
    ],
    caption: "436 unit groups, matching the official ILO count exactly · " +
      "Best-tested retrieval: flat + official-text enrichment · Keyword " +
      "major-group anchoring · LLM re-ranking when confidence < 0.92",
  },
  {
    id: "isic",
    title: "ISIC Rev.4 Catalogue Structure (4 Levels)",
    stages: [
      { stage: "Level 1", label: "Section",  count: "21" },
      { stage: "Level 2", label: "Division", count: "68" },
      { stage: "Level 3", label: "Group",    count: "118" },
      { stage: "Level 4", label: "Class",    count: "134" },
    ],
    caption: "134 of 419 official ISIC Rev.4 classes populated — a real, " +
      "disclosed coverage gap, not yet full · Same best-tested recipe as " +
      "ISCO-08 (flat retrieval + official-text enrichment + e5-large " +
      "embeddings), infrastructure-complete and live · No WISCO-style " +
      "external accuracy benchmark exists for ISIC yet — only a synthetic, " +
      "LLM-generated benchmark (disclosed in full in the thesis)",
  },
  {
    id: "isced",
    title: "ISCED-F 2013 Field Structure (3 Levels)",
    stages: [
      { stage: "Level 1", label: "Broad Field",    count: "11" },
      { stage: "Level 2", label: "Narrow Field",   count: "25" },
      { stage: "Level 3", label: "Detailed Field", count: "63" },
    ],
    caption: "63 of ~80 official ISCED-F 2013 detailed fields populated — " +
      "a real, disclosed coverage gap · ISCED 2011 education-attainment " +
      "level (9 categories, No formal education → PhD) is scored as a " +
      "separate, independent dimension, not part of this field hierarchy · " +
      "Same best-tested recipe as ISCO-08 · No WISCO-style external " +
      "accuracy benchmark exists yet — only a synthetic benchmark " +
      "(disclosed in full in the thesis)",
  },
];

const LANG_PILLS = [
  { code: "en", label: "EN" },
  { code: "ar", label: "AR" },
  { code: "ur", label: "UR" },
  { code: "hi", label: "HI" },
  { code: "tl", label: "TL" },
];

// ── Component ─────────────────────────────────────────────────────────────────
export default function LoginPage() {
  const router = useRouter();
  const [lang, setLang]       = useState("en");
  const [step, setStep]       = useState("input");
  const [email, setEmail]     = useState("");
  const [otp, setOtp]         = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [resendMsg, setResendMsg] = useState("");

  const t   = T[lang] || T.en;
  const dir = (lang === "ar" || lang === "ur") ? "rtl" : "ltr";

  useEffect(() => {
    if (typeof window !== "undefined" && localStorage.getItem("lfs_token")) {
      router.replace("/chat");
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleEmailSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await requestOtp(email.trim());
      if (data.dev_otp) setOtp(data.dev_otp);
      setStep("otp");
    } catch (err) {
      setError(err.message || t.errorGeneric);
    } finally {
      setLoading(false);
    }
  }

  async function handleOtpSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await verifyOtp(email.trim(), otp.trim());
      localStorage.setItem("lfs_token", data.access_token);
      localStorage.setItem("lfs_lang", lang);
      router.push("/chat");
    } catch (err) {
      setError(err.message || t.errorGeneric);
      setOtp("");
      try {
        const fresh = await requestOtp(email.trim());
        if (fresh.dev_otp) setOtp(fresh.dev_otp);
      } catch (_) {}
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    setResendMsg("");
    setError("");
    try {
      const data = await requestOtp(email.trim());
      if (data.dev_otp) setOtp(data.dev_otp);
      setResendMsg(t.resendOk);
      setTimeout(() => setResendMsg(""), 4000);
    } catch (err) {
      setError(err.message || t.errorGeneric);
    }
  }

  return (
    <>
      <Head>
        <title>{`${t.title} — AI Platform`}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div dir={dir} style={{ fontFamily: SANS, background: C.bg, minHeight: "100vh", color: C.text }}>

        {/* ── Header ───────────────────────────────────────────────────────── */}
        <header style={{ background: C.white, borderBottom: `1px solid ${C.border}`, position: "sticky", top: 0, zIndex: 10 }}>
          <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 24px", height: 56, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ background: C.green, fontFamily: MONO, width: 32, height: 32, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 11, fontWeight: 700 }}>
                LFS
              </div>
              <div>
                <span style={{ fontFamily: MONO, color: C.text, fontSize: 13, fontWeight: 600 }}>Labour Force Survey</span>
                <span style={{ color: C.faint, fontSize: 11, marginLeft: 8 }}>AI Platform</span>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <Link href="/questionnaire" style={{ fontSize: 12, color: C.green, textDecoration: "underline" }}>
                Full questionnaire
              </Link>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 11, color: C.muted, fontFamily: MONO }}>LANG:</span>
                {LANG_PILLS.map(lp => (
                  <button key={lp.code} onClick={() => setLang(lp.code)}
                    style={{
                      fontFamily: MONO, fontSize: 10, fontWeight: 700, padding: "3px 10px", borderRadius: 99,
                      background: lang === lp.code ? C.green : "transparent",
                      color:      lang === lp.code ? "#fff"  : C.muted,
                      border:     lang === lp.code ? `1px solid ${C.green}` : `1px solid ${C.border}`,
                      cursor: "pointer", transition: "all 0.15s",
                    }}>
                    {lp.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </header>

        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 24px" }}>

          {/* ── Hero ─────────────────────────────────────────────────────────── */}
          <section style={{ padding: "48px 0 36px", display: "grid", gridTemplateColumns: "1fr auto", gap: 48, alignItems: "center" }}>

            {/* Left: title + meta */}
            <div>
              {/* Thesis badge */}
              <div style={{ display: "inline-flex", alignItems: "center", gap: 6, background: C.greenLt, border: `1px solid ${C.green}40`, color: C.green, fontSize: 10, fontWeight: 600, fontFamily: MONO, padding: "4px 12px", borderRadius: 99, marginBottom: 20 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: C.green }} className="animate-pulse" />
                MSc Thesis — Multilingual Conversational AI Survey System
              </div>

              <h1 style={{ fontSize: 40, fontWeight: 700, color: C.text, lineHeight: 1.2, marginBottom: 12 }}>
                {t.title}
              </h1>
              <p style={{ fontSize: 16, color: C.muted, marginBottom: 24, maxWidth: 500 }}>{t.subtitle}</p>

              {/* Standards pills */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 28 }}>
                {["ISCO-08", "ISIC Rev.4", "ISCED 2011", "UN M49", "ILO ICLS-19"].map(s => (
                  <span key={s} style={{ fontFamily: MONO, fontSize: 10, fontWeight: 600, background: C.white, border: `1px solid ${C.border}`, color: C.muted, padding: "3px 10px", borderRadius: 6 }}>
                    {s}
                  </span>
                ))}
              </div>

              {/* Language flags — text only, no emoji for professional look */}
              <div style={{ display: "flex", gap: 16 }}>
                {[
                  { code: "GB", name: "English" },
                  { code: "AE", name: "Arabic" },
                  { code: "PK", name: "Urdu" },
                  { code: "IN", name: "Hindi" },
                  { code: "PH", name: "Filipino" },
                ].map(l => (
                  <div key={l.code} style={{ textAlign: "center" }}>
                    <p style={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: C.green }}>{l.code}</p>
                    <p style={{ fontSize: 10, color: C.faint }}>{l.name}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: login card */}
            <div style={{ width: 340, flexShrink: 0 }}>
              <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 16, padding: 28, boxShadow: "0 4px 24px rgba(0,0,0,0.06)" }}>
                <p style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>Secure Access</p>
                <h2 style={{ fontSize: 17, fontWeight: 700, color: C.text, marginBottom: 4 }}>{t.loginHeading}</h2>
                <p style={{ fontSize: 12, color: C.muted, marginBottom: 20 }}>{t.loginSub}</p>

                {/* Email step */}
                {step === "input" && (
                  <form onSubmit={handleEmailSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <div>
                      <label htmlFor="email" style={{ display: "block", fontSize: 12, fontWeight: 600, color: C.text, marginBottom: 6 }}>
                        {t.emailLabel}
                      </label>
                      <input
                        id="email" type="email" required autoFocus
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        placeholder={t.emailPlaceholder}
                        dir="ltr"
                        style={{
                          width: "100%", padding: "10px 12px", borderRadius: 8, fontSize: 13,
                          border: `1px solid ${C.border}`, background: C.bg, color: C.text,
                          outline: "none", boxSizing: "border-box", fontFamily: SANS,
                        }}
                        onFocus={e => e.target.style.borderColor = C.green}
                        onBlur={e => e.target.style.borderColor = C.border}
                      />
                    </div>
                    {error && (
                      <p style={{ fontSize: 12, color: "#dc2626", background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 8, padding: "8px 12px" }}>{error}</p>
                    )}
                    <button type="submit" disabled={loading || !email.trim()}
                      style={{
                        width: "100%", padding: "11px", background: loading || !email.trim() ? C.faint : C.green,
                        color: "#fff", border: "none", borderRadius: 8, fontSize: 13, fontWeight: 600,
                        cursor: loading || !email.trim() ? "default" : "pointer", fontFamily: SANS,
                        transition: "background 0.15s",
                      }}>
                      {loading ? "Sending…" : t.emailButton}
                    </button>
                  </form>
                )}

                {/* OTP step */}
                {step === "otp" && (
                  <form onSubmit={handleOtpSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <div style={{ background: C.greenLt, border: `1px solid ${C.green}40`, borderRadius: 8, padding: "10px 12px", fontSize: 12, color: C.green }}>
                      {t.otpSent(email)}
                    </div>
                    {otp && process.env.NODE_ENV !== "production" && (
                      <div style={{ background: C.amberBg, border: `1px solid #fde68a`, borderRadius: 8, padding: "8px 12px", fontSize: 11, color: C.amber, fontFamily: MONO, textAlign: "center" }}>
                        Dev — code: {otp}
                      </div>
                    )}
                    <div>
                      <label htmlFor="otp" style={{ display: "block", fontSize: 12, fontWeight: 600, color: C.text, marginBottom: 6 }}>
                        {t.otpLabel}
                      </label>
                      <input
                        id="otp" type="text" inputMode="numeric" pattern="[0-9]{6}" maxLength={6} required autoFocus
                        value={otp}
                        onChange={e => setOtp(e.target.value.replace(/\D/g, ""))}
                        placeholder={t.otpPlaceholder}
                        dir="ltr"
                        style={{
                          width: "100%", padding: "10px 12px", borderRadius: 8, fontSize: 20,
                          fontFamily: MONO, fontWeight: 700, letterSpacing: "0.3em", textAlign: "center",
                          border: `1px solid ${C.border}`, background: C.bg, color: C.text,
                          outline: "none", boxSizing: "border-box",
                        }}
                        onFocus={e => e.target.style.borderColor = C.green}
                        onBlur={e => e.target.style.borderColor = C.border}
                      />
                    </div>
                    {error && (
                      <p style={{ fontSize: 12, color: "#dc2626", background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 8, padding: "8px 12px" }}>{error}</p>
                    )}
                    {resendMsg && (
                      <p style={{ fontSize: 12, color: C.green, textAlign: "center" }}>{resendMsg}</p>
                    )}
                    <button type="submit" disabled={loading || otp.length < 6}
                      style={{
                        width: "100%", padding: "11px", background: loading || otp.length < 6 ? C.faint : C.green,
                        color: "#fff", border: "none", borderRadius: 8, fontSize: 13, fontWeight: 600,
                        cursor: loading || otp.length < 6 ? "default" : "pointer", fontFamily: SANS,
                      }}>
                      {loading ? "Verifying…" : t.otpButton}
                    </button>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                      <button type="button" onClick={() => { setStep("input"); setError(""); setOtp(""); }}
                        style={{ color: C.muted, background: "none", border: "none", cursor: "pointer", textDecoration: "underline" }}>
                        {t.backLink}
                      </button>
                      <button type="button" onClick={handleResend}
                        style={{ color: C.muted, background: "none", border: "none", cursor: "pointer", textDecoration: "underline" }}>
                        {t.resendLink}
                      </button>
                    </div>
                  </form>
                )}
              </div>
              <p style={{ textAlign: "center", fontSize: 11, color: C.faint, marginTop: 12 }}>
                UAE PDPL compliant · GDPR Art. 15
              </p>
            </div>
          </section>

          {/* ── Stats row ────────────────────────────────────────────────────── */}
          <section style={{ borderTop: `1px solid ${C.border}`, borderBottom: `1px solid ${C.border}`, padding: "24px 0", display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 0 }}>
            {STATS.map((s, i) => (
              <div key={s.label} style={{ textAlign: "center", padding: "8px 16px", borderRight: i < 3 ? `1px solid ${C.border}` : "none" }}>
                <p style={{ fontFamily: MONO, fontSize: 32, fontWeight: 700, color: C.green, lineHeight: 1 }}>{s.value}</p>
                <p style={{ fontSize: 11, color: C.muted, marginTop: 6 }}>{s.label}</p>
              </div>
            ))}
          </section>

          {/* ── Feature cards ────────────────────────────────────────────────── */}
          <section style={{ padding: "36px 0 28px" }}>
            <p style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 20 }}>
              System Architecture
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
              {FEATURES.map(f => (
                <div key={f.id} style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, transition: "border-color 0.15s" }}
                     onMouseEnter={e => e.currentTarget.style.borderColor = `${C.green}60`}
                     onMouseLeave={e => e.currentTarget.style.borderColor = C.border}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <div style={{ background: C.greenLt, border: `1px solid ${C.green}30`, borderRadius: 8, width: 32, height: 32, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: C.green }}>
                        {f.id === "nlp" ? "NLP" : f.id === "rag" ? "RAG" : "STD"}
                      </span>
                    </div>
                    <h3 style={{ fontSize: 14, fontWeight: 700, color: C.text }}>{f.title}</h3>
                    {f.tag && (
                      <span style={{ background: C.greenLt, border: `1px solid ${C.green}40`, color: C.green, fontSize: 9, fontWeight: 700, fontFamily: MONO, padding: "2px 7px", borderRadius: 99 }}>
                        {f.tag}
                      </span>
                    )}
                  </div>
                  <ul style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                    {f.points.map(p => (
                      <li key={p} style={{ display: "flex", gap: 8, fontSize: 12, color: C.muted }}>
                        <span style={{ color: C.green, flexShrink: 0, fontWeight: 700 }}>›</span>
                        {p}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>

          {/* ── Catalogue structure: ISCO-08, ISIC Rev.4, ISCED-F ─────────────── */}
          {STRUCTURE_SECTIONS.map((sec, secIdx) => {
            const lastIdx = sec.stages.length - 1;
            const isLastSection = secIdx === STRUCTURE_SECTIONS.length - 1;
            return (
              <section key={sec.id} style={{ padding: isLastSection ? "0 0 48px" : "0 0 20px" }}>
                <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 12, padding: "24px 28px" }}>
                  <p style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 16 }}>
                    {sec.title}
                  </p>
                  <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
                    {sec.stages.map((st, idx) => (
                      <div key={st.stage} style={{ display: "flex", alignItems: "center", flex: 1 }}>
                        <div style={{ flex: 1, background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10, padding: "14px 12px", textAlign: "center",
                          ...(idx === lastIdx ? { background: C.greenLt, border: `1px solid ${C.green}40` } : {}) }}>
                          <p style={{ fontFamily: MONO, fontSize: 9, color: C.faint, marginBottom: 4 }}>{st.stage}</p>
                          <p style={{ fontFamily: MONO, fontSize: 26, fontWeight: 700, color: idx === lastIdx ? C.green : C.text, lineHeight: 1 }}>{st.count}</p>
                          <p style={{ fontSize: 11, color: idx === lastIdx ? C.green : C.muted, marginTop: 4 }}>{st.label}</p>
                        </div>
                        {idx < lastIdx && (
                          <div style={{ padding: "0 8px", color: C.faint, fontSize: 18, fontWeight: 300 }}>›</div>
                        )}
                      </div>
                    ))}
                  </div>
                  <p style={{ fontSize: 11, color: C.faint, marginTop: 14, textAlign: "center" }}>
                    {sec.caption}
                  </p>
                </div>
              </section>
            );
          })}

        </div>
      </div>
    </>
  );
}
