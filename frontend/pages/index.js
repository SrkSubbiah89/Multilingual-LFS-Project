/**
 * Login page — email + OTP two-step authentication.
 *
 * Step 1: user enters email → POST /auth/request-otp
 * Step 2: user enters 6-digit OTP → POST /auth/verify-otp → JWT stored in
 *         localStorage → redirect to /chat
 *
 * Supports English (LTR) and Arabic (RTL) with a toggle.
 */

import { useState, useEffect } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { requestOtp, verifyOtp } from "../components/api";
import LanguageToggle from "../components/LanguageToggle";

// ── i18n strings ──────────────────────────────────────────────────────────────

const T = {
  en: {
    title: "Labour Force Survey",
    subtitle: "AI-Powered Employment Data Collection",
    emailLabel: "Email address",
    emailPlaceholder: "you@example.com",
    emailButton: "Send verification code",
    otpLabel: "Verification code",
    otpPlaceholder: "Enter 6-digit code",
    otpSent: (email) => `A verification code was sent to ${email}.`,
    otpButton: "Sign in",
    backLink: "Use a different email",
    resendLink: "Resend code",
    resendOk: "Code resent!",
    errorGeneric: "Something went wrong. Please try again.",
  },
  ar: {
    title: "مسح القوى العاملة",
    subtitle: "جمع بيانات التوظيف بالذكاء الاصطناعي",
    emailLabel: "البريد الإلكتروني",
    emailPlaceholder: "example@email.com",
    emailButton: "إرسال رمز التحقق",
    otpLabel: "رمز التحقق",
    otpPlaceholder: "أدخل الرمز المكوّن من 6 أرقام",
    otpSent: (email) => `تم إرسال رمز التحقق إلى ${email}.`,
    otpButton: "تسجيل الدخول",
    backLink: "استخدام بريد إلكتروني مختلف",
    resendLink: "إعادة إرسال الرمز",
    resendOk: "تمت إعادة الإرسال!",
    errorGeneric: "حدث خطأ ما. يرجى المحاولة مرة أخرى.",
  },
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function LoginPage() {
  const router = useRouter();

  const [lang, setLang] = useState("en");
  const [step, setStep] = useState("email"); // "email" | "otp"
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [resendMsg, setResendMsg] = useState("");

  const t = T[lang];
  const dir = lang === "ar" ? "rtl" : "ltr";

  // Redirect if already logged in
  useEffect(() => {
    if (typeof window !== "undefined" && localStorage.getItem("lfs_token")) {
      router.replace("/chat");
    }
  }, [router]);

  // ── Handlers ────────────────────────────────────────────────────────────────

  async function handleEmailSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await requestOtp(email.trim());
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
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    setResendMsg("");
    setError("");
    try {
      await requestOtp(email.trim());
      setResendMsg(t.resendOk);
      setTimeout(() => setResendMsg(""), 4000);
    } catch (err) {
      setError(err.message || t.errorGeneric);
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────────

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
        className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex flex-col"
      >
        {/* Top bar */}
        <header className="flex justify-end p-4">
          <LanguageToggle lang={lang} onToggle={setLang} />
        </header>

        {/* Centered card */}
        <main className="flex-1 flex items-center justify-center px-4">
          <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-8 space-y-6">

            {/* Logo / title */}
            <div className="text-center space-y-1">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-blue-600 text-white text-2xl mb-2">
                📋
              </div>
              <h1 className="text-2xl font-bold text-gray-900">{t.title}</h1>
              <p className="text-sm text-gray-500">{t.subtitle}</p>
            </div>

            {/* Email step */}
            {step === "email" && (
              <form onSubmit={handleEmailSubmit} className="space-y-4">
                <div>
                  <label
                    htmlFor="email"
                    className="block text-sm font-medium text-gray-700 mb-1"
                  >
                    {t.emailLabel}
                  </label>
                  <input
                    id="email"
                    type="email"
                    required
                    autoFocus
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder={t.emailPlaceholder}
                    className="w-full px-4 py-2.5 border border-gray-300 rounded-lg text-sm
                               focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                               placeholder:text-gray-400"
                    dir="ltr"
                  />
                </div>

                {error && (
                  <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">
                    {error}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading || !email.trim()}
                  className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 disabled:opacity-50
                             text-white font-medium rounded-lg text-sm transition-colors"
                >
                  {loading ? <Spinner /> : t.emailButton}
                </button>
              </form>
            )}

            {/* OTP step */}
            {step === "otp" && (
              <form onSubmit={handleOtpSubmit} className="space-y-4">
                <p className="text-sm text-gray-600 bg-blue-50 px-3 py-2 rounded-lg">
                  {t.otpSent(email)}
                </p>

                <div>
                  <label
                    htmlFor="otp"
                    className="block text-sm font-medium text-gray-700 mb-1"
                  >
                    {t.otpLabel}
                  </label>
                  <input
                    id="otp"
                    type="text"
                    inputMode="numeric"
                    pattern="[0-9]{6}"
                    maxLength={6}
                    required
                    autoFocus
                    value={otp}
                    onChange={(e) => setOtp(e.target.value.replace(/\D/g, ""))}
                    placeholder={t.otpPlaceholder}
                    className="w-full px-4 py-2.5 border border-gray-300 rounded-lg text-sm
                               tracking-widest text-center focus:outline-none focus:ring-2
                               focus:ring-blue-500 focus:border-transparent placeholder:tracking-normal
                               placeholder:text-gray-400"
                    dir="ltr"
                  />
                </div>

                {error && (
                  <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">
                    {error}
                  </p>
                )}

                {resendMsg && (
                  <p className="text-sm text-green-600 bg-green-50 px-3 py-2 rounded-lg">
                    {resendMsg}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading || otp.length < 6}
                  className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 disabled:opacity-50
                             text-white font-medium rounded-lg text-sm transition-colors"
                >
                  {loading ? <Spinner /> : t.otpButton}
                </button>

                <div className="flex justify-between text-sm">
                  <button
                    type="button"
                    onClick={() => { setStep("email"); setOtp(""); setError(""); }}
                    className="text-gray-500 hover:text-gray-800 underline"
                  >
                    {t.backLink}
                  </button>
                  <button
                    type="button"
                    onClick={handleResend}
                    className="text-blue-600 hover:text-blue-800 underline"
                  >
                    {t.resendLink}
                  </button>
                </div>
              </form>
            )}

          </div>
        </main>
      </div>
    </>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Spinner() {
  return (
    <span className="inline-flex items-center justify-center gap-2">
      <svg
        className="animate-spin h-4 w-4 text-white"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8v8H4z"
        />
      </svg>
      Loading…
    </span>
  );
}
