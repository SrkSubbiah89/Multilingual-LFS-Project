/**
 * Thin API client for the FastAPI backend.
 * Base URL is read from NEXT_PUBLIC_API_URL (defaults to http://localhost:8000).
 */

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options;
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...extraHeaders },
    ...rest,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.detail || `HTTP ${res.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

// ── Auth ─────────────────────────────────────────────────────────────────────

export function requestOtp(email) {
  return request("/auth/request-otp", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function verifyOtp(email, code) {
  return request("/auth/verify-otp", {
    method: "POST",
    body: JSON.stringify({ email, code }),
  });
}

export function requestSmsOtp(phone) {
  return request("/auth/request-sms-otp", {
    method: "POST",
    body: JSON.stringify({ phone }),
  });
}

export function verifySmsOtp(phone, code) {
  return request("/auth/verify-sms-otp", {
    method: "POST",
    body: JSON.stringify({ phone, code }),
  });
}

// ── Survey sessions ───────────────────────────────────────────────────────────

export function createSession(token, language) {
  return request("/survey/sessions", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({ language }),
  });
}

// `correction`, when passed, is { field, value } from the VALIDATING-state
// structured field picker (see chat.js's handleStructuredCorrection) — sent
// alongside `message` so the backend applies it deterministically instead of
// re-parsing free text (see MessageBody.correction_field/correction_value).
export function sendMessage(token, sessionId, message, preferredLanguage = null, correction = null) {
  return request(`/survey/sessions/${sessionId}/message`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      message,
      ...(preferredLanguage && { preferred_language: preferredLanguage }),
      ...(correction?.field && correction?.value != null && {
        correction_field: correction.field,
        correction_value: correction.value,
      }),
    }),
  });
}

export function getReport(token, sessionId, regenerate = false) {
  const qs = regenerate ? "?regenerate=true" : "";
  return request(`/survey/sessions/${sessionId}/report${qs}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

// ── HITL supervisor review ────────────────────────────────────────────────────

export function getHitlQueue(token, statusFilter = "pending") {
  return request(`/survey/hitl/queue?status_filter=${statusFilter}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export function submitHitlReview(token, escalationId, action, code = null, notes = null) {
  return request("/survey/hitl/review", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({ escalation_id: escalationId, action, code, notes }),
  });
}
