/**
 * Pure decisions for the Stage 5C browser job.
 * This module does not launch a browser, open a database, or read secrets.
 */
const API_ORIGIN = "http://127.0.0.1:18000";
const WEB_ORIGIN = "http://127.0.0.1:13000";
const EMPLOYMENT_QUESTION = "employment_status";
const EMPLOYMENT_ANSWER = "employed";
const EMPLOYMENT_CONTROL = "Employed";
const EMAIL_RE = /^stage5c-(en|intruder|ar)-[0-9a-f]{8}@stage5b\.example\.com$/;
const SESSION_RE = /^[1-9][0-9]{0,17}$/;
const BANNED_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"];
const FONT_STYLESHEET_HOST = "fonts.googleapis.com";
const FONT_FILE_HOST = "fonts.gstatic.com";
const FONT_QUERY_KEYS = new Set(["family", "display"]);
const HMR_WEBSOCKET = "ws://127.0.0.1:13000/_next/webpack-hmr";
const PUBLIC_CODES = new Set([
  "unexpected-failure",
  "otp-not-filled",
  "chat-timeout",
  "session-id",
  "employment-control-missing",
  "answer-http",
  "answer-payload",
  "answer-missing",
  "isolation-status",
  "isolation-body",
  "isolation-marker",
  "isolation-policy",
  "isolation-session",
  "bundle-script",
  "bundle-origin",
  "banned-api-request",
  "sql-failed",
  "protected-query",
  "network-blocked",
  "login-direction",
]);

function coded(code) {
  const error = new Error(PUBLIC_CODES.has(code) ? code : "unexpected-failure");
  error.code = PUBLIC_CODES.has(code) ? code : "unexpected-failure";
  return error;
}

function publicDetail(error) {
  const code = error && PUBLIC_CODES.has(error.code) ? error.code : "unexpected-failure";
  return code;
}

function sessionReady(sessionId, textareaEnabled) {
  return SESSION_RE.test(String(sessionId || "")) && textareaEnabled === true;
}

function assertSessionId(value) {
  const text = String(value || "");
  if (!SESSION_RE.test(text)) throw coded("session-id");
  return text;
}

function emailLiteral(value) {
  if (!EMAIL_RE.test(String(value || ""))) throw coded("unexpected-failure");
  return "'" + value + "'";
}

function decideWebSocket(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch (_error) {
    return { allow: false, code: "websocket" };
  }
  if (url.protocol !== "ws:" || url.username || url.password || url.search) {
    return { allow: false, code: "websocket" };
  }
  if (url.hostname !== "127.0.0.1" || url.port !== "13000" || url.pathname !== "/_next/webpack-hmr") {
    return { allow: false, code: "websocket" };
  }
  return { allow: true, code: "frontend-websocket" };
}

function decideFont(url, type) {
  if (url.protocol !== "https:" || url.username || url.password) return { allow: false, code: "font-type" };
  if (url.hostname === FONT_STYLESHEET_HOST && type === "stylesheet" && url.pathname === "/css2") {
    const keys = [...url.searchParams.keys()];
    if (keys.length > 0 && keys.every((key) => FONT_QUERY_KEYS.has(key))) return { allow: true, code: "font" };
    return { allow: false, code: "font-type" };
  }
  if (url.hostname === FONT_FILE_HOST && type === "font" && url.search === "" &&
      url.pathname.startsWith("/s/") && /\.(woff2|woff|ttf)$/i.test(url.pathname)) {
    return { allow: true, code: "font" };
  }
  if ((url.hostname === FONT_STYLESHEET_HOST || url.hostname === FONT_FILE_HOST) &&
      type === "other" && url.pathname === "/" && url.search === "") {
    return { allow: true, code: "font-preconnect" };
  }
  return { allow: false, code: "font-type" };
}

function decideRequest(rawUrl, resourceType) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch (_error) {
    return { allow: false, code: "unparsed-url" };
  }
  const type = String(resourceType || "");
  if (url.protocol === "data:") return { allow: true, code: "data" };
  if (type === "websocket" || url.protocol === "ws:" || url.protocol === "wss:") {
    return decideWebSocket(rawUrl);
  }
  if (url.protocol === "http:" && url.hostname === "127.0.0.1" && url.port === "13000") {
    return { allow: true, code: "frontend" };
  }
  if (url.protocol === "http:" && url.hostname === "127.0.0.1" && url.port === "18000") {
    if (type === "fetch" || type === "xhr") return { allow: true, code: "api" };
    return { allow: false, code: "api-resource" };
  }
  if (url.hostname === FONT_STYLESHEET_HOST || url.hostname === FONT_FILE_HOST) return decideFont(url, type);
  return { allow: false, code: "destination" };
}

function contextOptions() {
  return { serviceWorkers: "block", viewport: { width: 1280, height: 720 } };
}

function violationLabel(decision, rawUrl) {
  let host = "unknown";
  try {
    const url = new URL(rawUrl);
    if (/^[a-z0-9.-]+$/i.test(url.hostname)) host = url.hostname;
  } catch (_error) {
    host = "unknown";
  }
  return "network-blocked " + (decision && decision.code ? decision.code : "destination") + " " + host;
}

function assessInlinedOrigin(corpus) {
  const text = String(corpus || "");
  if (!text.includes(API_ORIGIN)) return { ok: false, code: "bundle-origin", fallbackTextPresent: false };
  return {
    ok: true,
    code: "configured-origin",
    fallbackTextPresent: BANNED_ORIGINS.some((origin) => text.includes(origin)),
  };
}

function assessApiOrigins(origins) {
  const values = Array.isArray(origins) ? origins : [];
  if (values.length === 0) return { ok: false, code: "bundle-origin" };
  for (const origin of values) {
    if (origin !== API_ORIGIN || BANNED_ORIGINS.includes(origin)) {
      return { ok: false, code: "banned-api-request" };
    }
  }
  return { ok: true, code: "configured-origin" };
}

function ownerQuery(sessionId, email) {
  return "SELECT users.id::text FROM users JOIN survey_sessions ON survey_sessions.user_id = users.id " +
    "WHERE survey_sessions.id = " + assertSessionId(sessionId) + " AND users.email = " + emailLiteral(email);
}

function activeAnswerQuery(sessionId) {
  return "SELECT question_id || '|' || answer || '|' || COALESCE(supersedes_id::text, '') || '|' || " +
    "CASE WHEN deleted_at IS NULL THEN 'active' ELSE 'retired' END " +
    "FROM survey_responses WHERE session_id = " + assertSessionId(sessionId) +
    " AND deleted_at IS NULL ORDER BY id";
}

function retiredAnswerQuery(sessionId) {
  return "SELECT count(*) FROM survey_responses WHERE session_id = " + assertSessionId(sessionId) +
    " AND question_id = '" + EMPLOYMENT_QUESTION + "' AND deleted_at IS NOT NULL";
}

function auditQuery(sessionId, userId) {
  return "SELECT count(*) FROM audit_logs WHERE event_type = 'session_started' AND session_id = " +
    assertSessionId(sessionId) + " AND user_id = " + assertSessionId(userId);
}

function accessQuery(sessionId, userId) {
  return "SELECT count(*) FROM data_access_logs WHERE access_type = 'write' AND resource_type = 'survey_session' " +
    "AND resource_id = " + assertSessionId(sessionId) + " AND user_id = " + assertSessionId(userId);
}

function employmentQuickReplyPayload() {
  return {
    message: EMPLOYMENT_CONTROL,
    preferred_language: "en",
    answer_field: EMPLOYMENT_QUESTION,
    answer_value: EMPLOYMENT_ANSWER,
  };
}

function sameRequestPayload(actual, expected) {
  if (!actual || typeof actual !== "object" || Array.isArray(actual)) return false;
  if (!expected || typeof expected !== "object" || Array.isArray(expected)) return false;
  const actualKeys = Object.keys(actual);
  const expectedKeys = Object.keys(expected);
  if (actualKeys.length !== expectedKeys.length) return false;
  return expectedKeys.every((key) => actualKeys.includes(key) && actual[key] === expected[key]);
}

function scrubReport(lines, marker) {
  const secrets = (Array.isArray(marker) ? marker : [marker]).map((item) => String(item || "")).filter(Boolean);
  if (secrets.length === 0) return lines.slice();
  return lines.map((line) => (
    secrets.some((secret) => String(line).includes(secret)) ? "FAIL report — unexpected-failure" : line
  ));
}

function expectedActiveAnswer() {
  return EMPLOYMENT_QUESTION + "|" + EMPLOYMENT_ANSWER + "||active";
}

module.exports = {
  API_ORIGIN,
  WEB_ORIGIN,
  EMPLOYMENT_QUESTION,
  EMPLOYMENT_ANSWER,
  EMPLOYMENT_CONTROL,
  BANNED_ORIGINS,
  HMR_WEBSOCKET,
  coded,
  publicDetail,
  decideWebSocket,
  contextOptions,
  sessionReady,
  assertSessionId,
  emailLiteral,
  decideRequest,
  violationLabel,
  assessInlinedOrigin,
  assessApiOrigins,
  ownerQuery,
  activeAnswerQuery,
  retiredAnswerQuery,
  auditQuery,
  accessQuery,
  scrubReport,
  expectedActiveAnswer,
  employmentQuickReplyPayload,
  sameRequestPayload,
};
