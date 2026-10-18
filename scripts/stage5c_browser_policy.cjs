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
  "questionnaire-order",
  "validating-screen",
  "correction-control",
  "correction-payload",
  "wage-revision",
  "answer-preserved",
  "immutability",
  "models-present",
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
  return "SELECT question_id || '|' || id::text || '|' || answer || '|' || COALESCE(supersedes_id::text, '') || '|' || " +
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

const WAGE_FIELD = "monthly_wage_range";
const WAGE_ORIGINAL = "10001_20000";
const WAGE_REVISED = "20001_50000";
const WAGE_ORIGINAL_LABEL = "10,001\u201320,000";
const WAGE_REVISED_LABEL = "20,001\u201350,000";
const CORRECTION_CONTROL = "No, I'd like to correct something";
const WAGE_FIELD_LABEL = "Monthly Wage";
const REQUIRED_ANSWERS = {
  employment_status: "employed",
  education_level: "secondary",
  gender: "male",
  nationality: "indian",
  marital_status: "single",
  emirate: "dubai",
  employment_nature: "paid_employee",
  employment_sector: "private",
  job_title: "software engineer",
  job_duties: "writes and tests application software",
  industry: "technology",
  hours_per_week: "40",
  employment_type: "full_time",
  main_skills: "communication",
  platform_work: "no",
  job_satisfaction: "4",
  question_clarity: "5",
  ai_preference: "prefer_ai",
  data_confidence: "very_confident",
};

function choicePayload(message, field, value) {
  const payload = { message: message, preferred_language: "en" };
  if (field) {
    payload.answer_field = field;
    payload.answer_value = value;
  }
  return payload;
}

function textPayload(message) {
  return { message: message, preferred_language: "en" };
}

function employedJourney() {
  const choice = (field, prompt, name, value) => ({
    kind: "choice", field: field, prompt: prompt, name: name,
    payload: choicePayload(name, value == null ? null : field, value),
  });
  const text = (field, prompt, message) => ({
    kind: "text", field: field, prompt: prompt, payload: textPayload(message),
  });
  return [
    choice(EMPLOYMENT_QUESTION, "current employment status", EMPLOYMENT_CONTROL, EMPLOYMENT_ANSWER),
    choice("education_level", "highest level of education", "Secondary", "secondary"),
    choice("gender", "What is your gender", "Male"),
    choice("nationality", "nationality or country of origin", "Indian", "indian"),
    choice("marital_status", "marital status", "Single"),
    choice("emirate", "currently reside", "Dubai"),
    choice("uae_residence_duration", "residing in the UAE", "5\u20139 years"),
    choice("vocational_training", "professional certifications", "No", "no"),
    choice("employment_nature", "employment arrangement", "Paid employee", "paid_employee"),
    choice("employment_sector", "sector does your employer", "Private"),
    text("job_title", "What is your job title?", "software engineer"),
    text("job_duties", "main tasks and duties", "writes and tests application software"),
    text("industry", "industry or sector do you work", "technology"),
    text("actual_hours_worked", "actually work in your main job", "40"),
    text("hours_per_week", "usually work per week", "40"),
    choice("secondary_job", "other job or business", "No", "no"),
    choice("underemployment", "work more hours", "No"),
    choice("employment_type", "Full-time / Part-time", "Full-time"),
    choice("contract_type", "contract permanent or temporary", "Permanent"),
    choice("remote_work", "work from home", "Never"),
    choice(WAGE_FIELD, "monthly salary range", WAGE_ORIGINAL_LABEL, WAGE_ORIGINAL),
    text("salary_allowances", "following allowances", "none"),
    choice("bonuses", "bonuses or incentives", "No"),
    choice("health_insurance", "health insurance", "No"),
    choice("pension_scheme", "pension fund", "No"),
    choice("qualification_match", "qualifications match", "Well matched"),
    text("main_skills", "main work-related skills", "communication"),
    choice("training_participation", "participate in any vocational", "No"),
    text("labour_market_barriers", "main barriers you face", "language"),
    choice("platform_work", "digital platforms", "No", "no"),
    choice("online_business", "online business", "No"),
    choice("job_satisfaction", "satisfied are you", "4"),
    choice("work_safety", "safe and healthy", "Always"),
    text("workplace_issues", "workplace in the past 12 months", "none"),
    choice("work_life_balance", "work-life balance", "Yes"),
    choice("question_clarity", "clarity of the questions", "5 \u2014 Very clear"),
    choice("difficulty_answering", "difficulty answering", "No \u2014 all questions were clear"),
    choice("ai_preference", "AI assistant", "Prefer AI"),
    choice("data_confidence", "kept private", "Very confident"),
  ];
}

function wageCorrectionPayload() {
  return {
    message: WAGE_FIELD_LABEL + " \u2192 " + WAGE_REVISED_LABEL,
    preferred_language: "en",
    correction_field: WAGE_FIELD,
    correction_value: WAGE_REVISED,
  };
}

function parseActiveSnapshot(text) {
  const lines = String(text || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const rows = [];
  const seen = new Set();
  for (const line of lines) {
    const parts = line.split("|");
    if (parts.length < 5) return null;
    const question = parts[0];
    const id = parts[1];
    const state = parts[parts.length - 1];
    const supersedes = parts[parts.length - 2];
    const answer = parts.slice(2, -2).join("|");
    if (!question || seen.has(question) || state !== "active" || !SESSION_RE.test(id)) return null;
    if (supersedes !== "" && !SESSION_RE.test(supersedes)) return null;
    seen.add(question);
    rows.push({ question: question, id: id, answer: answer, supersedes: supersedes, state: state });
  }
  return rows;
}

function employmentStillActive(activeText, retiredCount) {
  const rows = parseActiveSnapshot(activeText);
  const employment = rows && rows.find((row) => row.question === EMPLOYMENT_QUESTION);
  return !!employment && employment.answer === EMPLOYMENT_ANSWER && String(retiredCount) === "0";
}

function canonicalRequiredSnapshot(text) {
  const rows = parseActiveSnapshot(text);
  if (!rows) return false;
  const byQuestion = new Map(rows.map((row) => [row.question, row]));
  for (const [field, value] of Object.entries(REQUIRED_ANSWERS)) {
    const row = byQuestion.get(field);
    if (!row || row.answer !== value) return false;
  }
  const wage = byQuestion.get(WAGE_FIELD);
  return !!wage && wage.answer === WAGE_ORIGINAL;
}

function preservedRequiredAnswers(beforeText, afterText) {
  const before = parseActiveSnapshot(beforeText);
  const after = parseActiveSnapshot(afterText);
  if (!before || !after || before.length === 0 || before.length !== after.length) return false;
  if (!canonicalRequiredSnapshot(beforeText)) return false;
  const afterByQuestion = new Map(after.map((row) => [row.question, row]));
  let wageSeen = false;
  for (const row of before) {
    const next = afterByQuestion.get(row.question);
    if (!next) return false;
    if (row.question === WAGE_FIELD) {
      wageSeen = true;
      if (row.answer !== WAGE_ORIGINAL || next.answer !== WAGE_REVISED) return false;
      if (next.id === row.id || next.supersedes !== row.id) return false;
    } else if (next.id !== row.id || next.answer !== row.answer || next.supersedes !== row.supersedes) {
      return false;
    }
  }
  for (const [field, value] of Object.entries(REQUIRED_ANSWERS)) {
    const afterRow = afterByQuestion.get(field);
    if (!afterRow || afterRow.answer !== value) return false;
  }
  return wageSeen && afterByQuestion.size === before.length;
}

function uniqueActiveWage(text) {
  const rows = parseActiveSnapshot(text);
  if (!rows) return false;
  const wage = rows.filter((row) => row.question === WAGE_FIELD);
  return wage.length === 1 && wage[0].answer === WAGE_REVISED;
}

function activeQuestion(bubbles, prompt) {
  if (!Array.isArray(bubbles) || bubbles.length === 0 || !prompt) return false;
  return String(bubbles[bubbles.length - 1]).includes(prompt);
}

function serverProgress(body, expected) {
  if (!body || typeof body !== "object" || Array.isArray(body) || !expected) return false;
  if (expected === "validating") {
    return body.state === "validating" && (body.next_field == null || body.next_field === "");
  }
  return body.state === "collecting_info" && body.next_field === expected;
}

function wageRevisionQuery(sessionId) {
  return "SELECT id::text || '|' || answer || '|' || COALESCE(supersedes_id::text, '') || '|' || " +
    "CASE WHEN deleted_at IS NULL THEN 'active' ELSE 'retired' END " +
    "FROM survey_responses WHERE session_id = " + assertSessionId(sessionId) +
    " AND question_id = '" + WAGE_FIELD + "' ORDER BY id";
}

function wageOwnerQuery(sessionId, userId) {
  return "SELECT count(*) FROM survey_responses JOIN survey_sessions " +
    "ON survey_sessions.id = survey_responses.session_id " +
    "WHERE survey_responses.session_id = " + assertSessionId(sessionId) +
    " AND survey_sessions.user_id = " + assertSessionId(userId) +
    " AND survey_responses.question_id = '" + WAGE_FIELD + "'";
}

function parseWageRows(text) {
  return String(text || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line) => {
    const parts = line.split("|");
    return { id: parts[0], answer: parts[1], supersedes: parts[2], state: parts[3], width: parts.length };
  });
}

function assessOriginalWage(text) {
  const rows = parseWageRows(text);
  if (rows.length !== 1 || rows[0].width !== 4) return false;
  const row = rows[0];
  return SESSION_RE.test(row.id) && row.answer === WAGE_ORIGINAL && row.supersedes === "" && row.state === "active";
}

function assessWageRevision(beforeText, afterText) {
  const before = parseWageRows(beforeText);
  const after = parseWageRows(afterText);
  if (before.length !== 1 || after.length !== 2) return false;
  if (before[0].width !== 4 || after.some((row) => row.width !== 4)) return false;
  const original = before[0];
  const retired = after[0];
  const active = after[1];
  return SESSION_RE.test(original.id) && SESSION_RE.test(active.id) && active.id !== original.id &&
    retired.id === original.id && retired.answer === WAGE_ORIGINAL && retired.state === "retired" &&
    active.answer === WAGE_REVISED && active.supersedes === original.id && active.state === "active";
}

function immutabilityCommand() {
  return {
    script: "scripts/run_review_tests.py",
    args: [
      "backend/tests/test_survey_routes.py::TestSubmitResponse::test_completed_session_returns_409",
      "-q",
      "--tb=line",
    ],
  };
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
  WAGE_FIELD,
  WAGE_ORIGINAL,
  WAGE_REVISED,
  WAGE_ORIGINAL_LABEL,
  WAGE_REVISED_LABEL,
  CORRECTION_CONTROL,
  WAGE_FIELD_LABEL,
  REQUIRED_ANSWERS,
  employedJourney,
  wageCorrectionPayload,
  parseActiveSnapshot,
  canonicalRequiredSnapshot,
  employmentStillActive,
  preservedRequiredAnswers,
  uniqueActiveWage,
  activeQuestion,
  serverProgress,
  wageRevisionQuery,
  wageOwnerQuery,
  assessOriginalWage,
  assessWageRevision,
  immutabilityCommand,
};
