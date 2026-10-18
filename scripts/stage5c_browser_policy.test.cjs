const test = require("node:test");
const assert = require("node:assert/strict");
const policy = require("./stage5c_browser_policy.cjs");

test("public reports are deny-by-default codes", () => {
  const leaked = new Error(
    "bearer abc.def.ghi postgresql://user:secret@127.0.0.1:5432/lfs otp 123456 stage5c-en-abcdef12@stage5b.example.com ?token=jwt"
  );
  leaked.code = "print-the-message";
  assert.equal(policy.publicDetail(leaked), "unexpected-failure");
  assert.equal(policy.publicDetail(policy.coded("answer-http")), "answer-http");
  assert.equal(policy.publicDetail(policy.coded("answer-http")).includes("secret"), false);
});

test("chat readiness requires a stored numeric session and an enabled textarea", () => {
  assert.equal(policy.sessionReady("15", true), true);
  assert.equal(policy.sessionReady("", true), false);
  assert.equal(policy.sessionReady("15", false), false);
  assert.equal(policy.sessionReady("0", true), false);
});

test("websocket allowlist is only the local Next.js HMR endpoint", () => {
  assert.equal(policy.decideWebSocket(policy.HMR_WEBSOCKET).allow, true);
  assert.equal(policy.decideWebSocket("ws://127.0.0.1:18000/socket").allow, false);
  assert.equal(policy.decideWebSocket("wss://127.0.0.1:13000/_next/webpack-hmr").allow, false);
  assert.equal(policy.decideWebSocket("ws://127.0.0.1:13000/_next/webpack-hmr?token=secret").allow, false);
  assert.equal(policy.decideWebSocket("ws://evil.example/socket").allow, false);
  assert.equal(policy.decideRequest("ws://127.0.0.1:8000/socket", "websocket").allow, false);
});

test("font allowlist rejects broad paths, query tokens, and other-type collection", () => {
  const css = "https://fonts.googleapis.com/css2?family=Inter&display=swap";
  assert.equal(policy.decideRequest(css, "stylesheet").allow, true);
  assert.equal(policy.decideRequest("https://fonts.gstatic.com/s/inter/v1/a.woff2", "font").allow, true);
  assert.equal(policy.decideRequest("https://fonts.googleapis.com/", "other").allow, true);
  assert.equal(policy.decideRequest("https://fonts.gstatic.com/collect?token=secret", "other").allow, false);
  assert.equal(policy.decideRequest("https://fonts.googleapis.com/css2?family=Inter&token=secret", "stylesheet").allow, false);
  assert.equal(policy.decideRequest("https://fonts.googleapis.com/css?family=Inter", "stylesheet").allow, false);
  assert.equal(policy.decideRequest("http://fonts.googleapis.com/css2?family=Inter", "stylesheet").allow, false);
  assert.equal(policy.decideRequest("https://fonts.gstatic.com/s/inter/v1/a.woff2?token=secret", "font").allow, false);
});

test("redirects and port 8000 are not allowed destinations", () => {
  assert.equal(policy.decideRequest("https://evil.example/redirect", "document").allow, false);
  assert.equal(policy.decideRequest("http://127.0.0.1:8000/health", "fetch").allow, false);
  assert.equal(policy.decideRequest("http://localhost:8000/health", "xhr").allow, false);
  assert.equal(policy.contextOptions().serviceWorkers, "block");
});

test("reports omit a distinctive answer marker", () => {
  const marker = "s5c0123456789abcdef";
  const clean = policy.scrubReport(["PASS cross-user-isolation — unauthorized conversation read returned 404 and a separate session was opened"], marker);
  assert.equal(clean[0].includes(marker), false);
  const leaked = policy.scrubReport(["FAIL isolation-body — " + marker], marker);
  assert.equal(leaked[0], "FAIL report — unexpected-failure");
  assert.equal(leaked[0].includes(marker), false);
});

test("request policy blocks before an unauthorized destination is allowed", () => {
  assert.equal(policy.decideRequest("http://127.0.0.1:13000/chat", "document").allow, true);
  assert.equal(policy.decideRequest("ws://127.0.0.1:13000/_next/webpack-hmr", "websocket").allow, true);
  assert.equal(policy.decideRequest("http://127.0.0.1:18000/health", "fetch").allow, true);
  assert.equal(policy.decideRequest("ws://127.0.0.1:18000/socket", "websocket").allow, false);
  assert.equal(policy.decideRequest("https://fonts.googleapis.com/css2?family=Inter", "stylesheet").allow, true);
  assert.equal(policy.decideRequest("https://fonts.gstatic.com/s/inter.woff2", "font").allow, true);
  assert.equal(policy.decideRequest("http://fonts.googleapis.com/css2?family=Inter", "stylesheet").allow, false);
  assert.equal(policy.decideRequest("https://fonts.googleapis.com/collect", "fetch").allow, false);
  assert.equal(policy.decideRequest("https://evil.example/redirect", "document").allow, false);
  assert.equal(policy.decideRequest("http://127.0.0.1:8000/health", "fetch").allow, false);
  const label = policy.violationLabel(
    { code: "destination" },
    "https://evil.example/path?token=secret&email=person@example.com"
  );
  assert.equal(label, "network-blocked destination evil.example");
  assert.equal(label.includes("token"), false);
  assert.equal(label.includes("person@"), false);
});

test("bundle fallback text is not a failure when the effective API origin is the disposable API", () => {
  const corpus = "const BASE = process.env.NEXT_PUBLIC_API_URL || \"http://localhost:8000\"; BASE = \"http://127.0.0.1:18000\";";
  const inlined = policy.assessInlinedOrigin(corpus);
  assert.equal(inlined.ok, true);
  assert.equal(inlined.fallbackTextPresent, true);
  assert.equal(policy.assessApiOrigins(["http://127.0.0.1:18000"]).ok, true);
  assert.equal(policy.assessApiOrigins(["http://localhost:8000"]).ok, false);
  assert.equal(policy.assessInlinedOrigin("no configured origin").ok, false);
});

test("employment quick reply payload matches the visible control", () => {
  const expected = policy.employmentQuickReplyPayload();
  assert.deepEqual(expected, {
    message: "Employed",
    preferred_language: "en",
    answer_field: "employment_status",
    answer_value: "employed",
  });
  assert.equal(policy.sameRequestPayload({ ...expected }, expected), true);
  assert.equal(policy.sameRequestPayload({ ...expected, message: "s5c0123456789abcdef" }, expected), false);
  assert.equal(policy.sameRequestPayload({ ...expected, extra: "employed" }, expected), false);
  assert.equal(policy.sameRequestPayload({ message: "Employed" }, expected), false);
});

test("committed answer query names the question, canonical value, owner and active revision", () => {
  const email = "stage5c-en-abcdef12@stage5b.example.com";
  const active = policy.activeAnswerQuery("4");
  assert.match(active, /question_id/);
  assert.match(active, /id::text/);
  assert.match(active, /deleted_at IS NULL/);
  assert.match(active, /supersedes_id/);
  assert.equal(policy.expectedActiveAnswer(), "employment_status|employed||active");
  assert.match(policy.ownerQuery("4", email), /users.email = 'stage5c-en-abcdef12@stage5b.example.com'/);
  assert.match(policy.auditQuery("4", "9"), /user_id = 9/);
  assert.match(policy.accessQuery("4", "9"), /resource_id = 4/);
  assert.match(policy.retiredAnswerQuery("4"), /employment_status/);
  assert.throws(() => policy.ownerQuery("4", "person@real.example"), /unexpected-failure/);
  assert.throws(() => policy.activeAnswerQuery("0"), /session-id/);
});

test("employed journey follows displayed controls and keeps the employment payload", () => {
  const steps = policy.employedJourney();
  assert.equal(steps[0].payload.message, "Employed");
  assert.deepEqual(steps[0].payload, policy.employmentQuickReplyPayload());
  assert.equal(steps.filter((step) => step.kind === "text").map((step) => step.payload.message).join("|"),
    "software engineer|writes and tests application software|technology|40|40|none|communication|language|none");
  const wage = steps.find((step) => step.payload.answer_field === "monthly_wage_range");
  assert.equal(wage.payload.answer_value, "10001_20000");
  assert.equal(wage.name, "10,001\u201320,000");
  const prompts = steps.map((step) => step.prompt);
  const fields = steps.map((step) => step.field);
  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(fields).size, fields.length);
  assert.equal(fields[0], "employment_status");
  assert.equal(fields[fields.length - 1], "data_confidence");
  assert.equal(policy.serverProgress(
    { state: "collecting_info", next_field: "education_level" }, "education_level"), true);
  assert.equal(policy.serverProgress(
    { state: "clarifying", next_field: "employment_status" }, "education_level"), false);
  assert.equal(policy.serverProgress(
    { state: "collecting_info", next_field: "gender" }, "education_level"), false);
  assert.equal(policy.serverProgress({ state: "validating", next_field: null }, "validating"), true);
  assert.equal(policy.serverProgress(
    { state: "collecting_info", next_field: "survey_comments" }, "validating"), false);
  assert.equal(policy.activeQuestion(
    ["current employment status", "highest level of education"], "current employment status"), false);
  assert.equal(policy.activeQuestion(
    ["current employment status", "highest level of education"], "highest level of education"), true);
  for (const step of steps) {
    assert.equal(policy.sameRequestPayload({ ...step.payload }, step.payload), true);
    assert.equal(Object.hasOwn(step.payload, "correction_field"), false);
  }
});

test("structured wage correction payload is exact and does not rewrite the request", () => {
  const expected = policy.wageCorrectionPayload();
  assert.deepEqual(expected, {
    message: "Monthly Wage \u2192 20,001\u201350,000",
    preferred_language: "en",
    correction_field: "monthly_wage_range",
    correction_value: "20001_50000",
  });
  assert.equal(policy.sameRequestPayload({ ...expected }, expected), true);
  assert.equal(policy.sameRequestPayload({ ...expected, correction_value: "10001_20000" }, expected), false);
  assert.equal(policy.sameRequestPayload({ message: expected.message }, expected), false);
});

test("wage revision keeps the retired row and one active successor", () => {
  const before = "15|10001_20000||active";
  const after = "15|10001_20000||retired\n16|20001_50000|15|active";
  assert.equal(policy.assessOriginalWage(before), true);
  assert.equal(policy.assessWageRevision(before, after), true);
  assert.equal(policy.assessWageRevision(before, "16|20001_50000|15|active"), false);
  assert.equal(policy.assessWageRevision(before, "15|10001_20000||active\n16|20001_50000|15|active"), false);
  assert.equal(policy.assessWageRevision(before, "15|10001_20000||retired\n16|10001_20000|15|active"), false);
  assert.equal(policy.assessWageRevision(before, "15|10001_20000||retired\n16|20001_50000||active"), false);
  const query = policy.wageRevisionQuery("4");
  assert.match(query, /deleted_at/);
  assert.match(query, /supersedes_id/);
  assert.match(query, /monthly_wage_range/);
  assert.doesNotMatch(query, /insert |update |delete /i);
  assert.match(policy.wageOwnerQuery("4", "9"), /user_id = 9/);
});

test("required answers stay unchanged except the corrected wage", () => {
  const rows = Object.entries(policy.REQUIRED_ANSWERS)
    .filter(([field]) => field !== "monthly_wage_range")
    .map(([field, value], index) => field + "|" + (index + 1) + "|" + value + "||active");
  rows.push("salary_allowances|30|none||active");
  const before = ["monthly_wage_range|21|10001_20000||active", ...rows].join("\n");
  const after = before.replace(
    "monthly_wage_range|21|10001_20000||active",
    "monthly_wage_range|40|20001_50000|21|active");
  const title = rows.find((row) => row.startsWith("job_title|"));
  assert.equal(policy.preservedRequiredAnswers(before, after), true);
  assert.equal(policy.uniqueActiveWage(after), true);
  assert.equal(policy.employmentStillActive(after, "0"), true);
  assert.equal(policy.employmentStillActive(after, "1"), false);
  const changed = after.replace(title, title.replace("|software engineer|", "|other|"));
  assert.equal(policy.preservedRequiredAnswers(before, changed), false);
  const rewritten = after.replace(title, title.replace(/^job_title\|\d+\|/, "job_title|99|"));
  assert.equal(policy.preservedRequiredAnswers(before, rewritten), false);
  const duplicate = after + "\n" + title.replace(/^job_title\|\d+\|/, "job_title|98|");
  assert.equal(policy.parseActiveSnapshot(duplicate), null);
  assert.equal(policy.preservedRequiredAnswers(before, duplicate), false);
  const sameWageId = after.replace(
    "monthly_wage_range|40|20001_50000|21|active", "monthly_wage_range|21|20001_50000||active");
  assert.equal(policy.preservedRequiredAnswers(before, sameWageId), false);
  const missingRetirementLink = after.replace("|21|active", "||active");
  assert.equal(policy.preservedRequiredAnswers(before, missingRetirementLink), false);
});

function canonicalSnapshot(overrides = {}, wageLine = "monthly_wage_range|21|10001_20000||active") {
  const rows = Object.entries(policy.REQUIRED_ANSWERS).map(([field, value], index) => {
    const answer = Object.hasOwn(overrides, field) ? overrides[field] : value;
    return field + "|" + (index + 1) + "|" + answer + "||active";
  });
  rows.push("salary_allowances|30|none||active");
  rows.push(wageLine);
  return rows.join("\n");
}

test("canonical required answers reject noncanonical and incomplete snapshots", () => {
  const before = canonicalSnapshot();
  const after = canonicalSnapshot({}, "monthly_wage_range|40|20001_50000|21|active");
  assert.equal(policy.canonicalRequiredSnapshot(before), true);
  assert.equal(policy.preservedRequiredAnswers(before, after), true);
  const privateSector = canonicalSnapshot({ employment_sector: "Private" });
  assert.equal(policy.canonicalRequiredSnapshot(privateSector), false);
  assert.equal(policy.preservedRequiredAnswers(privateSector, privateSector.replace(
    "monthly_wage_range|21|10001_20000||active",
    "monthly_wage_range|40|20001_50000|21|active")), false);
  const preferAi = canonicalSnapshot({ ai_preference: "Prefer AI" });
  assert.equal(policy.canonicalRequiredSnapshot(preferAi), false);
  assert.equal(policy.preservedRequiredAnswers(preferAi, preferAi.replace(
    "monthly_wage_range|21|10001_20000||active",
    "monthly_wage_range|40|20001_50000|21|active")), false);
  const missing = before.split("\n").filter((line) => !line.startsWith("data_confidence|")).join("\n");
  assert.equal(policy.canonicalRequiredSnapshot(missing), false);
  assert.equal(policy.preservedRequiredAnswers(missing, missing.replace(
    "monthly_wage_range|21|10001_20000||active",
    "monthly_wage_range|40|20001_50000|21|active")), false);
  const title = before.split("\n").find((line) => line.startsWith("job_title|"));
  const rewritten = after.replace(title, title.replace(/^job_title\|\d+\|/, "job_title|99|"));
  assert.equal(policy.preservedRequiredAnswers(before, rewritten), false);
  const duplicate = before + "\nemployment_status|77|employed||active";
  assert.equal(policy.parseActiveSnapshot(duplicate), null);
  assert.equal(policy.canonicalRequiredSnapshot(duplicate), false);
  assert.equal(policy.preservedRequiredAnswers(duplicate, after), false);
});

test("rewritten unrelated revision fails when only its response id changes", () => {
  const before = [
    "employment_status|1|employed||active",
    "job_title|11|software engineer||active",
    "monthly_wage_range|21|10001_20000||active",
  ].join("\n");
  const sameTextNewId = [
    "employment_status|1|employed||active",
    "job_title|99|software engineer||active",
    "monthly_wage_range|40|20001_50000|21|active",
  ].join("\n");
  assert.equal(policy.preservedRequiredAnswers(before, sameTextNewId), false);
  assert.equal(policy.parseActiveSnapshot(before + "\njob_title|12|software engineer||active"), null);
});

test("completed-answer regression command is the existing isolated test", () => {
  const command = policy.immutabilityCommand();
  assert.equal(command.script, "scripts/run_review_tests.py");
  assert.equal(command.args[0], "backend/tests/test_survey_routes.py::TestSubmitResponse::test_completed_session_returns_409");
  assert.equal(JSON.stringify(command).includes("PGPASSWORD"), false);
  assert.equal(JSON.stringify(command).includes("postgresql"), false);
});
