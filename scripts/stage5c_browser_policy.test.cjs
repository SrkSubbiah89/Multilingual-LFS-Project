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
