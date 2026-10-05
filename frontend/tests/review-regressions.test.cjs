const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test, afterEach } = require("node:test");

// The client has no imports; load its actual ES-module source without changing
// the CommonJS package settings required by next.config.js.
const loadModule = (relative) => import(`data:text/javascript;base64,${
  fs.readFileSync(path.join(__dirname, "..", relative)).toString("base64")
}`);
const clientPromise = loadModule("components/api.js");
const reviewPromise = loadModule("components/review-status.js");
const optionsPromise = loadModule("components/survey-options.js");
const originalFetch = global.fetch;
afterEach(() => { global.fetch = originalFetch; });

function storageFor(id) {
  const values = new Map(id == null ? [] : [["lfs_active_session", String(id)]]);
  return {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  };
}

function mockResponses(...responses) {
  const calls = [];
  global.fetch = async (url, options) => {
    calls.push({ url, ...options });
    assert.ok(responses.length, "Unexpected API request");
    const response = responses.shift();
    if (response instanceof Error) throw response;
    const [status, body] = response;
    return { ok: status >= 200 && status < 300, status, json: async () => body };
  };
  return calls;
}

test("reload resumes the same interview and preserves its transcript without an answer request", async () => {
  const client = await clientPromise;
  const conversation = {
    state: "collecting_info", next_field: "field_of_study",
    collected_data: { employment_status: "employed", education_level: "bachelor" },
    history: [{ role: "user", content: "Bachelor's degree" }, { role: "assistant", content: "What did you study?" }],
  };
  const calls = mockResponses([200, conversation]);
  const storage = storageFor(42);
  const restored = await client.openSurveySession("test-token", "en", storage);
  assert.equal(restored.id, 42);
  assert.deepEqual(restored.conversation, conversation);
  assert.equal(calls.length, 1);
  assert.ok(calls[0].url.endsWith("/survey/sessions/42/conversation"));
  assert.equal(calls[0].body, undefined);
  assert.equal(storage.getItem(client.ACTIVE_SESSION_KEY), "42");
});

test("a missing or inaccessible saved session starts a new owned session", async () => {
  const client = await clientPromise;
  const calls = mockResponses([404, { detail: "Session not found." }], [200, { id: 43 }], [200, { reply: "First question" }]);
  const storage = storageFor(42);
  const opened = await client.openSurveySession("test-token", "hi", storage);
  assert.equal(opened.id, 43);
  assert.equal(storage.getItem(client.ACTIVE_SESSION_KEY), "43");
  assert.deepEqual(JSON.parse(calls[1].body), { language: "hi" });
  assert.equal(calls[1].method, "POST");
  assert.ok(calls[2].url.endsWith("/43/conversation"));
});

test("API failures preserve the active session and do not create a replacement", async () => {
  const client = await clientPromise;
  for (const status of [401, 403, 500, 503]) {
    const calls = mockResponses([status, { detail: "Request failed" }]);
    const storage = storageFor(42);
    await assert.rejects(client.openSurveySession("test-token", "en", storage), error => error.status === status);
    assert.equal(storage.getItem(client.ACTIVE_SESSION_KEY), "42");
    assert.equal(calls.length, 1);
  }
});

test("interrupted initial prompt loading retains the newly created session for reload", async () => {
  const client = await clientPromise;
  const storage = storageFor();
  const calls = mockResponses([200, { id: 44 }], new Error("Network interrupted"));
  await assert.rejects(client.openSurveySession("test-token", "en", storage), /Network interrupted/);
  assert.equal(storage.getItem(client.ACTIVE_SESSION_KEY), "44");
  assert.equal(calls.length, 2);
});

test("invalid saved identifiers cannot be interpolated into API paths", async () => {
  const client = await clientPromise;
  const calls = mockResponses([200, { id: 45 }], [200, { reply: "First question" }]);
  const opened = await client.openSurveySession("test-token", "en", storageFor("42/message"));
  assert.equal(opened.id, 45);
  assert.equal(calls.length, 2);
  assert.ok(calls[0].url.endsWith("/survey/sessions"));
});

test("language updates use PATCH with a language only and never submit a fabricated answer", async () => {
  const client = await clientPromise;
  const calls = mockResponses([200, { reply: "Hindi prompt", collected_data: {} }]);
  const response = await client.updateSessionLanguage("test-token", 42, "hi");
  assert.equal(response.reply, "Hindi prompt");
  assert.equal(calls[0].method, "PATCH");
  assert.ok(calls[0].url.endsWith("/survey/sessions/42/language"));
  assert.deepEqual(JSON.parse(calls[0].body), { language: "hi" });
  assert.equal(calls[0].headers.Authorization, "Bearer test-token");
});

test("every translated skip-gate button retains its label and has a canonical answer", async () => {
  const { QUICK_OPTIONS, QUICK_VALUES, getQuickOptions } = await optionsPromise;
  for (const [field, values] of Object.entries(QUICK_VALUES)) {
    for (const language of ["en", "ar", "ur", "hi", "tl"]) {
      const options = getQuickOptions(field, language);
      assert.equal(options.length, values.length, `${field}/${language}`);
      assert.deepEqual(options.map(option => option.label), QUICK_OPTIONS[field][language]);
      assert.deepEqual(options.map(option => option.value), values);
    }
  }
  assert.equal(getQuickOptions("education_level", "hi")[4].value, "bachelor");
  assert.equal(getQuickOptions("employment_nature", "ur")[0].value, "paid_employee");
  assert.equal(getQuickOptions("ever_worked", "tl")[2].value, "never_worked");
  assert.equal(getQuickOptions("platform_work", "ar")[0].value, "yes_primary");
});

test("quick answers send their validated field and code alongside the respondent's display label", async () => {
  const client = await clientPromise;
  const { getQuickOptions } = await optionsPromise;
  const option = getQuickOptions("employment_nature", "hi")[0];
  const calls = mockResponses([200, { reply: "Next question" }]);
  await client.sendMessage("test-token", 42, option.label, "hi", null,
    { field: "employment_nature", value: option.value });
  assert.deepEqual(JSON.parse(calls[0].body), {
    message: option.label, preferred_language: "hi",
    answer_field: "employment_nature", answer_value: "paid_employee",
  });
});

test("the actual chat quick-reply handler forwards canonical answers and retains the displayed response", async () => {
  const client = await clientPromise;
  const { getQuickOptions } = await optionsPromise;
  const source = fs.readFileSync(path.join(__dirname, "..", "pages/chat.js"), "utf8");
  const handler = source.match(/const handleQuickReply = useCallback\(([\s\S]*?)\n  }, \[[^\]]*\]\);/);
  assert.ok(handler, "Chat quick-reply handler must exist");
  const displayed = [];
  const calls = mockResponses([200, { reply: "Next question" }]);
  const callback = vm.runInNewContext(`(${handler[1]}\n})`, {
    sending: false, completed: false, sessionId: 42, token: "test-token", lang: "ur",
    sendMessage: client.sendMessage,
    setMessages: update => displayed.push(...update([])),
    setSending: () => {}, setInput: () => {}, setLastMeta: () => {}, buildMeta: () => ({}),
    processResponse: () => {}, inputRef: { current: { focus: () => {} } },
  });
  const option = getQuickOptions("employment_nature", "ur")[0];
  callback(option.label, { field: "employment_nature", value: option.value });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(displayed[0].text, option.label);
  assert.equal(displayed[0].role, "user");
  assert.equal(JSON.parse(calls[0].body).answer_value, "paid_employee");
  assert.equal(JSON.parse(calls[0].body).answer_field, "employment_nature");
});

test("the actual correction picker sends the canonical code while displaying its translated label", async () => {
  const client = await clientPromise;
  const { getQuickOptions } = await optionsPromise;
  const source = fs.readFileSync(path.join(__dirname, "..", "pages/chat.js"), "utf8");
  const handler = source.match(/const handleStructuredCorrection = useCallback\(([\s\S]*?)\n  }, \[[^\]]*\]\);/);
  assert.ok(handler);
  const displayed = [];
  const calls = mockResponses([200, { reply: "Next question" }]);
  const callback = vm.runInNewContext(`(${handler[1]}\n})`, {
    sending: false, completed: false, sessionId: 42, token: "test-token", lang: "hi",
    sendMessage: client.sendMessage, PREFILL_LABELS: { education_level: "Education" },
    setMessages: update => displayed.push(...update([])),
    setSending: () => {}, setLastMeta: () => {}, buildMeta: () => ({}), processResponse: () => {},
    setCorrectionMode: () => {}, setCorrectingField: () => {}, setCorrectionInput: () => {},
    inputRef: { current: { focus: () => {} } },
  });
  const option = getQuickOptions("education_level", "hi")[4];
  callback("education_level", option.value, option.label);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(displayed[0].text, `Education → ${option.label}`);
  assert.deepEqual(JSON.parse(calls[0].body), {
    message: `Education → ${option.label}`, preferred_language: "hi",
    correction_field: "education_level", correction_value: "bachelor",
  });
});

test("pending high-confidence classifications require human review", async () => {
  const { getReviewStatus } = await reviewPromise;
  assert.equal(getReviewStatus({ pending_review: true, quality_status: "pass", profile: { isco_confidence: 0.95 } }), "pending");
  assert.equal(getReviewStatus({ quality_status: "escalated", profile: { isco_confidence: 0.95 } }), "pending");
  assert.equal(getReviewStatus({ quality_status: "pass", semantic_coherence: { violations: [{ severity: "HIGH" }] } }), "pending");
});

test("completed human decisions take precedence over confidence and legacy quality state", async () => {
  const { getReviewStatus } = await reviewPromise;
  assert.equal(getReviewStatus({ pending_review: false, human_review_status: "reviewed", quality_status: "escalated", profile: { isco_confidence: 0.4 } }), "reviewed");
  assert.equal(getReviewStatus({ pending_review: false, human_review_status: "rejected", quality_status: "pass", profile: { isco_confidence: 0.95 } }), "rejected");
  assert.equal(getReviewStatus({ pending_review: true, human_review_status: "reviewed" }), "pending");
});

test("unassessed reports do not claim AI verification from confidence alone", async () => {
  const { getReviewStatus } = await reviewPromise;
  assert.equal(getReviewStatus({ profile: { isco_confidence: 0.95 } }), "unknown");
  assert.equal(getReviewStatus({ pending_review: false, quality_status: "escalated" }), "unknown");
  assert.equal(getReviewStatus({ pending_review: false, quality_status: "pass" }), "verified");
});
