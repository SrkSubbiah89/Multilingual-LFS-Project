/**
 * Stage 5C browser journey for the GitHub-hosted Ubuntu job.
 *
 * The job creates its own loopback PostgreSQL and Redis. This file does not
 * use PowerShell, WSL, SSH, or the laptop databases. Reports contain public
 * result codes only.
 */
const { spawnSync } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const policy = require("./stage5c_browser_policy.cjs");
const guard = require("./stage5c_browser_guard.cjs");

const API = policy.API_ORIGIN;
const WEB = policy.WEB_ORIGIN;
const MIN_FREE_MB = 3072;
const PREFERRED_FREE_MB = 4096;
const LOGIN_LABELS = {
  en: ["Send verification code", "ltr"],
  ar: ["إرسال رمز التحقق", "rtl"],
  ur: ["تصدیقی کوڈ بھیجیں", "rtl"],
  hi: ["सत्यापन कोड भेजें", "ltr"],
  tl: ["Ipadala ang verification code", "ltr"],
};

function blocked(message) {
  const allowed = new Set([
    "GitHub-hosted browser execution is not authorized",
    "API or frontend target is not the disposable loopback",
    "the browser runner must not receive a database URL",
    "SQL target is not the CI disposable database",
    "CI database password was not provided",
    "the browser runner requires the Ubuntu job",
    "free memory is below the required margin",
    "Playwright 1.55.0 is not available outside the frontend",
    "disposable API storage is not ready",
  ]);
  console.log("BLOCKED: " + (allowed.has(message) ? message : "execution was not authorized"));
  process.exit(2);
}

function assertTargets() {
  if (process.env.STAGE5C_EXECUTE !== "1" || process.env.STAGE5C_CI !== "1" || process.env.GITHUB_ACTIONS !== "true") {
    blocked("GitHub-hosted browser execution is not authorized");
  }
  if (process.env.STAGE5C_API_URL !== API || process.env.STAGE5C_WEB_URL !== WEB) {
    blocked("API or frontend target is not the disposable loopback");
  }
  if (process.env.DATABASE_URL || process.env.STAGE5C_DATABASE_URL) {
    blocked("the browser runner must not receive a database URL");
  }
  if (process.env.PGHOST !== "127.0.0.1" || process.env.PGPORT !== "5432" ||
      process.env.PGUSER !== "lfs_stage5c_ci" || process.env.PGDATABASE !== "lfs_stage5c_ci") {
    blocked("SQL target is not the CI disposable database");
  }
  if (!process.env.PGPASSWORD) blocked("CI database password was not provided");
}

function freeMemoryMb() {
  if (process.platform !== "linux") blocked("the browser runner requires the Ubuntu job");
  const text = fs.readFileSync("/proc/meminfo", "utf8");
  const match = text.match(/MemAvailable:\s+(\d+)\s+kB/);
  return match ? Math.floor(Number(match[1]) / 1024) : NaN;
}

function loadPlaywright() {
  const configured = process.env.STAGE5C_PLAYWRIGHT_PATH;
  if (!configured || !fs.existsSync(path.resolve(configured))) blocked("Playwright 1.55.0 is not available outside the frontend");
  const resolved = fs.realpathSync(path.resolve(configured));
  const frontendModules = path.resolve(__dirname, "..", "frontend", "node_modules") + path.sep;
  if (resolved.startsWith(frontendModules)) blocked("Playwright 1.55.0 is not available outside the frontend");
  const pkg = JSON.parse(fs.readFileSync(path.join(resolved, "package.json"), "utf8"));
  if (pkg.name !== "playwright" || pkg.version !== "1.55.0") blocked("Playwright 1.55.0 is not available outside the frontend");
  return require(resolved);
}

function sql(statement) {
  const lowered = " " + statement.toLowerCase().replace(/\s+/g, " ");
  if (!lowered.startsWith(" select ")) throw policy.coded("protected-query");
  for (const banned of ["insert ", "update ", "delete ", "alter ", "drop ", "truncate ",
    "lfs_native_test", "lfs_db", "15432", "25432"]) {
    if (lowered.includes(banned)) throw policy.coded("protected-query");
  }
  const result = spawnSync("psql", [
    "-h", "127.0.0.1", "-p", "5432", "-U", "lfs_stage5c_ci", "-d", "lfs_stage5c_ci",
    "-v", "ON_ERROR_STOP=1", "-Atc", statement,
  ], { encoding: "utf8", env: process.env });
  if (result.error || result.status !== 0) throw policy.coded("sql-failed");
  return result.stdout.trim();
}

async function assertBundle(page) {
  await page.goto(WEB + "/", { waitUntil: "domcontentloaded" });
  const sources = await page.locator("script[src]").evaluateAll((nodes) => nodes.map((node) => node.src));
  let corpus = "";
  for (const source of sources) {
    const url = new URL(source);
    if (url.origin !== WEB) throw policy.coded("bundle-script");
    const response = await fetch(source);
    if (!response.ok) throw policy.coded("bundle-script");
    corpus += await response.text() + "\n";
  }
  const assessed = policy.assessInlinedOrigin(corpus);
  if (!assessed.ok) throw policy.coded(assessed.code);
  return assessed.fallbackTextPresent
    ? "configured API origin is inlined; an unused fallback string is not a request"
    : "configured API origin is inlined";
}

async function waitForReadyChat(page) {
  await page.waitForURL("**/chat", { timeout: 30000 });
  try {
    await page.waitForFunction(() => {
      const id = localStorage.getItem("lfs_active_session");
      const box = document.querySelector("textarea");
      return /^[1-9][0-9]{0,17}$/.test(id || "") && !!box && !box.disabled;
    }, null, { timeout: 20000 });
  } catch (_error) {
    throw policy.coded("chat-timeout");
  }
  const sessionId = await page.evaluate(() => localStorage.getItem("lfs_active_session"));
  if (!policy.sessionReady(sessionId, true)) throw policy.coded("chat-timeout");
  return sessionId;
}

async function login(page, email, language) {
  await page.goto(WEB + "/", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: language, exact: true }).click();
  await page.locator("#email").fill(email);
  await page.locator("button[type=submit]").click();
  await page.locator("#otp").waitFor({ state: "visible", timeout: 20000 });
  const filled = await page.locator("#otp").evaluate((element) => /^\d{6}$/.test(element.value));
  if (!filled) throw policy.coded("otp-not-filled");
  await page.locator("button[type=submit]").click();
  return waitForReadyChat(page);
}

async function visibleText(page) {
  return page.locator("body").innerText();
}

function messageResponse(page) {
  return page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.origin === API && url.pathname.endsWith("/message") && response.request().method() === "POST";
  }, { timeout: 60000 });
}

function monotonicMs() {
  return Number(process.hrtime.bigint() / 1000000n);
}

function installMessagePacer() {
  let previous = null;
  return async function paceBeforeMessage() {
    const wait = policy.messageSpacingMs(previous, monotonicMs());
    if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
    previous = monotonicMs();
  };
}

let paceBeforeMessage = async function paceBeforeMessage() {};

function attachExceptionType(diagnostic) {
  const reference = / reference=([0-9a-f]{12})$/.exec(String(diagnostic || ""));
  const logPath = policy.disposableApiLog(process.env.STAGE5C_API_LOG || "");
  if (!reference || !logPath) return diagnostic;
  let text = "";
  try {
    text = fs.readFileSync(logPath, "utf8");
  } catch (_error) {
    return diagnostic;
  }
  const type = policy.exceptionTypeForReference(text, reference[1]);
  return type ? diagnostic + " exception=" + type : diagnostic;
}

async function rejectHttp(response, stepIndex, field) {
  let detail = "";
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string" && body.detail.length <= 200) detail = body.detail;
  } catch (_error) {
    detail = "";
  }
  const error = policy.coded("answer-http");
  error.diagnostic = attachExceptionType(policy.answerHttpDetail(stepIndex, field, response.status(), detail));
  throw error;
}

async function assistantBubbles(page) {
  return page.locator("main > div").evaluateAll((nodes) => nodes.filter((node) => {
    return (node.getAttribute("style") || "").includes("flex-start");
  }).map((node) => {
    const bubble = node.querySelector("div");
    return bubble ? bubble.innerText : "";
  }));
}

async function waitForActiveQuestion(page, prompt) {
  try {
    await page.waitForFunction((expected) => {
      const assistants = [...document.querySelectorAll("main > div")].filter((node) => {
        return (node.getAttribute("style") || "").includes("flex-start");
      });
      const last = assistants[assistants.length - 1];
      const bubble = last && last.querySelector("div");
      return !!(bubble && bubble.innerText.includes(expected));
    }, prompt, { timeout: 60000 });
  } catch (_error) {
    throw policy.coded("questionnaire-order");
  }
}

async function submitJourneyStep(page, step, nextStep, stepIndex) {
  await waitForActiveQuestion(page, step.prompt);
  if (!policy.activeQuestion(await assistantBubbles(page), step.prompt)) throw policy.coded("questionnaire-order");
  const beforeCount = (await assistantBubbles(page)).length;
  const pending = messageResponse(page);
  if (step.kind === "text") {
    await page.locator("textarea").fill(step.payload.message);
  }
  await paceBeforeMessage();
  if (step.kind === "text") {
    await page.locator("footer form button[type=submit]").click();
  } else {
    await page.getByRole("button", { name: step.name, exact: true }).click();
  }
  const response = await pending;
  if (response.status() >= 400) await rejectHttp(response, stepIndex, step.field);
  const sent = response.request().postDataJSON();
  if (!policy.sameRequestPayload(sent, step.payload)) throw policy.coded("answer-payload");
  let body;
  try {
    body = await response.json();
  } catch (_error) {
    throw policy.coded("questionnaire-order");
  }
  if (!policy.serverProgress(body, nextStep ? nextStep.field : "validating")) throw policy.coded("questionnaire-order");
  try {
    await page.waitForFunction((count) => {
      const assistants = [...document.querySelectorAll("main > div")].filter((node) => {
        return (node.getAttribute("style") || "").includes("flex-start");
      });
      return assistants.length > count;
    }, beforeCount, { timeout: 60000 });
  } catch (_error) {
    throw policy.coded("questionnaire-order");
  }
  if (nextStep) {
    await waitForActiveQuestion(page, nextStep.prompt);
    if (nextStep.kind === "choice") {
      await page.getByRole("button", { name: nextStep.name, exact: true }).waitFor({ timeout: 60000 });
    } else {
      await page.locator("textarea:not([disabled])").waitFor({ timeout: 60000 });
    }
  }
  return response;
}

function runCompletedAnswerRegression() {
  const command = policy.immutabilityCommand();
  const env = { ...process.env };
  for (const key of [
    "DATABASE_URL", "STAGE5C_DATABASE_URL", "PGPASSWORD", "PGHOST", "PGPORT", "PGUSER", "PGDATABASE",
    "QDRANT_URL", "QDRANT_API_KEY",
  ]) {
    delete env[key];
  }
  return spawnSync(process.env.STAGE5C_PYTHON || "python3", [command.script, ...command.args], {
    encoding: "utf8",
    cwd: path.resolve(__dirname, ".."),
    env: env,
    timeout: 300000,
  });
}

async function main() {
  assertTargets();
  const freeMb = freeMemoryMb();
  if (!Number.isFinite(freeMb) || freeMb < MIN_FREE_MB) blocked("free memory is below the required margin");
  if (freeMb < PREFERRED_FREE_MB) console.log("MEMORY_WARNING: free memory is below the preferred margin");
  const { chromium } = loadPlaywright();
  const healthResponse = await fetch(API + "/health");
  if (!healthResponse.ok) blocked("disposable API storage is not ready");
  const health = await healthResponse.json();
  if (health.required?.database !== "ok" || health.required?.redis !== "ok") blocked("disposable API storage is not ready");
  const modelsDown = ["qdrant", "ollama"].some((name) => health.diagnostics?.[name] !== "ok");
  const results = [];
  const record = (name, status, detail) => {
    let safe = detail;
    if (status === "FAIL") {
      if (policy.isSafeAnswerHttpDetail(detail)) {
        safe = detail;
      } else {
        const code = policy.publicDetail({ code: detail });
        safe = code === detail || /^network-blocked [a-z-]+ [a-z0-9.-]+$/i.test(String(detail))
          ? detail
          : "unexpected-failure";
      }
    }
    results.push({ name, status, detail: safe });
  };
  record("classification-and-report", "SKIP",
    "Qdrant, Ollama, and the classification crew stay unused; this skip is not a classification or report pass");
  record("live-model-execution", "NOT_RUN", "no model was downloaded or executed");
  if (!modelsDown) record("model-health", "FAIL", "models-present");

  const run = crypto.randomBytes(4).toString("hex");
  const email = (label) => `stage5c-${label}-${run}@stage5b.example.com`;
  const evidence = {
    emails: { en: email("en"), intruder: email("intruder"), ar: email("ar") },
    sessions: {},
    answerSubmitted: false,
  };
  const browser = await chromium.launch({ headless: true });
  const violations = [];
  const apiOrigins = [];
  let journeyFailed = false;
  try {
    const context = await browser.newContext(guard.contextOptions());
    await guard.installGuard(context, violations, apiOrigins);
    const page = await context.newPage();
    try {
      record("bundle-api-origin", "PASS", await assertBundle(page));
    } catch (error) {
      record("bundle-api-origin", "FAIL", policy.publicDetail(error));
      journeyFailed = true;
    }
    for (const [code, [label, direction]] of Object.entries(LOGIN_LABELS)) {
      if (journeyFailed) {
        record("login-interface-" + code, "NOT_RUN", "not reached after an earlier failure");
        continue;
      }
      try {
        await page.getByRole("button", { name: code.toUpperCase(), exact: true }).click();
        const visible = await page.getByRole("button", { name: label, exact: true }).isVisible();
        const actual = await page.locator("div[dir]").first().getAttribute("dir");
        if (!visible || actual !== direction) {
          record("login-interface-" + code, "FAIL", "login-direction");
          journeyFailed = true;
        } else {
          record("login-interface-" + code, "PASS", "login control rendered");
        }
      } catch (_error) {
        record("login-interface-" + code, "FAIL", "login-direction");
        journeyFailed = true;
      }
    }
    let englishSession = "";
    if (!journeyFailed) {
      try {
        englishSession = await login(page, evidence.emails.en, "EN");
        evidence.sessions.en = { id: englishSession, expectedLanguage: "ar" };
        record("otp-login-and-session", "PASS", "chat initialized with a stored session");
      } catch (error) {
        record("otp-login-and-session", "FAIL", policy.publicDetail(error));
        journeyFailed = true;
      }
    }
    if (englishSession && modelsDown) {
      paceBeforeMessage = installMessagePacer();
      try {
        const steps = policy.employedJourney();
        for (let index = 0; index < steps.length; index += 1) {
          await submitJourneyStep(page, steps[index], steps[index + 1] || null, index);
          if (index === 0) {
            evidence.answerSubmitted = true;
            record("answer-submission", "PASS", "employment status control returned a non-error response");
          }
        }
        record("questionnaire-completion", "PASS", "required employed questions were answered through the displayed controls");
      } catch (error) {
        if (!evidence.answerSubmitted) record("answer-submission", "FAIL", policy.publicDetail(error));
        record("questionnaire-completion", "FAIL", error.diagnostic || policy.publicDetail(error));
      }
      let validating = false;
      if (evidence.answerSubmitted && results.some((result) => result.name === "questionnaire-completion" && result.status === "PASS")) {
        try {
          await page.getByRole("button", { name: policy.CORRECTION_CONTROL }).waitFor({ timeout: 20000 });
          const header = await page.locator("header").innerText();
          if (!header.toLowerCase().includes("validating")) throw policy.coded("validating-screen");
          validating = true;
          record("validating-screen", "PASS", "the interview reached the validating screen");
        } catch (error) {
          record("validating-screen", "FAIL", policy.publicDetail(error));
        }
      } else {
        record("validating-screen", "FAIL", "questionnaire-order");
      }
      if (validating) {
        try {
          evidence.activeBefore = sql(policy.activeAnswerQuery(englishSession));
          evidence.wageBefore = sql(policy.wageRevisionQuery(englishSession));
          if (!policy.canonicalRequiredSnapshot(evidence.activeBefore)) throw policy.coded("answer-preserved");
          if (!policy.assessOriginalWage(evidence.wageBefore)) throw policy.coded("wage-revision");
          await page.getByRole("button", { name: policy.CORRECTION_CONTROL }).click();
          record("correction-control", "PASS", "the displayed correction control was used");
          await page.getByRole("button", { name: new RegExp("^" + policy.WAGE_FIELD_LABEL + "\\b") }).click();
          const revised = page.getByRole("button", { name: policy.WAGE_REVISED_LABEL, exact: true });
          await revised.waitFor({ timeout: 20000 });
          const pending = messageResponse(page);
          await paceBeforeMessage();
          await revised.click();
          const response = await pending;
          if (response.status() >= 400) {
            await rejectHttp(response, policy.employedJourney().length, policy.wageCorrectionPayload().correction_field);
          }
          if (!policy.sameRequestPayload(response.request().postDataJSON(), policy.wageCorrectionPayload())) {
            throw policy.coded("correction-payload");
          }
          record("correction-payload", "PASS", "the structured correction named monthly_wage_range and 20001_50000");
          await page.getByRole("button", { name: policy.CORRECTION_CONTROL }).waitFor({ timeout: 20000 });
          const afterHeader = await page.locator("header").innerText();
          if (!afterHeader.toLowerCase().includes("validating")) throw policy.coded("validating-screen");
          evidence.correctionSubmitted = true;
          evidence.activeAfter = sql(policy.activeAnswerQuery(englishSession));
          evidence.wageAfter = sql(policy.wageRevisionQuery(englishSession));
        } catch (error) {
          if (!results.some((result) => result.name === "correction-control")) {
            record("correction-control", "FAIL", error.diagnostic || policy.publicDetail(error));
          }
          if (!results.some((result) => result.name === "correction-payload")) {
            record("correction-payload", "FAIL", error.diagnostic || policy.publicDetail(error));
          }
        }
      } else {
        record("correction-control", "FAIL", "validating-screen");
        record("correction-payload", "FAIL", "validating-screen");
      }
      try {
        await page.getByRole("button", { name: "AR", exact: true }).click({ timeout: 15000 });
        await page.locator("div[dir=rtl]").first().waitFor({ timeout: 15000 });
        record("language-switch-display", "PASS", "Arabic chat direction is rtl");
      } catch (_error) {
        record("language-switch-display", "FAIL", "login-direction");
      }
    } else if (englishSession) {
      record("answer-submission", "FAIL", "models-present");
      record("questionnaire-completion", "FAIL", "models-present");
      record("validating-screen", "FAIL", "models-present");
      record("correction-control", "FAIL", "models-present");
      record("correction-payload", "FAIL", "models-present");
      record("language-switch-display", "FAIL", "models-present");
    } else {
      record("answer-submission", "FAIL", "session-id");
      record("questionnaire-completion", "FAIL", "session-id");
      record("validating-screen", "FAIL", "session-id");
      record("correction-control", "FAIL", "session-id");
      record("correction-payload", "FAIL", "session-id");
      record("language-switch-display", "FAIL", "session-id");
    }

    if (!journeyFailed) {
      const intruderContext = await browser.newContext(guard.contextOptions());
      await guard.installGuard(intruderContext, violations, apiOrigins);
      const intruder = await intruderContext.newPage();
      try {
        const intruderSession = await login(intruder, evidence.emails.intruder, "EN");
        const before = await visibleText(intruder);
        if (before.includes(evidence.emails.en)) throw policy.coded("isolation-body");
        const denied = intruder.waitForResponse((response) => {
          return response.url() === API + "/survey/sessions/" + englishSession + "/conversation" &&
            response.request().method() === "GET";
        }, { timeout: 20000 });
        await intruder.evaluate((sessionId) => localStorage.setItem("lfs_active_session", sessionId), englishSession);
        await intruder.reload({ waitUntil: "domcontentloaded" });
        const deniedResponse = await denied;
        const deniedBody = await deniedResponse.text();
        if (deniedResponse.status() !== 404) throw policy.coded("isolation-status");
        if (deniedBody.includes(evidence.emails.en) || !deniedBody.includes("Session not found.")) {
          throw policy.coded("isolation-body");
        }
        await intruder.waitForFunction((oldId) => {
          const current = localStorage.getItem("lfs_active_session");
          const box = document.querySelector("textarea");
          return /^[1-9][0-9]{0,17}$/.test(current || "") && current !== oldId && !!box && !box.disabled;
        }, englishSession, { timeout: 20000 });
        const afterId = await intruder.evaluate(() => localStorage.getItem("lfs_active_session"));
        const after = await visibleText(intruder);
        if (!afterId || afterId === englishSession || afterId === intruderSession || after.includes(evidence.emails.en)) {
          throw policy.coded("isolation-session");
        }
        evidence.sessions.intruder = { id: afterId, expectedLanguage: "en" };
        record("cross-user-isolation", "PASS", "unauthorized conversation read returned 404 and a separate session was opened");
      } catch (error) {
        record("cross-user-isolation", "FAIL", policy.publicDetail(error));
      } finally {
        await intruderContext.close();
      }

      const arabicContext = await browser.newContext(guard.contextOptions());
      await guard.installGuard(arabicContext, violations, apiOrigins);
      const arabic = await arabicContext.newPage();
      try {
        await arabic.goto(WEB + "/", { waitUntil: "domcontentloaded" });
        await arabic.getByRole("button", { name: "AR", exact: true }).click();
        const direction = await arabic.locator("div[dir]").first().getAttribute("dir");
        if (direction !== "rtl") throw policy.coded("login-direction");
        record("arabic-login-interface", "PASS", "login direction rtl");
      } catch (error) {
        if (!results.some((result) => result.name === "arabic-login-interface")) {
          record("arabic-login-interface", "FAIL", policy.publicDetail(error));
        }
      }
      try {
        const arabicSession = await login(arabic, evidence.emails.ar, "AR");
        evidence.sessions.ar = { id: arabicSession, expectedLanguage: "ar" };
        const chatDirection = await arabic.locator("div[dir]").first().getAttribute("dir");
        if (chatDirection !== "rtl") throw policy.coded("login-direction");
        record("arabic-session-display", "PASS", "Arabic session opened right to left");
      } catch (error) {
        record("arabic-session-display", "FAIL", policy.publicDetail(error));
      } finally {
        await arabicContext.close();
      }
    }
  } catch (error) {
    record("browser-journey", "FAIL", policy.publicDetail(error));
  } finally {
    await browser.close();
  }
  const apiAssessment = policy.assessApiOrigins(apiOrigins);
  record("effective-api-origin", apiAssessment.ok ? "PASS" : "FAIL", apiAssessment.code);
  record("request-targets", violations.length === 0 ? "PASS" : "FAIL",
    violations.length ? violations[0] : "browser requests stayed on the allowlist");
  verifyCommitted(evidence, record);
  const count = (status) => results.filter((result) => result.status === status).length;
  const failed = count("FAIL");
  const stage = failed > 0 ? "FAIL" : "PARTIAL";
  const summary = [
    "SUMMARY passed=" + count("PASS") + " failed=" + failed + " skipped=" + count("SKIP") + " not_run=" + count("NOT_RUN"),
    "SKIP is not a pass. NOT_RUN is not a pass.",
    "Answer correction is accepted only when its own checks passed.",
    "The isolated SQLite 409 regression is not a live PostgreSQL HTTP check.",
    "Classification and model-generated reports were not accepted.",
    "A green check accepts only the non-AI checks that ran.",
    "STAGE_RESULT=" + stage,
    "REQUIRED_RESULT=" + (failed > 0 ? "FAIL" : "PASS"),
  ];
  const lines = results.map((result) => result.status + " " + result.name + " — " + result.detail).concat(summary);
  const safeLines = policy.scrubReport(lines, Object.values(evidence.emails));
  if (safeLines.some((line, index) => line !== lines[index])) {
    console.log("FAIL report — unexpected-failure");
    console.log("STAGE_RESULT=FAIL");
    process.exit(1);
  }
  for (const line of safeLines) console.log(line);
  if (process.env.STAGE5C_REPORT) fs.writeFileSync(process.env.STAGE5C_REPORT, safeLines.join("\n") + "\n");
  process.exit(failed > 0 ? 1 : 0);
}

function verifyCommitted(evidence, record) {
  try {
    const emails = Object.values(evidence.emails);
    const listed = sql("SELECT count(*) FROM users WHERE email IN (" + emails.map(policy.emailLiteral).join(", ") + ")");
    const strangers = sql("SELECT count(*) FROM users WHERE email NOT IN (" + emails.map(policy.emailLiteral).join(", ") + ")");
    record("disposable-users-only", strangers === "0" && listed === String(emails.length) ? "PASS" : "FAIL",
      strangers === "0" && listed === String(emails.length) ? "only the three synthetic accounts exist" : "sql-failed");
    const owners = {};
    for (const [label, item] of Object.entries(evidence.sessions)) {
      const owner = sql(policy.ownerQuery(item.id, evidence.emails[label]));
      if (!/^[1-9][0-9]{0,17}$/.test(owner)) {
        record("session-owner-" + label, "FAIL", "session-id");
        continue;
      }
      owners[label] = owner;
      record("session-owner-" + label, "PASS", "session owner matched the synthetic account");
      const language = sql("SELECT language FROM survey_sessions WHERE id = " + policy.assertSessionId(item.id));
      record("session-language-" + label, language === item.expectedLanguage ? "PASS" : "FAIL",
        language === item.expectedLanguage ? "stored language matched" : "session-id");
      const audit = sql(policy.auditQuery(item.id, owner));
      record("session-start-audit-" + label, audit === "1" ? "PASS" : "FAIL",
        audit === "1" ? "one session_started row belongs to the owner" : "sql-failed");
      const access = sql(policy.accessQuery(item.id, owner));
      record("session-start-data-access-" + label, access === "1" ? "PASS" : "FAIL",
        access === "1" ? "one data-access write belongs to the owner" : "sql-failed");
    }
    for (const label of ["en", "intruder", "ar"]) {
      if (!evidence.sessions[label]) record("session-records-" + label, "FAIL", "session-id");
    }
    if (owners.en && owners.intruder) {
      record("session-owners-distinct", owners.en === owners.intruder ? "FAIL" : "PASS",
        owners.en === owners.intruder ? "isolation-session" : "the two accounts own different users");
    }
    if (!evidence.answerSubmitted || !evidence.sessions.en) {
      record("answer-persistence", "FAIL", "answer-missing");
    } else {
      const active = evidence.activeAfter || sql(policy.activeAnswerQuery(evidence.sessions.en.id));
      const retired = sql(policy.retiredAnswerQuery(evidence.sessions.en.id));
      const matched = policy.employmentStillActive(active, retired);
      record("answer-persistence", matched ? "PASS" : "FAIL",
        matched ? "active employment_status revision is employed and has no superseded row" : "answer-missing");
    }
    if (!evidence.correctionSubmitted || !evidence.sessions.en || !owners.en) {
      record("sql-superseded-revision", "FAIL", "wage-revision");
      record("active-answer-uniqueness", "FAIL", "wage-revision");
      record("required-answers-preserved", "FAIL", "answer-preserved");
    } else {
      const revised = policy.assessWageRevision(evidence.wageBefore, evidence.wageAfter);
      const owned = sql(policy.wageOwnerQuery(evidence.sessions.en.id, owners.en));
      record("sql-superseded-revision", revised && owned === "2" ? "PASS" : "FAIL",
        revised && owned === "2" ? "the original wage row is retired and the new row supersedes it" : "wage-revision");
      record("active-answer-uniqueness",
        policy.uniqueActiveWage(evidence.activeAfter) ? "PASS" : "FAIL",
        policy.uniqueActiveWage(evidence.activeAfter)
          ? "one active monthly_wage_range revision remains" : "wage-revision");
      record("required-answers-preserved",
        policy.preservedRequiredAnswers(evidence.activeBefore, evidence.activeAfter) ? "PASS" : "FAIL",
        policy.preservedRequiredAnswers(evidence.activeBefore, evidence.activeAfter)
          ? "required answers matched their canonical values and kept their original ids" : "answer-preserved");
    }
    if (evidence.sessions.intruder) {
      const intruderRows = sql(policy.activeAnswerQuery(evidence.sessions.intruder.id));
      record("intruder-answer-absent", intruderRows === "" ? "PASS" : "FAIL",
        intruderRows === "" ? "the second account has no active answer" : "isolation-body");
    }
  } catch (error) {
    record("committed-sql", "FAIL", policy.publicDetail(error));
  }
  record("completed-session-cloud-http-immutability", "NOT_RUN",
    "the disposable cloud API was not asked to change a completed session");
  try {
    const immutability = runCompletedAnswerRegression();
    const immutabilityPassed = immutability.status === 0 && /1 passed/.test(immutability.stdout || "");
    record("completed-session-sqlite-regression", immutabilityPassed ? "PASS" : "FAIL",
      immutabilityPassed
        ? "isolated sqlite regression returned 409; this is not a live PostgreSQL HTTP check"
        : "immutability");
  } catch (error) {
    record("completed-session-sqlite-regression", "FAIL", policy.publicDetail(error));
  }
}

main().catch((error) => {
  console.log("FAIL browser-runner — " + policy.publicDetail(error));
  console.log("STAGE_RESULT=FAIL");
  console.log("Classification and model-generated reports were not accepted.");
  process.exit(1);
});
