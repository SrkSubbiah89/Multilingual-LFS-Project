const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const modulePromise = import(`data:text/javascript;base64,${fs.readFileSync(
  path.join(__dirname, "..", "components", "agent-execution.js")).toString("base64")}`);

test("component results reflect reported completion, cached, skipped and failed states", async () => {
  const { getAgentExecutionStatus } = await modulePromise;
  const execution = { ISCOClassifier: "cached", ISICClassifier: "completed",
    LanguageProcessor: "skipped", ISCEDClassifier: "failed" };
  assert.equal(getAgentExecutionStatus(execution, "ISCOClassifier"), "cached");
  assert.equal(getAgentExecutionStatus(execution, "ISICClassifier"), "done");
  assert.equal(getAgentExecutionStatus(execution, "LanguageProcessor"), "skipped");
  assert.equal(getAgentExecutionStatus(execution, "ISCEDClassifier"), "failed");
  assert.equal(getAgentExecutionStatus(execution, "ClassificationEvidenceAuditor"), "idle");
});

test("waiting for a response does not invent active agent execution", async () => {
  const { getAgentExecutionStatus } = await modulePromise;
  assert.equal(getAgentExecutionStatus({ ISCOClassifier: "completed" }, "ISCOClassifier", true), "waiting");
  assert.equal(getAgentExecutionStatus(undefined, "LanguageProcessor", true), "waiting");
  assert.equal(getAgentExecutionStatus(undefined, "LanguageProcessor"), "idle");
});
