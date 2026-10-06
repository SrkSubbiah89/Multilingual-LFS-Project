const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const { transformSync } = require("next/dist/build/swc");

// Render the actual report card JSX with Next's existing compiler. This checks
// the code-presence conditional as well as the status helper, without needing
// a router or fetching a report during a unit test.
const source = fs.readFileSync(path.join(__dirname, "..", "pages/report.js"), "utf8").replaceAll("\r\n", "\n");
const card = source.match(/<div className="bg-white border border-violet-200 rounded-xl overflow-hidden shadow-sm">[\s\S]*?\n              <\/div>/);
assert.ok(card, "The actual ISCO report card must be available to render");
const translations = source.match(/const T = ([\s\S]*?\n});/);
assert.ok(translations, "The report's translated labels must be available");
const labels = new Function(`return (${translations[1]});`)();
const compiled = transformSync(`
  const React = require("react");
  const ConfidenceBar = () => null;
  module.exports = function ReportCard({ p, t, reviewStatus }) {
    const majorInfo = null;
    const majorCode = p.isco_code ? p.isco_code[0] : null;
    return (${card[0]});
  };
`, {
  jsc: { parser: { syntax: "ecmascript", jsx: true }, transform: { react: { runtime: "classic" } } },
  module: { type: "commonjs" },
});
const cardModule = { exports: {} };
new Function("require", "module", "exports", compiled.code)(require, cardModule, cardModule.exports);
const ReportCard = cardModule.exports;
const statusPromise = import(`data:text/javascript;base64,${fs.readFileSync(
  path.join(__dirname, "..", "components/review-status.js")).toString("base64")}`);

for (const language of ["en", "ar"]) {
  test(`a rejected report without an ISCO code renders its ${language} human decision`, async () => {
    const { getReviewStatus } = await statusPromise;
    const report = { profile: { isco_code: null }, pending_review: false, human_review_status: "rejected" };
    const t = labels[language];
    const html = renderToStaticMarkup(React.createElement(ReportCard, {
      p: report.profile, t, reviewStatus: getReviewStatus(report),
    }));
    assert.ok(html.includes(t.hitlRejected));
    assert.match(html, /No ISCO classification available for this session\./);
    assert.ok(!html.includes(t.hitlOk));
    assert.ok(!html.includes(t.hitlRequired));
  });
}

test("pending and completed review badges remain visible beside classified occupations", async () => {
  const { getReviewStatus } = await statusPromise;
  for (const [queue, decision, expected] of [[true, null, "hitlRequired"], [false, "reviewed", "hitlReviewed"]]) {
    const report = { profile: { isco_code: "2512", isco_confidence: 0.4 }, pending_review: queue, human_review_status: decision };
    const html = renderToStaticMarkup(React.createElement(ReportCard, {
      p: report.profile, t: labels.en, reviewStatus: getReviewStatus(report),
    }));
    assert.ok(html.includes(labels.en[expected]));
    assert.match(html, /2512/);
    assert.ok(!html.includes(labels.en.hitlRejected));
  }
});

for (const language of ["en", "ar", "ur", "hi", "tl"]) {
  test(`${language} reports identify code prefixes as taxonomy rather than executed retrieval stages`, () => {
    const t = labels[language];
    assert.ok(t.iscoHierarchy, "Every supported report language needs a hierarchy label");
    const html = renderToStaticMarkup(React.createElement(ReportCard, {
      p: { isco_code: "2512" }, t, reviewStatus: "reviewed",
    }));
    assert.ok(html.includes(t.iscoHierarchy));
    assert.ok(!html.includes("4-Stage Hierarchical Pipeline"));
    assert.ok(!html.includes("40.95%"), "Historical accuracy must not be attached to an individual ISCO classification");
  });
}
