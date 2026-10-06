const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { createRequire } = require("node:module");
const { test } = require("node:test");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const { transformSync } = require("next/dist/build/swc");

const root = path.resolve(__dirname, "../..");
const componentPath = path.join(root, "frontend/components/ParentDocumentEvidence.js");
const source = fs.readFileSync(componentPath, "utf8").replaceAll("\r\n", "\n");
const data = require("../data/parent-document-comparison-2026-10-06.json");
const literal = source.match(/const T = ([\s\S]*?\n});/);
assert.ok(literal);
const labels = new Function(`return (${literal[1]});`)();
const compiled = transformSync(source, {
  jsc: { parser: { syntax: "ecmascript", jsx: true }, transform: { react: { runtime: "automatic" } } },
  module: { type: "commonjs" },
});
const componentModule = { exports: {} };
new Function("require", "module", "exports", compiled.code)(createRequire(componentPath), componentModule, componentModule.exports);
const Evidence = componentModule.exports.default;
const METHODS = ["dense_flat", "parent_document_rag"];
const METRICS = ["n", "top1_correct", "top1_accuracy", "top1_wilson95_case_level", "top5_correct", "top5_accuracy"];
const metricSubset = metric => Object.fromEntries(METRICS.map(key => [key, metric[key]]));

test("published comparison preserves source counts, splits and frozen settings", () => {
  assert.equal(data.schema_version, 1);
  assert.equal(data.embedding_model, "intfloat/multilingual-e5-small");
  assert.equal(data.catalogue_profile, "official_ilo2021_v1_enriched");
  assert.equal(data.reranker, "off");
  assert.deepEqual(data.selected, { child_weight: 0.5, aggregation: "max" });
  assert.deepEqual(data.comparisons.map(row => [row.split, row.n]), [["heldout", 18747], ["validation", 642]]);
  assert.deepEqual(data.comparisons[0].methods.dense_flat.top1_correct, 6102);
  assert.deepEqual(data.comparisons[0].methods.parent_document_rag.top1_correct, 7279);
  assert.equal(data.historically_reused_benchmark, true);
  assert.equal(data.prior_heldout_informed_enrichment, true);
  assert.equal(data.labour_force_survey_field_validation, false);
  assert.equal(data.case_level_intervals_descriptive, true);
  for (const comparison of data.comparisons) {
    for (const method of METHODS) {
      const metric = comparison.methods[method];
      assert.equal(metric.n, comparison.n);
      assert.ok(Math.abs(metric.top1_accuracy - metric.top1_correct / metric.n) < 1e-12);
      assert.ok(Math.abs(metric.top5_accuracy - metric.top5_correct / metric.n) < 1e-12);
      const byLanguage = Object.values(comparison.per_language).map(row => row[method]);
      assert.equal(byLanguage.reduce((sum, row) => sum + row.n, 0), metric.n);
      assert.equal(byLanguage.reduce((sum, row) => sum + row.top1_correct, 0), metric.top1_correct);
      assert.equal(byLanguage.reduce((sum, row) => sum + row.top5_correct, 0), metric.top5_correct);
      assert.ok(metric.top1_wilson95_case_level[0] < metric.top1_accuracy);
      assert.ok(metric.top1_wilson95_case_level[1] > metric.top1_accuracy);
    }
  }
});

test("published summary exactly matches completed private reports when available", () => {
  for (const comparison of data.comparisons) {
    const reportPath = path.join(root, "Software/rag_accuracy_2026-10-06", comparison.source_report);
    if (!fs.existsSync(reportPath)) continue; // Source runs are intentionally ignored in clean checkouts.
    const bytes = fs.readFileSync(reportPath);
    assert.equal(crypto.createHash("sha256").update(bytes).digest("hex"), comparison.source_report_sha256);
    const original = JSON.parse(bytes);
    assert.equal(original.n, comparison.n);
    assert.equal(original.split, comparison.split);
    assert.equal(original.selection_sha256, data.selection_sha256);
    for (const method of METHODS) assert.deepEqual(comparison.methods[method], metricSubset(original.methods[method]));
    for (const [language, metrics] of Object.entries(comparison.per_language)) {
      for (const method of METHODS) assert.deepEqual(metrics[method], metricSubset(original.per_language[language][method]));
    }
  }
});

for (const language of ["en", "ar", "ur", "hi", "tl"]) {
  test(`${language} evidence renders both reused splits, exact metrics and scope limitations`, () => {
    const t = labels[language];
    const html = renderToStaticMarkup(React.createElement(Evidence, { language }));
    for (const key of ["title", "heldout", "validation", "limitation", "intervalNote", "comparisonNote"]) assert.ok(html.includes(t[key]), key);
    for (const value of ["6,102/18,747", "7,279/18,747", "32.55%", "38.83%", "209/642", "236/642", "36.76%", "[38.13%, 39.53%]"]) assert.ok(html.includes(value), value);
    // Validation Arabic decreases: the card must preserve that result as well.
    assert.ok(html.includes("36/128 · 28.13%"));
    assert.ok(html.includes("30/128 · 23.44%"));
    for (const value of Object.values(t.languages)) assert.ok(html.includes(value));
    assert.ok(html.includes("E5-large"));
    assert.ok(!html.includes("undefined"));
  });
}

test("new evidence leaves the historical WISCO benchmark numbers intact", () => {
  const report = fs.readFileSync(path.join(root, "frontend/pages/report.js"), "utf8").replaceAll("\r\n", "\n");
  const historical = report.match(/const WISCO_EVAL = ([\s\S]*?\n});/);
  assert.ok(historical);
  const baseline = new Function(`return (${historical[1]});`)();
  assert.equal(baseline.heldoutCases, 18747);
  assert.deepEqual(baseline.headline.map(row => [row.key, row.correct, row.accuracy]), [
    ["flat", 3973, 0.2119], ["hier", 1941, 0.1035],
  ]);
  assert.ok(report.includes("<ParentDocumentEvidence language={lang} />"));
  const serialized = JSON.stringify(data);
  assert.doesNotMatch(serialized, /case_id|input_text|gold_isco|Bearer|localhost|127\.0\.0\.1|[A-Z]:\\\\/);
});
