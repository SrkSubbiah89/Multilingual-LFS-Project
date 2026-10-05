// Pending queue items and completed human decisions take precedence over
// confidence. Legacy reports without queue state retain their escalation flags.
export function getReviewStatus(report) {
  if (report?.pending_review === true) return "pending";
  if (report?.human_review_status === "rejected") return "rejected";
  if (report?.human_review_status === "reviewed") return "reviewed";
  if (report?.pending_review == null && (
    report?.hitl_required === true
    || ["escalated", "fail"].includes(report?.quality_status)
    || report?.semantic_coherence?.violations?.some(v => v.severity === "HIGH")
    || (report?.profile?.isco_confidence != null && report.profile.isco_confidence < 0.70)
  )) return "pending";
  return report?.quality_status === "pass" ? "verified" : "unknown";
}
