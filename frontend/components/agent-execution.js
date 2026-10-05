// Status comes from the completed API turn. The client has no live worker feed.
export function getAgentExecutionStatus(execution, name, pending = false) {
  if (pending) return "waiting";
  const reported = execution?.[name];
  if (reported === "completed") return "done";
  if (["cached", "skipped", "failed"].includes(reported)) return reported;
  return "idle";
}
