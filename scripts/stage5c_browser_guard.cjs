/**
 * Playwright enforcement for the Stage 5C browser contexts.
 * HTTP is intercepted with context.route. WebSockets are intercepted with
 * context.routeWebSocket. Service workers are disabled by contextOptions().
 */
const policy = require("./stage5c_browser_policy.cjs");

function contextOptions() {
  return policy.contextOptions();
}

async function installGuard(context, violations, apiOrigins) {
  await context.route("**/*", async (route) => {
    const request = route.request();
    const decision = policy.decideRequest(request.url(), request.resourceType());
    if (!decision.allow) {
      violations.push(policy.violationLabel(decision, request.url()));
      await route.abort("blockedbyclient");
      return;
    }
    if (decision.code === "api") {
      try { apiOrigins.push(new URL(request.url()).origin); } catch (_error) { apiOrigins.push(""); }
    }
    await route.continue();
  });
  if (typeof context.routeWebSocket !== "function") {
    throw policy.coded("network-blocked");
  }
  await context.routeWebSocket(/.*/, (socket) => {
    const decision = policy.decideWebSocket(socket.url());
    if (!decision.allow) {
      violations.push(policy.violationLabel(decision, socket.url()));
      socket.close({ code: 1008 });
      return;
    }
    const server = socket.connectToServer();
    socket.onMessage((message) => server.send(message));
    server.onMessage((message) => socket.send(message));
  });
}

module.exports = { contextOptions, installGuard };
