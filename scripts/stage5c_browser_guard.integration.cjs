/**
 * Proves the Stage 5C Playwright guard aborts real browser traffic.
 * The fixture listens only on 127.0.0.1:18765. It does not use project services.
 */
const http = require("node:http");
const crypto = require("node:crypto");
const guard = require("./stage5c_browser_guard.cjs");

const FIXTURE = "http://127.0.0.1:18765";

function acceptKey(key) {
  return crypto.createHash("sha1").update(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest("base64");
}

function listen() {
  const seen = { http: [], upgrades: [] };
  const server = http.createServer((request, response) => {
    seen.http.push(request.url.split("?")[0]);
    if (request.url === "/sw.js") {
      response.writeHead(200, { "Content-Type": "application/javascript" });
      response.end("self.addEventListener('fetch', function () {});");
      return;
    }
    if (request.url === "/redirect") {
      response.writeHead(302, { Location: "http://127.0.0.1:8000/health" });
      response.end();
      return;
    }
    response.writeHead(200, { "Content-Type": "text/html" });
    response.end("<!doctype html><title>guard</title>");
  });
  server.on("upgrade", (request, socket, head) => {
    seen.upgrades.push(request.url.split("?")[0]);
    const key = request.headers["sec-websocket-key"];
    socket.write(
      "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " +
      acceptKey(String(key || "")) + "\r\n\r\n"
    );
    if (head && head.length) socket.unshift(head);
  });
  return new Promise((resolve) => {
    server.listen(18765, "127.0.0.1", () => resolve({ server, seen }));
  });
}

async function main() {
  const playwrightPath = process.env.STAGE5C_PLAYWRIGHT_PATH;
  if (!playwrightPath) {
    console.log("BLOCKED: Playwright is not configured for the guard integration test");
    process.exit(2);
  }
  if (guard.contextOptions().serviceWorkers !== "block") {
    console.log("GUARD_INTEGRATION_FAIL service-workers");
    process.exit(1);
  }
  const { chromium } = require(playwrightPath);
  const { server, seen } = await listen();
  const violations = [];
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext(guard.contextOptions());
    await guard.installGuard(context, violations, []);
    await context.route(FIXTURE + "/**", (route) => route.continue());
    const page = await context.newPage();
    const blocked = [];
    page.on("requestfailed", (request) => {
      const failure = request.failure();
      blocked.push((failure && failure.errorText) || "");
    });
    await page.goto(FIXTURE + "/", { waitUntil: "domcontentloaded" });
    const worker = await page.evaluate(async () => {
      const registration = await navigator.serviceWorker.register("/sw.js");
      return {
        registration: Boolean(registration),
        controller: Boolean(navigator.serviceWorker.controller),
      };
    });
    const activeWorkers = typeof context.serviceWorkers === "function" ? context.serviceWorkers().length : 0;
    if (worker.registration || worker.controller || activeWorkers !== 0) throw new Error("service-worker");
    await page.evaluate(() => fetch("http://127.0.0.1:8000/health").catch(() => null));
    await page.evaluate(() => fetch("/redirect").catch(() => null));
    const socketResult = await page.evaluate(() => new Promise((resolve) => {
      const socket = new WebSocket("ws://127.0.0.1:18765/unauthorized");
      const done = (value) => resolve(value);
      socket.addEventListener("open", () => done("open"));
      socket.addEventListener("error", () => done("error"));
      socket.addEventListener("close", () => done("close"));
      setTimeout(() => done("timeout"), 3000);
    }));
    if (socketResult === "open" || seen.upgrades.length !== 0) throw new Error("websocket");
    if (!seen.http.includes("/redirect")) throw new Error("redirect-fixture");
    if (!blocked.some((text) => text.includes("ERR_BLOCKED_BY_CLIENT"))) throw new Error("port-8000");
    const labels = violations.join(" ");
    if (!labels.includes("127.0.0.1") || labels.includes("?") || labels.includes("token")) {
      throw new Error("report");
    }
    console.log("GUARD_INTEGRATION_PASS workers=" + activeWorkers + " blocked=" + blocked.length);
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}

main().catch((error) => {
  const name = error && error.message ? error.message.split(/\s/)[0] : "unexpected-failure";
  const safe = ["service-worker", "websocket", "redirect-fixture", "port-8000", "report"].includes(name)
    ? name
    : "unexpected-failure";
  console.log("GUARD_INTEGRATION_FAIL " + safe);
  process.exit(1);
});
