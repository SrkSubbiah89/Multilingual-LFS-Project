/* Actual browser checks against explicitly isolated smoke fixtures.
 * Install the browser driver outside project dependencies:
 * npm install --prefix Software/browser-smoke-tools --no-audit --no-fund --package-lock=false playwright
 * node scripts/browser_smoke.cjs --fixture Software/browser_smoke/fixtures.json
 * The fixture contains secrets and must stay in ignored Software/. Results do not.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

function argument(name, fallback) {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
}
const root = path.resolve(__dirname, '..');
const fixturePath = path.resolve(argument('--fixture', path.join(root, 'Software/browser_smoke/fixtures.json')));
const output = path.resolve(argument('--output', path.join(root, 'Documentation/ACTIVATION_2026-10-05_BROWSER_RESULTS.json')));
const screenshots = path.resolve(argument('--screenshots', path.join(root, 'Software/browser_smoke/screenshots', path.basename(output, '.json'))));
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const frontend = new URL(fixture.frontend_url);
const api = new URL(fixture.api_url);
const envFile = path.join(root, 'frontend/.env.local');
const publicEnv = fs.existsSync(envFile) ? fs.readFileSync(envFile, 'utf8').match(/^NEXT_PUBLIC_API_URL\s*=\s*(.+)$/m)?.[1]?.trim().replace(/^['"]|['"]$/g, '') : null;
const proxyOrigins = new Set(['http://localhost:8000', 'http://127.0.0.1:8000']);
if (fixture.original_api_url || publicEnv) proxyOrigins.add(new URL(fixture.original_api_url || publicEnv).origin);
for (const url of [frontend, api]) {
  assert.ok(['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname), 'Smoke services must use loopback URLs');
  assert.equal(url.protocol, 'http:');
}
assert.notEqual(api.port || '80', '8000', 'The smoke API must be distinct from the user API');
const { chromium } = require(path.join(root, 'Software/browser-smoke-tools/node_modules/playwright'));
const results = {
  timestamp: new Date().toISOString(), frontend_url: frontend.origin, smoke_api_url: api.origin,
  playwright_version: require(path.join(root, 'Software/browser-smoke-tools/node_modules/playwright/package.json')).version,
  scope: 'Real Chromium against the existing frontend; original API requests proxied exclusively to isolated fixtures. Proxy supplies CORS headers. OTP request delivery is intercepted; OTP verification uses the seeded smoke account.',
  proxied_api_origins: [...proxyOrigins],
  checks: [], runtime_errors: [], unexpected_requests: [], screenshots: [],
};
fs.mkdirSync(screenshots, { recursive: true });

async function options() {
  const source = fs.readFileSync(path.join(root, 'frontend/components/survey-options.js'));
  return import(`data:text/javascript;base64,${source.toString('base64')}`);
}
async function check(name, action, page) {
  const start = Date.now();
  try {
    const detail = await action();
    results.checks.push({ name, status: 'passed', detail, duration_ms: Date.now() - start });
  } catch (error) {
    results.checks.push({ name, status: 'failed', error: error.message, duration_ms: Date.now() - start });
  }
  if (page && !page.isClosed()) {
    const filename = `${String(results.checks.length).padStart(2, '0')}-${name.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}.png`;
    await page.screenshot({ path: path.join(screenshots, filename), fullPage: true }).catch(() => {});
    results.screenshots.push(path.relative(root, path.join(screenshots, filename)).replaceAll('\\', '/'));
  }
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(results, null, 2) + '\n');
}
async function snapshot(context, token, session) {
  const response = await context.request.get(`${api.origin}/survey/sessions/${session}/conversation`, { headers: { Authorization: `Bearer ${token}` } });
  assert.equal(response.status(), 200, 'Conversation snapshot must succeed');
  return response.json();
}
async function waitAction(page, route, action, method = null) {
  const [response] = await Promise.all([
    page.waitForResponse(response => response.url().includes(route) && (!method || response.request().method() === method)),
    action(),
  ]);
  assert.ok(response.ok(), `${method || 'API'} ${route}: HTTP ${response.status()}`);
  return { body: await response.json(), request: response.request().postDataJSON() };
}
async function newContext(browser, user, session = null, language = 'en') {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, serviceWorkers: 'block' });
  await context.route('**/*', async route => {
    const request = route.request();
    const original = new URL(request.url());
    if (original.protocol === 'data:' || original.protocol === 'about:') return route.continue();
    // No OTP email/SMS request is ever forwarded. The optional login fixture
    // supplies an already-seeded OTP, whose verification is still real.
    if (proxyOrigins.has(original.origin)) {
      const corsHeaders = {
        'access-control-allow-origin': frontend.origin,
        'access-control-allow-methods': 'GET, POST, PATCH, DELETE, OPTIONS',
        'access-control-allow-headers': 'authorization, content-type',
      };
      if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: corsHeaders });
      if (original.pathname === '/auth/request-otp') {
        if (!fixture.login?.otp) return route.abort('blockedbyclient');
        return route.fulfill({ status: 200, headers: corsHeaders, contentType: 'application/json', body: JSON.stringify({ message: 'Smoke OTP fixture (no delivery).', dev_otp: fixture.login.otp }) });
      }
      if (original.pathname.includes('sms') || original.pathname.includes('request-otp')) return route.abort('blockedbyclient');
      const target = `${api.origin}${original.pathname}${original.search}`;
      const response = await route.fetch({ url: target, maxRedirects: 0 });
      return route.fulfill({ response, headers: { ...response.headers(), ...corsHeaders } });
    }
    if (original.origin === frontend.origin) return route.continue();
    // Fonts and other optional third-party assets are excluded from this run.
    return route.abort('blockedbyclient');
  });
  if (user) {
    await context.addInitScript(({ token, session, language, frontendOrigin }) => {
      if (location.origin !== frontendOrigin) return;
      localStorage.setItem('lfs_token', token);
      if (!localStorage.getItem('lfs_lang')) localStorage.setItem('lfs_lang', language);
      if (session != null && !localStorage.getItem('lfs_active_session')) localStorage.setItem('lfs_active_session', String(session));
    }, { token: user.token, session, language, frontendOrigin: frontend.origin });
  }
  context.on('page', page => {
    page.on('pageerror', error => results.runtime_errors.push({ path: new URL(page.url()).pathname, error: error.message }));
    page.on('request', request => {
      const url = new URL(request.url());
      if (url.port === '8000' && !['localhost', '127.0.0.1'].includes(url.hostname)) results.unexpected_requests.push({ url: url.origin + url.pathname });
    });
  });
  return context;
}
async function mobileCheck(context, page, session) {
  await page.setViewportSize({ width: 390, height: 844 });
  const layout = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll_width: document.documentElement.scrollWidth }));
  assert.ok(layout.scroll_width <= layout.width + 1, 'Page overflows the mobile viewport horizontally');
  const visibleLanguage = page.locator('header').getByRole('button', { name: 'AR', exact: true });
  const alternative = page.getByRole('combobox', { name: 'Select language' });
  assert.ok(await visibleLanguage.isVisible() || await alternative.isVisible(), 'No language selector is visible at a 390 px phone width');
  const before = await snapshot(context, fixture.ordinary.token, session);
  const targetLanguage = before.language === 'ar' ? 'en' : 'ar';
  const changed = await waitAction(page, `/${session}/language`, () => alternative.isVisible().then(visible => visible
    ? alternative.selectOption(targetLanguage)
    : page.locator('header').getByRole('button', { name: targetLanguage.toUpperCase(), exact: true }).click()), 'PATCH');
  assert.deepEqual(changed.body.collected_data, before.collected_data);
  assert.equal(changed.body.history.length, before.history.length);
  await page.waitForFunction(() => document.querySelector('textarea') && !document.querySelector('textarea').disabled);
  const changedLayout = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll_width: document.documentElement.scrollWidth }));
  assert.ok(changedLayout.scroll_width <= changedLayout.width + 1, 'Translated mobile header overflows the viewport');
  return { ...layout, changed_language: targetLanguage, translated_layout: changedLayout };
}
async function reportCheck(page) {
  const response = page.waitForResponse(response => response.url().includes(`/${fixture.report_session_id}/report`));
  await page.goto(`${frontend.origin}/report?session=${fixture.report_session_id}`);
  const dataResponse = await response;
  assert.equal(dataResponse.status(), 200);
  const data = await dataResponse.json();
  assert.equal(data.profile.isco_code, fixture.review_code || '2512');
  assert.equal(data.pending_review, false);
  assert.equal(data.human_review_status, 'reviewed');
  await page.getByText('Human Reviewed', { exact: true }).waitFor();
  assert.doesNotMatch(await page.locator('body').innerText(), /ISCO confidence adjustment/i);
  await page.getByRole('combobox', { name: 'Select language' }).selectOption('ar');
  await page.waitForFunction(() => document.querySelector('#__next > div[dir]')?.getAttribute('dir') === 'rtl');
  assert.doesNotMatch(await page.locator('body').innerText(), /ISCO confidence adjustment/i);
  return {
    isco_code: data.profile.isco_code, isco_confidence: data.profile.isco_confidence,
    pending_review: data.pending_review, human_review_status: data.human_review_status,
    semantic_rule_delta: data.semantic_coherence?.confidence_adjustment,
    confidence_adjustment_claim_visible: false,
  };
}

(async () => {
  const browser = await chromium.launch({ executablePath: argument('--browser', 'C:/Program Files/Google/Chrome/Application/chrome.exe'), headless: true });
  results.browser = browser.version();
  const contexts = [];
  try {
    if (process.argv.includes('--parent-rag-only')) {
      assert.ok(fixture.report_session_id, 'A seeded report is required');
      for (const language of ['en', 'ar', 'ur', 'hi', 'tl']) {
        const context = await newContext(browser, fixture.ordinary, null, language); contexts.push(context);
        const page = await context.newPage();
        await check(`parent-document evidence in ${language}`, async () => {
          await page.goto(`${frontend.origin}/report?session=${fixture.report_session_id}`);
          const card = page.locator('section[aria-label]').filter({ hasText: 'intfloat/multilingual-e5-small' });
          await card.waitFor();
          const content = await card.innerText();
          assert.match(content, /7,279\/18,747/);
          assert.match(content, /6,102\/18,747/);
          assert.match(content, /38\.83%/);
          assert.match(content, /32\.55%/);
          assert.match(content, /236\/642/);
          assert.match(content, /209\/642/);
          const direction = await page.locator('#__next > div[dir]').getAttribute('dir');
          assert.equal(direction, ['ar', 'ur'].includes(language) ? 'rtl' : 'ltr');
          await page.setViewportSize({ width: 390, height: 844 });
          const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
          assert.ok(dimensions.scroll <= dimensions.width + 1, 'Mobile page must fit the viewport');
          return { language, direction, exact_matches: '7279/18747', accuracy: '38.83%', mobile_width: dimensions.width };
        }, page);
      }
      await check('no JavaScript runtime errors', async () => {
        assert.deepEqual(results.runtime_errors, []);
        assert.deepEqual(results.unexpected_requests, []);
        return { runtime_errors: 0 };
      });
      return;
    }
    if (process.argv.includes('--report-only')) {
      assert.ok(fixture.report_session_id, 'A seeded reviewed report is required');
      const context = await newContext(browser, fixture.ordinary); contexts.push(context);
      const page = await context.newPage();
      await check('reviewed report preserves code and omits unapplied confidence claim', () => reportCheck(page), page);
      await check('no JavaScript runtime errors', async () => assert.deepEqual(results.runtime_errors, []));
      return;
    }
    if (process.argv.includes('--mobile-only')) {
      const context = await newContext(browser, fixture.ordinary, fixture.ordinary.session_id); contexts.push(context);
      const page = await context.newPage();
      await page.goto(`${frontend.origin}/chat`);
      await page.waitForFunction(() => document.querySelector('textarea') && !document.querySelector('textarea').disabled);
      const session = Number(await page.evaluate(() => localStorage.getItem('lfs_active_session')));
      await check('mobile chat input and language control remain usable', () => mobileCheck(context, page, session), page);
      await check('no JavaScript runtime errors', async () => assert.deepEqual(results.runtime_errors, []));
      return;
    }
    if (fixture.login?.otp) {
      const context = await newContext(browser); contexts.push(context);
      const page = await context.newPage();
      await check('login with seeded OTP and no delivery', async () => {
        await page.goto(frontend.origin);
        await page.locator('#email').fill(fixture.login.email);
        await page.getByRole('button', { name: 'Send verification code', exact: true }).click();
        await page.locator('#otp').fill(fixture.login.otp);
        await page.getByRole('button', { name: 'Sign in', exact: true }).click();
        await page.waitForURL('**/chat');
        await page.locator('header').getByRole('button', { name: 'EN', exact: true }).waitFor();
        assert.ok(await page.evaluate(() => localStorage.getItem('lfs_token')));
        return { otp_delivery: 'intercepted', verification: 'actual isolated API' };
      }, page);
    }
    const context = await newContext(browser, fixture.ordinary, fixture.ordinary.session_id); contexts.push(context);
    const page = await context.newPage();
    let session;
    await check('authenticated chat loads or resumes', async () => {
      await page.goto(`${frontend.origin}/chat`);
      await page.locator('header').getByRole('button', { name: 'HI', exact: true }).waitFor();
      await page.locator('textarea').waitFor();
      await page.waitForFunction(() => document.querySelector('textarea') && !document.querySelector('textarea').disabled);
      session = Number(await page.evaluate(() => localStorage.getItem('lfs_active_session')));
      assert.ok(session > 0);
      const state = await snapshot(context, fixture.ordinary.token, session);
      return { session_id: session, state: state.state, next_field: state.next_field };
    }, page);
    await check('five language controls preserve answers and history', async () => {
      for (const language of ['ar', 'ur', 'tl', 'en', 'hi']) {
        let before = await snapshot(context, fixture.ordinary.token, session);
        if (before.language === language) {
          const alternate = language === 'en' ? 'hi' : 'en';
          await waitAction(page, `/${session}/language`, () => page.locator('header').getByRole('button', { name: alternate.toUpperCase(), exact: true }).click(), 'PATCH');
          await page.waitForFunction(() => document.querySelector('textarea') && !document.querySelector('textarea').disabled);
          before = await snapshot(context, fixture.ordinary.token, session);
        }
        const changed = await waitAction(page, `/${session}/language`, () => page.locator('header').getByRole('button', { name: language.toUpperCase(), exact: true }).click(), 'PATCH');
        assert.deepEqual(changed.request, { language });
        assert.deepEqual(changed.body.collected_data, before.collected_data);
        assert.equal(changed.body.next_field, before.next_field);
        assert.equal(changed.body.history.length, before.history.length);
        const direction = ['ar', 'ur'].includes(language) ? 'rtl' : 'ltr';
        await page.waitForFunction(expected => document.querySelector('#__next > div[dir]')?.getAttribute('dir') === expected, direction);
        await page.waitForFunction(() => document.querySelector('textarea') && !document.querySelector('textarea').disabled);
      }
      return { languages: ['ar', 'ur', 'tl', 'en', 'hi'], answer_requests: 0 };
    }, page);
    await check('Hindi quick answer sends and saves canonical field value', async () => {
      const before = await snapshot(context, fixture.ordinary.token, session);
      const module = await options();
      const option = module.getQuickOptions(before.next_field, 'hi')[0];
      assert.ok(option?.value, 'Fixture must start at a canonical quick-answer question');
      const sent = await waitAction(page, `/${session}/message`, () => page.getByRole('button', { name: option.label, exact: true }).click(), 'POST');
      assert.equal(sent.request.answer_field, before.next_field);
      assert.equal(sent.request.answer_value, option.value);
      assert.equal(sent.request.message, option.label);
      assert.equal(sent.body.collected_data[before.next_field], option.value);
      return { field: before.next_field, canonical_value: option.value, localized_label: option.label };
    }, page);
    await check('reload retains the same interview answers and user turns', async () => {
      const before = await snapshot(context, fixture.ordinary.token, session);
      const userTurns = before.history.filter(turn => turn.role === 'user');
      await waitAction(page, `/${session}/conversation`, () => page.reload(), 'GET');
      const after = await snapshot(context, fixture.ordinary.token, session);
      assert.equal(Number(await page.evaluate(() => localStorage.getItem('lfs_active_session'))), session);
      assert.deepEqual(after.collected_data, before.collected_data);
      assert.deepEqual(after.history.filter(turn => turn.role === 'user'), userTurns);
      return { session_id: session, retained_user_turns: userTurns.length };
    }, page);
    await check('mobile chat input and language control remain usable', () => mobileCheck(context, page, session), page);
    if (fixture.validation_session_id) {
      const correctionContext = await newContext(browser, fixture.ordinary, fixture.validation_session_id, 'en'); contexts.push(correctionContext);
      const correction = await correctionContext.newPage();
      await check('structured education correction reopens field of study', async () => {
        await correction.goto(`${frontend.origin}/chat`);
        await correction.getByRole('button', { name: /No, I'd like to correct something/ }).click();
        await correction.getByRole('button', { name: /^Education / }).click();
        const sent = await waitAction(correction, `/${fixture.validation_session_id}/message`, () => correction.getByRole('button', { name: "Bachelor's degree", exact: true }).click(), 'POST');
        assert.equal(sent.request.correction_field, 'education_level');
        assert.equal(sent.request.correction_value, 'bachelor');
        assert.equal(sent.body.collected_data.education_level, 'bachelor');
        assert.equal(sent.body.next_field, 'field_of_study');
        return { canonical_correction: 'bachelor', next_field: sent.body.next_field };
      }, correction);
    }
    const deniedContext = await newContext(browser, fixture.ordinary); contexts.push(deniedContext);
    const denied = await deniedContext.newPage();
    await check('ordinary respondent is denied supervisor access', async () => {
      const response = denied.waitForResponse(response => response.url().includes('/survey/hitl/queue'));
      await denied.goto(`${frontend.origin}/supervisor_review`);
      assert.equal((await response).status(), 403);
      await denied.getByText('Supervisor review access is required.', { exact: false }).waitFor();
      assert.equal(await denied.getByRole('button', { name: 'Approve', exact: true }).count(), 0);
      return { status: 403 };
    }, denied);
    if (fixture.reviewer && fixture.escalation_id) {
      const reviewerContext = await newContext(browser, fixture.reviewer); contexts.push(reviewerContext);
      const reviewer = await reviewerContext.newPage();
      await check('authorized reviewer sees actual confidence and corrects fixture', async () => {
        const queued = reviewer.waitForResponse(response => response.url().includes('/survey/hitl/queue'));
        await reviewer.goto(`${frontend.origin}/supervisor_review`);
        const queueResponse = await queued;
        assert.equal(queueResponse.status(), 200);
        const item = (await queueResponse.json()).find(item => item.id === fixture.escalation_id);
        assert.ok(item, 'Seeded escalation must be present');
        const row = reviewer.getByRole('row').filter({ has: reviewer.getByRole('cell', { name: String(fixture.escalation_id), exact: true }) });
        await row.getByText(`${(item.ai_confidence * 100).toFixed(1)}%`, { exact: true }).waitFor();
        await row.getByRole('button', { name: 'Correct', exact: true }).click();
        await reviewer.getByPlaceholder('e.g. 2512').fill(fixture.review_code || '2512');
        await waitAction(reviewer, '/survey/hitl/review', () => reviewer.getByRole('button', { name: 'Submit', exact: true }).click(), 'POST');
        await reviewer.getByText('Review submitted.', { exact: true }).waitFor();
        return { escalation_id: fixture.escalation_id, confidence: item.ai_confidence, corrected_code: fixture.review_code || '2512' };
      }, reviewer);
    }
    if (fixture.report_session_id) {
      const reportContext = await newContext(browser, fixture.ordinary); contexts.push(reportContext);
      const report = await reportContext.newPage();
      await check('report shows corrected occupation and human review state', () => reportCheck(report), report);
    }
    const publicContext = await browser.newContext({ serviceWorkers: 'block' }); contexts.push(publicContext);
    const readinessUrl = 'http://localhost:8000/ready';
    await publicContext.route('**/*', route => {
      const target = new URL(route.request().url());
      return target.origin === frontend.origin || target.href === readinessUrl ? route.continue() : route.abort('blockedbyclient');
    });
    const publicPage = await publicContext.newPage();
    await check('actual application browser readiness CORS and DNS', async () => {
      await publicPage.goto(frontend.origin);
      const ready = await publicPage.evaluate(async url => {
        const response = await fetch(url);
        return { status: response.status, response: await response.json() };
      }, readinessUrl);
      assert.equal(ready.status, 200);
      return { scope: 'Unproxied read-only request; no token or respondent writes', url: readinessUrl, ...ready };
    });
    await check('no JavaScript runtime errors', async () => {
      assert.deepEqual(results.runtime_errors, []);
      assert.deepEqual(results.unexpected_requests, []);
      return { runtime_errors: 0 };
    });
  } finally {
    for (const context of contexts) await context.close();
    await browser.close();
    results.passed = results.checks.filter(check => check.status === 'passed').length;
    results.failed = results.checks.filter(check => check.status === 'failed').length;
    fs.writeFileSync(output, JSON.stringify(results, null, 2) + '\n');
    console.log(JSON.stringify({ passed: results.passed, failed: results.failed, output }));
    if (results.failed) process.exitCode = 1;
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
