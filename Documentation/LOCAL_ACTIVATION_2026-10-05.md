# Local activation and browser verification — 5 October 2026

This continues the [code-fix pass](CODE_FIXES_2026-10-05.md) by applying the changes to the existing local app and checking actual browser workflows.

## Applied configuration and migration

- Upgraded the local `lfs_db` from `c7e2a48f9d31` to **`d841b2079c65`**. Every model table/column matches. All pre-existing table row counts were preserved; no respondent answers or decisions were edited.
- Configured the requested existing account, active database **user ID 5**, in the server-only `HITL_REVIEWER_USER_IDS` setting. Its real supervisor queue request returned HTTP 200. This grant covers all active respondents, as documented in the earlier fix report.
- Started the actual backend at **http://127.0.0.1:8000** and the production frontend at **http://127.0.0.1:3000**. Database, Redis, Qdrant, and Ollama checks succeeded; classifier and correction-schema warmup completed.
- Rebuilt the local frontend with process-only `NEXT_PUBLIC_API_URL=http://localhost:8000`. The user's `.env.local` tunnel setting remains available for future tunnel builds. The compiled local app uses the local API.

[Activation evidence](LOCAL_ACTIVATION_2026-10-05_RESULTS.json) contains migration, count, readiness, and access-check results without credentials or respondent content.

## Fixes discovered during activation

The readiness endpoint previously accepted traffic while Redis was unavailable. `/ready` now returns HTTP 503 if PostgreSQL or Redis fails, closes probe resources, and bounds Redis connect/read waits. Compose uses `/ready` to gate frontend startup. Diagnostic Qdrant checks respect configured host/port. README startup instructions now use Alembic directly.

At a **390 × 844** phone viewport, both language controls were hidden. The chat header now uses the existing language selector on phones and keeps desktop pills. Repeated Arabic language changes preserve answers and transcript length, and the header fits without horizontal overflow. [Before](ACTIVATION_2026-10-05_BROWSER_MOBILE_BEFORE.json) and [after](ACTIVATION_2026-10-05_BROWSER_MOBILE_AFTER.json) results preserve the reproduced failure and successful check.

The report also displayed an ISCO confidence adjustment from heuristic metadata even though the application never applies it to the stored score. That display was removed; the real confidence, human decision, and semantic warnings remain visible.

The reusable browser-fixture launcher now verifies that both connection URLs match the inspected task-owned containers before environment setup or migration. It rejects other hosts, ports, database names/usernames, Redis authentication, query overrides, and incomplete service state using explicit exceptions.

## Validation

- **2,779 backend/evaluation tests passed**, one slow test deselected, two known test-only warnings, in **174.92 seconds**. [JUnit results](LOCAL_ACTIVATION_2026-10-05_TEST_RESULTS.xml).
- **13 frontend tests passed**, and the production build generated seven routes.
- The real local API's CORS preflight accepted the frontend origin and authorization/content-type headers.
- A read-only live semantic-classifier probe for `Nurse` completed against the actual initialized classifier. It requested no LLM and wrote no survey rows. This proves the inference path executes; a generic job label and one prediction do not establish accuracy.
- **12 actual Chromium browser checks passed**, with no JavaScript runtime errors. [Browser results](ACTIVATION_2026-10-05_BROWSER_RESULTS.json) cover seeded-OTP login, five languages, canonical Hindi answers, reload/resume, phone controls, conditional education correction, ordinary-user denial, supervisor confidence/correction, report revision/human status, and actual browser readiness.
- **Two final report browser checks passed** after the presentation fix. The reviewed code and stored confidence remain correct, and the unapplied adjustment claim is absent in English and Arabic. [Final report results](ACTIVATION_2026-10-05_BROWSER_REPORT_AFTER.json).
- **24 additional fixture-isolation tests passed**, with one known test-plugin warning, in a separate focused run after the launcher guard was added. They include wrong-target rejection before migrations and private-URL error redaction. These tests were added after the 2,779-test full run. [Focused JUnit results](BROWSER_SMOKE_ISOLATION_2026-10-05_TEST_RESULTS.xml).

Browser tests use the production frontend and actual application routes backed by separately created PostgreSQL/Redis containers. Accounts, JWTs, answers, OTP, and classification outputs are synthetic fixtures. OTP request delivery is intercepted; seeded OTP verification exercises the real authentication route. AI generation/classification uses controlled fixtures, so these checks establish application behavior rather than live LLM accuracy. Interview/review requests are proxied exclusively to the isolated API on port 8001; its proxy supplies CORS headers. A separate unproxied public readiness request checks actual production browser CORS/DNS.

## Repeating browser checks

Requires the installed Python dependencies, Docker's local `postgres:15`/`redis:7` images, Node, and Chrome or Edge. Install the browser driver outside application dependencies:

```powershell
npm install --prefix Software/browser-smoke-tools --no-audit --no-fund --package-lock=false playwright
py -3.11 scripts/manage_browser_smoke_services.py start
py -3.11 scripts/run_browser_smoke_backend.py
```

In a separate terminal, with the local frontend running:

```powershell
node scripts/browser_smoke.cjs --fixture Software/browser_smoke/fixtures.json
```

Afterwards, stop the smoke backend and remove only its task-owned containers:

```powershell
py -3.11 scripts/manage_browser_smoke_services.py stop
```

Private fixture tokens/connection strings, process records, and screenshots stay in ignored `Software/`. The scripts check container identity and ownership before writes or cleanup. Repeating the full suite requires fresh fixtures because OTPs and review decisions are consumed.

The completed run's fixture backend was stopped and its task-owned PostgreSQL/Redis containers were removed. Final checks confirmed that the actual application remains ready and all original database row counts remain unchanged.

## Runtime notes

The backend/frontend are started as hidden local processes; existing infrastructure containers remain running. They use the intended `.env` and existing database. Process records and runtime logs are under `Software/local_activation/`. No `start.bat` listener-wide termination, registry changes, or public tunnel deployment was used.

For subsequent local frontend rebuilds:

```powershell
$env:NEXT_PUBLIC_API_URL = 'http://localhost:8000'
npm --prefix frontend run build
```

Historical encoder provenance and the thesis's missing empirical evidence remain as described in the earlier review. These local activation checks do not establish human usability, expert classification accuracy, or production load behavior.
