# start_demo_tunnels.ps1
#
# Brings the public demo (backend + frontend + both tunnels) back up after
# this machine sleeps, restarts, or a terminal session hosting these
# processes closes -- the exact recovery this project needed to do
# manually, repeatedly, on 2026-09-01/02.
#
# Assumes start.bat has already been run at least once (Docker/Postgres/
# Redis/Qdrant/Ollama up, Python deps installed, DB migrated) -- this
# script does NOT redo that, it only manages the backend/frontend
# processes and the two public tunnels on top of them.
#
# What it does, in order:
#   1. Confirms Docker infra (Postgres/Redis/Qdrant) and Ollama are up --
#      stops with a clear message if not, rather than starting a backend
#      that will fail its own health check.
#   2. Starts (or confirms already running) the backend on :8000 and the
#      frontend on :3000.
#   3. Starts (or confirms already running) the frontend's ngrok tunnel on
#      the fixed domain chili-molar-player.ngrok-free.dev -- this URL never
#      changes across runs of this script, so it's safe to keep sharing
#      with your team.
#   4. Starts (or confirms already running) the backend's Cloudflare quick
#      tunnel. This URL DOES change every time it has to be (re)started --
#      the script captures whatever URL it gets, writes it into
#      frontend/.env.local's NEXT_PUBLIC_API_URL and .env's CORS_ORIGINS,
#      then restarts the backend + frontend processes (not the tunnels) so
#      both pick up the new value. If the Cloudflare tunnel was already
#      running from a prior invocation, it's left alone and nothing gets
#      rewritten.
#   5. Prints both final public URLs.
#
# Usage: right-click -> "Run with PowerShell", or from a terminal:
#   powershell -ExecutionPolicy Bypass -File start_demo_tunnels.ps1

$ErrorActionPreference = "Stop"
Set-Location "C:\Multilingual_LFS_Project"

$NgrokExe = "C:\Users\Sivar\AppData\Local\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe"
$CloudflaredExe = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$NgrokDomain = "chili-molar-player.ngrok-free.dev"
$LogDir = "C:\Multilingual_LFS_Project\.demo_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Step($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

function Test-Url($url) {
    try {
        Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-ForUrl($url, $label, $maxTries = 40) {
    for ($i = 1; $i -le $maxTries; $i++) {
        if (Test-Url $url) { return $true }
        Write-Host "  Waiting for $label... ($i/$maxTries)"
        Start-Sleep -Seconds 3
    }
    return $false
}

# ---------------------------------------------------------------------------
# Step 1: confirm local infrastructure is already up
# ---------------------------------------------------------------------------
Write-Step "Checking Docker infra + Ollama"

$pgUp = (docker ps --filter "name=lfs_postgres" --filter "status=running" -q)
$redisUp = (docker ps --filter "name=lfs_redis" --filter "status=running" -q)
$qdrantUp = (docker ps --filter "name=lfs_qdrant" --filter "status=running" -q)
if (-not $pgUp -or -not $redisUp -or -not $qdrantUp) {
    Write-Host "ERROR: Postgres/Redis/Qdrant are not all running." -ForegroundColor Red
    Write-Host "Run start.bat first (it brings up the full local stack), then re-run this script." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Postgres, Redis, Qdrant containers running."

if (-not (Test-Url "http://localhost:11434/api/tags")) {
    Write-Host "ERROR: Ollama is not responding on :11434." -ForegroundColor Red
    Write-Host "Start the Ollama app, then re-run this script." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Ollama responding."

# ---------------------------------------------------------------------------
# Step 2: backend + frontend
# ---------------------------------------------------------------------------
Write-Step "Backend (:8000)"
if (Test-Url "http://localhost:8000/health") {
    Write-Host "  [OK] Backend already running."
} else {
    Write-Host "  Starting backend..."
    $env:TF_ENABLE_ONEDNN_OPTS = "0"
    $env:TF_CPP_MIN_LOG_LEVEL = "3"
    Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "backend.main:app", "--reload", "--reload-dir", "backend", "--port", "8000" `
        -WorkingDirectory "C:\Multilingual_LFS_Project" `
        -RedirectStandardOutput "$LogDir\backend.log" -RedirectStandardError "$LogDir\backend.err.log" `
        -WindowStyle Hidden
    if (-not (Wait-ForUrl "http://localhost:8000/health" "backend")) {
        Write-Host "ERROR: Backend did not become healthy. Check $LogDir\backend.err.log" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] Backend healthy."
}

Write-Step "Frontend (:3000)"
if (Test-Url "http://localhost:3000/") {
    Write-Host "  [OK] Frontend already running."
} else {
    Write-Host "  Starting frontend..."
    Start-Process -FilePath "npx" -ArgumentList "next", "dev", "--port", "3000" `
        -WorkingDirectory "C:\Multilingual_LFS_Project\frontend" `
        -RedirectStandardOutput "$LogDir\frontend.log" -RedirectStandardError "$LogDir\frontend.err.log" `
        -WindowStyle Hidden
    if (-not (Wait-ForUrl "http://localhost:3000/" "frontend")) {
        Write-Host "ERROR: Frontend did not come up. Check $LogDir\frontend.err.log" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] Frontend responding."
}

# ---------------------------------------------------------------------------
# Step 3: frontend tunnel (fixed ngrok domain -- never changes)
# ---------------------------------------------------------------------------
Write-Step "Frontend tunnel (ngrok, fixed domain)"
$ngrokRunning = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match [regex]::Escape($NgrokDomain) }
if ($ngrokRunning) {
    Write-Host "  [OK] ngrok tunnel already running -> https://$NgrokDomain"
} else {
    Write-Host "  Starting ngrok tunnel..."
    Start-Process -FilePath $NgrokExe -ArgumentList "http", "--domain=$NgrokDomain", "3000" `
        -RedirectStandardOutput "$LogDir\ngrok.log" -RedirectStandardError "$LogDir\ngrok.err.log" `
        -WindowStyle Hidden
    Start-Sleep -Seconds 6
    if (-not (Test-Url "https://$NgrokDomain/")) {
        Write-Host "  WARNING: ngrok tunnel did not respond yet -- it may still be connecting. Check $LogDir\ngrok.err.log if this persists." -ForegroundColor Yellow
    } else {
        Write-Host "  [OK] https://$NgrokDomain is live."
    }
}

# ---------------------------------------------------------------------------
# Step 4: backend tunnel (Cloudflare quick tunnel -- URL changes if restarted)
# ---------------------------------------------------------------------------
Write-Step "Backend tunnel (Cloudflare quick tunnel)"
$cfRunning = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "cloudflared.*:8000" }
$envLocalPath = "C:\Multilingual_LFS_Project\frontend\.env.local"
$envPath = "C:\Multilingual_LFS_Project\.env"

if ($cfRunning) {
    Write-Host "  [OK] Cloudflare tunnel for backend already running -- leaving it and the env files alone."
} else {
    Write-Host "  Starting a fresh Cloudflare tunnel for the backend (this WILL get a new URL)..."
    $cfLog = "$LogDir\cloudflared_backend.log"
    Remove-Item $cfLog -ErrorAction SilentlyContinue
    Start-Process -FilePath $CloudflaredExe -ArgumentList "tunnel", "--url", "http://localhost:8000" `
        -RedirectStandardOutput $cfLog -RedirectStandardError $cfLog `
        -WindowStyle Hidden
    Start-Sleep -Seconds 8

    $newBackendUrl = $null
    for ($i = 1; $i -le 10; $i++) {
        if (Test-Path $cfLog) {
            $match = Select-String -Path $cfLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
            if ($match) { $newBackendUrl = $match.Matches[0].Value; break }
        }
        Start-Sleep -Seconds 2
    }

    if (-not $newBackendUrl) {
        Write-Host "ERROR: Could not read the new backend tunnel URL from $cfLog" -ForegroundColor Red
        exit 1
    }
    Write-Host "  New backend URL: $newBackendUrl"

    # Update frontend/.env.local
    (Get-Content $envLocalPath) -replace "NEXT_PUBLIC_API_URL=.*", "NEXT_PUBLIC_API_URL=$newBackendUrl" |
        Set-Content $envLocalPath
    Write-Host "  Updated frontend/.env.local"

    # Update .env's CORS_ORIGINS (keep the localhost entries + the ngrok
    # frontend domain, refresh only the trycloudflare.com entry)
    $envLines = Get-Content $envPath
    $corsLine = $envLines | Where-Object { $_ -match "^CORS_ORIGINS=" }
    if ($corsLine) {
        $origins = ($corsLine -replace "^CORS_ORIGINS=", "") -split ","
        $keep = $origins | Where-Object { $_ -notmatch "trycloudflare\.com" }
        $newCors = "CORS_ORIGINS=" + (($keep + $newBackendUrl) -join ",")
        (Get-Content $envPath) -replace "^CORS_ORIGINS=.*", $newCors | Set-Content $envPath
        Write-Host "  Updated .env CORS_ORIGINS"
    }

    Write-Host "  Restarting backend + frontend to pick up the new URL (tunnels are left running)..."
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "uvicorn|multiprocessing.spawn_main" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    $frontendPid = (Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue).OwningProcess
    if ($frontendPid) { Stop-Process -Id $frontendPid -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2

    $env:TF_ENABLE_ONEDNN_OPTS = "0"
    $env:TF_CPP_MIN_LOG_LEVEL = "3"
    Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "backend.main:app", "--reload", "--reload-dir", "backend", "--port", "8000" `
        -WorkingDirectory "C:\Multilingual_LFS_Project" `
        -RedirectStandardOutput "$LogDir\backend.log" -RedirectStandardError "$LogDir\backend.err.log" `
        -WindowStyle Hidden
    Start-Process -FilePath "npx" -ArgumentList "next", "dev", "--port", "3000" `
        -WorkingDirectory "C:\Multilingual_LFS_Project\frontend" `
        -RedirectStandardOutput "$LogDir\frontend.log" -RedirectStandardError "$LogDir\frontend.err.log" `
        -WindowStyle Hidden

    if (-not (Wait-ForUrl "http://localhost:8000/health" "backend (restart)")) {
        Write-Host "ERROR: Backend did not come back up after restart." -ForegroundColor Red
        exit 1
    }
    if (-not (Wait-ForUrl "http://localhost:3000/" "frontend (restart)")) {
        Write-Host "ERROR: Frontend did not come back up after restart." -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] Backend and frontend restarted with the new backend URL."
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Step "Demo is up"
Write-Host ""
Write-Host "  App (share this):  https://$NgrokDomain" -ForegroundColor Green
$currentBackendUrl = (Get-Content $envLocalPath | Where-Object { $_ -match "NEXT_PUBLIC_API_URL=" }) -replace "NEXT_PUBLIC_API_URL=", ""
Write-Host "  Backend API:       $currentBackendUrl" -ForegroundColor Green
Write-Host ""
Write-Host "  Logs: $LogDir"
Write-Host "  Note: first-time visitors to the ngrok link see a one-time 'Visit Site' click-through page -- that's expected." -ForegroundColor Yellow
Write-Host ""
