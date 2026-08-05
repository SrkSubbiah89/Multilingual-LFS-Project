@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
title LFS Project Startup
cd /d C:\Multilingual_LFS_Project

echo ============================================
echo  Step 1: Starting Docker Desktop
echo ============================================

docker info >nul 2>&1
if %errorlevel%==0 (
    echo  [OK] Docker is already running.
    goto DOCKER_READY
)

echo  Launching Docker Desktop (minimized to system tray)...
REM -minimized flag starts Docker Desktop without showing the UI window
start /min "" "C:\Program Files\Docker\Docker\Docker Desktop.exe" -minimized

set /a dtries=0
:DOCKER_WAIT
set /a dtries+=1
echo  Waiting for Docker daemon... (!dtries!/24)
timeout /t 5 /nobreak >nul
docker info >nul 2>&1
if %errorlevel%==0 goto DOCKER_READY
if !dtries! geq 24 (
    echo  ERROR: Docker did not start after 2 minutes.
    echo  Please start Docker Desktop manually and re-run.
    pause & exit /b 1
)
goto DOCKER_WAIT

:DOCKER_READY
echo  [OK] Docker daemon is running.

REM Ensure Docker Desktop auto-starts with Windows (system tray, no UI)
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "Docker Desktop" /t REG_SZ /d "\"C:\Program Files\Docker\Docker\Docker Desktop.exe\" -minimized" /f >nul 2>&1

echo.
echo ============================================
echo  Step 2: Starting PostgreSQL container
echo ============================================

docker compose -f docker/docker-compose.yml up -d postgres

set /a ctries=0
:PG_WAIT
set /a ctries+=1
echo  Waiting for PostgreSQL... (!ctries!/24)
timeout /t 5 /nobreak >nul
docker exec lfs_postgres pg_isready -U lfs_user -d lfs_db >nul 2>&1
if %errorlevel%==0 goto PG_READY
if !ctries! geq 24 (
    echo  ERROR: PostgreSQL did not become ready after 2 minutes.
    pause & exit /b 1
)
goto PG_WAIT

:PG_READY
echo  [OK] PostgreSQL is ready.

echo.
echo ============================================
echo  Step 3: Installing Python dependencies
echo ============================================

python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo  WARNING: pip install had errors. Some packages may be missing.
)
echo  [OK] Python dependencies up to date.

echo.
echo ============================================
echo  Step 3b: Running database migrations
echo ============================================

python -m alembic upgrade head
if %errorlevel% neq 0 (
    echo  ERROR: Database migration failed.
    pause & exit /b 1
)
echo  [OK] Database schema is up to date.

echo.
echo ============================================
echo  Step 4: Starting Redis container
echo ============================================

docker compose -f docker/docker-compose.yml up -d redis

set /a ctries=0
:REDIS_WAIT
set /a ctries+=1
echo  Waiting for Redis... (!ctries!/12)
timeout /t 3 /nobreak >nul
docker exec lfs_redis redis-cli ping >nul 2>&1
if %errorlevel%==0 goto REDIS_READY
if !ctries! geq 12 (
    echo  ERROR: Redis did not become ready after 36 seconds.
    pause & exit /b 1
)
goto REDIS_WAIT

:REDIS_READY
echo  [OK] Redis is ready.

echo.
echo ============================================
echo  Step 5: Starting Qdrant container
echo ============================================

docker compose -f docker/docker-compose.yml up -d qdrant

set /a ctries=0
:QDRANT_WAIT
set /a ctries+=1
echo  Waiting for Qdrant... (!ctries!/12)
timeout /t 5 /nobreak >nul
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri http://localhost:6333/healthz -UseBasicParsing -TimeoutSec 2 -EA Stop;exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 goto QDRANT_READY
if !ctries! geq 12 (
    echo  WARNING: Qdrant did not respond after 60 s. Continuing...
    goto QDRANT_READY
)
goto QDRANT_WAIT

:QDRANT_READY
echo  [OK] Qdrant is ready.

echo.
echo ============================================
echo  Step 6: Starting Ollama (local LLM)
echo ============================================

REM Check if Ollama is already serving on port 11434
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri http://localhost:11434/api/tags -UseBasicParsing -TimeoutSec 2 -EA Stop;exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 (
    echo  [OK] Ollama is already running.
    goto OLLAMA_READY
)

REM Try to start Ollama (must be installed: https://ollama.com)
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo  WARNING: Ollama not found on PATH. LLM will fall back to Claude API.
    echo           Install Ollama from https://ollama.com and run: ollama pull llama3.2:1b
    goto OLLAMA_READY
)

echo  Starting Ollama server in background...
start /min "Ollama" cmd /c "ollama serve"

set /a otries=0
:OLLAMA_WAIT
set /a otries+=1
echo  Waiting for Ollama... (!otries!/20)
timeout /t 3 /nobreak >nul
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri http://localhost:11434/api/tags -UseBasicParsing -TimeoutSec 2 -EA Stop;exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 goto OLLAMA_UP
if !otries! geq 20 (
    echo  WARNING: Ollama did not start after 60 s. LLM will fall back to Claude API.
    goto OLLAMA_READY
)
goto OLLAMA_WAIT

:OLLAMA_UP
REM Ensure the required model is pulled (no-op if already present)
echo  Pulling llama3.2:1b model (skipped if already cached)...
ollama pull llama3.2:1b

:OLLAMA_READY
echo  [OK] Ollama step complete.

echo.
echo ============================================
echo  Step 7: Loading hierarchical ISCO data
echo           into Qdrant (first run only)
echo ============================================

REM Check if the isco08_major_groups collection already exists in Qdrant.
REM If it does, load_full_isco is a no-op (--recreate not passed).
REM If Qdrant is unreachable the script exits with non-zero; warn and continue.
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri http://localhost:6333/collections/isco08_major_groups -UseBasicParsing -TimeoutSec 3 -EA Stop;exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 (
    echo  [OK] Hierarchical ISCO collections already loaded. Skipping.
    goto ISCO_READY
)

echo  Loading full ISCO-08 hierarchy into Qdrant (this may take a few minutes on first run)...
python -m backend.rag.load_full_isco
if %errorlevel% neq 0 (
    echo  WARNING: ISCO data load failed. Classifier will use flat fallback.
) else (
    echo  [OK] ISCO hierarchical collections loaded.
)

:ISCO_READY

echo.
echo ============================================
echo  Step 8: Stopping Docker app containers
echo  (backend + frontend run natively instead)
echo ============================================

docker compose -f docker/docker-compose.yml stop backend frontend >nul 2>&1
echo  [OK] Docker backend/frontend stopped.

echo.
echo ============================================
echo  Step 9: Killing any stale app processes
echo ============================================

for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " ^| findstr LISTENING 2^>nul') do (
    echo  Killing PID %%p on port 8000...
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":3000 " ^| findstr LISTENING 2^>nul') do (
    echo  Killing PID %%p on port 3000...
    taskkill /PID %%p /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo.
echo ============================================
echo  Step 10: Starting LFS Backend (port 8000)
echo ============================================

start "LFS Backend" cmd /k "cd /d C:\Multilingual_LFS_Project && set TF_ENABLE_ONEDNN_OPTS=0 && set TF_CPP_MIN_LOG_LEVEL=3 && python -m uvicorn backend.main:app --reload --reload-dir backend --port 8000"

set /a btries=0
:BACKEND_WAIT
set /a btries+=1
echo  Waiting for backend + database... (!btries!/30)
timeout /t 3 /nobreak >nul
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri http://localhost:8000/ready -UseBasicParsing -TimeoutSec 5 -EA Stop;exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 goto BACKEND_READY
if !btries! geq 30 (
    echo  WARNING: Backend did not respond after 90 s. Starting frontend anyway...
    goto BACKEND_READY
)
goto BACKEND_WAIT

:BACKEND_READY
echo  [OK] Backend is ready.

echo.
echo ============================================
echo  Step 11: Starting LFS Frontend (port 3000)
echo ============================================

start "LFS Frontend" cmd /k "cd /d C:\Multilingual_LFS_Project\frontend && npm install --silent && npx next dev --port 3000"

echo.
echo ============================================
echo  ALL SERVICES ARE UP
echo.
echo   Frontend  : http://localhost:3000
echo   Backend   : http://localhost:8000
echo   API Docs  : http://localhost:8000/docs
echo   Ollama    : http://localhost:11434
echo   Qdrant UI : http://localhost:6333/dashboard
echo.
echo  IMPORTANT: Docker Desktop runs in the system
echo  tray (bottom-right). Do NOT close it or the
echo  database / vector store will stop working.
echo.
echo  Supervisor dashboard (HITL review):
echo    http://localhost:3000/supervisor_review
echo ============================================
echo.
pause
