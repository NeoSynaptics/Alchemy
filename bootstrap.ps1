# Alchemy Bootstrap — One-script setup for Windows 11
# Usage: powershell -ExecutionPolicy Bypass -File bootstrap.ps1
#
# What it does:
#   1. Checks/enables WSL2 + installs Ubuntu
#   2. Runs wsl/setup.sh inside Ubuntu (Xvfb, Fluxbox, x11vnc, etc.)
#   3. Checks/installs Ollama
#   4. Pulls the UI-TARS model
#   5. Installs Python dependencies
#   6. Creates .env from .env.example
#
# Prerequisites: Windows 11, admin rights for WSL install

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "=== Alchemy Bootstrap ===" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# 1. WSL2
# ---------------------------------------------------------------------------
Write-Host "[1/6] Checking WSL2..." -ForegroundColor Yellow

$wslAvailable = $false
try {
    $wslOutput = wsl --status 2>&1
    if ($LASTEXITCODE -eq 0) { $wslAvailable = $true }
} catch {}

if (-not $wslAvailable) {
    Write-Host "  WSL2 not found. Installing (requires admin + reboot)..."
    wsl --install --no-distribution
    Write-Host "  WSL2 installed. Please REBOOT and re-run this script." -ForegroundColor Red
    exit 1
}
Write-Host "  WSL2 is available." -ForegroundColor Green

# Check Ubuntu distro
$distros = wsl --list --quiet 2>&1
if ($distros -notmatch "Ubuntu") {
    Write-Host "  Ubuntu not found. Installing..."
    wsl --install -d Ubuntu --no-launch
    Write-Host "  Ubuntu installed. Launch it once to set up your user, then re-run." -ForegroundColor Red
    exit 1
}
Write-Host "  Ubuntu distro found." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. WSL packages (shadow desktop dependencies)
# ---------------------------------------------------------------------------
Write-Host "[2/6] Installing shadow desktop packages in WSL..." -ForegroundColor Yellow

$setupScript = "$RepoRoot/wsl/setup.sh"
$wslSetupPath = $setupScript -replace "C:", "/mnt/c" -replace "\\", "/"
wsl -d Ubuntu -- bash -c "tr -d '\r' < '$wslSetupPath' | bash"

if ($LASTEXITCODE -ne 0) {
    Write-Host "  WSL setup failed. Check output above." -ForegroundColor Red
    exit 1
}
Write-Host "  Shadow desktop packages installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 3. Ollama
# ---------------------------------------------------------------------------
Write-Host "[3/6] Checking Ollama..." -ForegroundColor Yellow

$ollamaExists = $false
try {
    $ollamaVersion = ollama --version 2>&1
    if ($LASTEXITCODE -eq 0) { $ollamaExists = $true }
} catch {}

if (-not $ollamaExists) {
    Write-Host "  Ollama not found. Installing..."
    $installerUrl = "https://ollama.com/download/OllamaSetup.exe"
    $installerPath = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
    Start-Process -FilePath $installerPath -Wait
    Write-Host "  Ollama installed. Restart your terminal and re-run if 'ollama' is not in PATH." -ForegroundColor Yellow
} else {
    Write-Host "  Ollama found: $ollamaVersion" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 4. Pull model
# ---------------------------------------------------------------------------
Write-Host "[4/6] Pulling UI-TARS model..." -ForegroundColor Yellow

# Read model from .env.example if no .env exists
$modelName = "rashakol/UI-TARS-72B-DPO"
$envFile = "$RepoRoot/.env"
if (Test-Path $envFile) {
    $envModel = Select-String -Path $envFile -Pattern "^OLLAMA_CPU_MODEL=(.+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    if ($envModel) { $modelName = $envModel }
}

Write-Host "  Model: $modelName"
Write-Host "  This may take a while for large models (72B = ~47GB)..."
ollama pull $modelName

if ($LASTEXITCODE -ne 0) {
    Write-Host "  Model pull failed. You can retry manually: ollama pull $modelName" -ForegroundColor Red
} else {
    Write-Host "  Model ready." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 5. Python dependencies
# ---------------------------------------------------------------------------
Write-Host "[5/6] Installing Python dependencies..." -ForegroundColor Yellow

Push-Location $RepoRoot
try {
    # Prefer uv if available, fall back to pip
    $uvExists = $false
    try { uv --version 2>&1 | Out-Null; $uvExists = $true } catch {}

    if ($uvExists) {
        Write-Host "  Using uv..."
        uv pip install -e ".[dev]"
    } else {
        Write-Host "  Using pip..."
        pip install -e ".[dev]"
    }
    Write-Host "  Dependencies installed." -ForegroundColor Green
} finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
# 6. Create .env
# ---------------------------------------------------------------------------
Write-Host "[6/6] Setting up configuration..." -ForegroundColor Yellow

if (-not (Test-Path "$RepoRoot/.env")) {
    Copy-Item "$RepoRoot/.env.example" "$RepoRoot/.env"
    Write-Host "  Created .env from .env.example. Edit to customize." -ForegroundColor Green
} else {
    Write-Host "  .env already exists. Skipping." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Alchemy Bootstrap Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick start:" -ForegroundColor White
Write-Host "  1. Start shadow desktop:  wsl -d Ubuntu -- bash -c 'tr -d ""\r"" < wsl/start_shadow.sh | bash'" -ForegroundColor Gray
Write-Host "  2. Start Alchemy server:  python -m alchemy.server" -ForegroundColor Gray
Write-Host "  3. Run tests:             pytest tests/" -ForegroundColor Gray
Write-Host "  4. View shadow desktop:   http://localhost:6080/vnc.html?autoconnect=true" -ForegroundColor Gray
Write-Host ""
