# Alchemy Bootstrap — One-script setup for Windows 11
# Usage: powershell -ExecutionPolicy Bypass -File bootstrap.ps1
#
# What it does:
#   1. Checks/installs Ollama
#   2. Pulls the UI-TARS model
#   3. Installs Python dependencies
#   4. Creates .env from .env.example
#
# Prerequisites: Windows 11

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "=== Alchemy Bootstrap ===" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Ollama
# ---------------------------------------------------------------------------
Write-Host "[1/4] Checking Ollama..." -ForegroundColor Yellow

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
# 2. Pull model
# ---------------------------------------------------------------------------
Write-Host "[2/4] Pulling UI-TARS model..." -ForegroundColor Yellow

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
# 3. Python dependencies
# ---------------------------------------------------------------------------
Write-Host "[3/4] Installing Python dependencies..." -ForegroundColor Yellow

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
# 4. Create .env
# ---------------------------------------------------------------------------
Write-Host "[4/4] Setting up configuration..." -ForegroundColor Yellow

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
Write-Host "  1. Start Alchemy server:  python -m alchemy.server" -ForegroundColor Gray
Write-Host "  2. Run tests:             pytest tests/" -ForegroundColor Gray
Write-Host ""
