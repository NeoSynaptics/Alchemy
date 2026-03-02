# Alchemy — Install WSL2 + Ubuntu
# Run as Administrator: powershell -ExecutionPolicy Bypass -File scripts\install_wsl.ps1

Write-Host "=== Alchemy: WSL2 + Ubuntu Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Run this script as Administrator." -ForegroundColor Red
    Write-Host "  Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    exit 1
}

# Install Ubuntu (this also enables WSL2 if needed)
Write-Host "[1/3] Installing Ubuntu on WSL2..." -ForegroundColor Green
wsl --install -d Ubuntu --no-launch
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: WSL install failed. You may need to restart and run again." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/3] Launching Ubuntu for first-time setup..." -ForegroundColor Green
Write-Host "  You'll be asked to create a UNIX username and password." -ForegroundColor Yellow
Write-Host "  Remember these — you'll need them for sudo." -ForegroundColor Yellow
Write-Host ""
wsl -d Ubuntu

Write-Host ""
Write-Host "[3/3] Verifying..." -ForegroundColor Green
$check = wsl -d Ubuntu -- echo ok 2>&1
if ($check -match "ok") {
    Write-Host "  WSL2 Ubuntu: OK" -ForegroundColor Green
    Write-Host ""
    Write-Host "=== Next step ===" -ForegroundColor Cyan
    Write-Host "  Run in this terminal:" -ForegroundColor Yellow
    Write-Host '  wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/info/GitHub/Alchemy && bash wsl/setup.sh"' -ForegroundColor White
} else {
    Write-Host "  WSL2 Ubuntu: NOT READY (may need restart)" -ForegroundColor Red
    Write-Host "  After restart, run: wsl -d Ubuntu" -ForegroundColor Yellow
}
