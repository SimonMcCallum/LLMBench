# LLM-Bench Service Setup Script
# Run as Administrator: powershell -ExecutionPolicy Bypass -File setup_service.ps1
#
# This script:
# 1. Creates a Windows Task Scheduler job for the daemon
# 2. Enables and starts OpenSSH Server
# 3. Configures SSH with key auth and Git Bash as default shell
# 4. Creates the shared model cache symlink
# 5. Provides Tailscale setup instructions

param(
    [switch]$DaemonOnly,
    [switch]$SSHOnly,
    [switch]$SymlinkOnly,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot
if (-not $RepoDir) { $RepoDir = "D:\git\LLM-Bench" }
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) { $PythonExe = "python" }

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "LLM-Bench Service Setup" -ForegroundColor Cyan
Write-Host "  Repo: $RepoDir" -ForegroundColor Cyan
Write-Host "  Python: $PythonExe" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ============================================================
# UNINSTALL
# ============================================================
if ($Uninstall) {
    Write-Host "`nUninstalling..." -ForegroundColor Yellow
    try {
        Unregister-ScheduledTask -TaskName "LLM-Bench-Daemon" -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "  Removed scheduled task" -ForegroundColor Green
    } catch { Write-Host "  No scheduled task to remove" }
    exit 0
}

# ============================================================
# 1. WINDOWS TASK SCHEDULER — DAEMON
# ============================================================
if (-not $SSHOnly -and -not $SymlinkOnly) {
    Write-Host "`n[1/4] Setting up Windows Task Scheduler daemon..." -ForegroundColor Yellow

    # Remove existing task if present
    Unregister-ScheduledTask -TaskName "LLM-Bench-Daemon" -Confirm:$false -ErrorAction SilentlyContinue

    $DaemonScript = Join-Path $RepoDir "service\daemon.py"
    $Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$DaemonScript`"" -WorkingDirectory $RepoDir

    # Trigger: at startup + repeat every 5 minutes for ~25 years
    $TriggerBoot = New-ScheduledTaskTrigger -AtStartup
    $TriggerRepeat = New-ScheduledTaskTrigger -Once -At (Get-Date) `
        -RepetitionInterval (New-TimeSpan -Minutes 5) `
        -RepetitionDuration (New-TimeSpan -Days 9000)

    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask `
        -TaskName "LLM-Bench-Daemon" `
        -Action $Action `
        -Trigger $TriggerBoot, $TriggerRepeat `
        -Settings $Settings `
        -RunLevel Highest `
        -Description "LLM-Bench daemon: polls git for benchmark tasks every 5 minutes"

    Write-Host "  Task 'LLM-Bench-Daemon' registered successfully" -ForegroundColor Green
    Write-Host "  Trigger: Every 5 minutes + at startup" -ForegroundColor Gray
}

# ============================================================
# 2. OPENSSH SERVER
# ============================================================
if (-not $DaemonOnly -and -not $SymlinkOnly) {
    Write-Host "`n[2/4] Configuring OpenSSH Server..." -ForegroundColor Yellow

    # Check if sshd service exists
    $sshd = Get-Service -Name sshd -ErrorAction SilentlyContinue
    if (-not $sshd) {
        Write-Host "  OpenSSH Server not installed. Installing..." -ForegroundColor Yellow
        Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    }

    # Start and enable
    Start-Service sshd
    Set-Service -Name sshd -StartupType Automatic
    Write-Host "  sshd service started and set to automatic" -ForegroundColor Green

    # Start ssh-agent
    Start-Service ssh-agent -ErrorAction SilentlyContinue
    Set-Service -Name ssh-agent -StartupType Automatic -ErrorAction SilentlyContinue

    # Firewall rule
    $rule = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
    if (-not $rule) {
        New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" `
            -DisplayName "OpenSSH Server (sshd)" `
            -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow
        Write-Host "  Firewall rule created for port 22" -ForegroundColor Green
    } else {
        Write-Host "  Firewall rule already exists" -ForegroundColor Gray
    }

    # Set default shell to Git Bash
    $GitBash = "C:\Program Files\Git\bin\bash.exe"
    if (Test-Path $GitBash) {
        New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
            -Value $GitBash -PropertyType String -Force | Out-Null
        Write-Host "  Default SSH shell set to Git Bash" -ForegroundColor Green
    } else {
        Write-Host "  Git Bash not found at $GitBash, using default shell" -ForegroundColor Yellow
    }

    # Configure sshd_config
    $sshdConfig = "C:\ProgramData\ssh\sshd_config"
    if (Test-Path $sshdConfig) {
        $config = Get-Content $sshdConfig -Raw
        if ($config -notmatch "PubkeyAuthentication yes") {
            Write-Host "  NOTE: Enable PubkeyAuthentication in $sshdConfig" -ForegroundColor Yellow
        }
    }

    # Ensure authorized_keys file exists
    $authKeys = "$env:USERPROFILE\.ssh\authorized_keys"
    if (-not (Test-Path $authKeys)) {
        New-Item -ItemType File -Path $authKeys -Force | Out-Null
        Write-Host "  Created $authKeys (add your public key here)" -ForegroundColor Yellow
    }

    Write-Host "  SSH setup complete. Test with: ssh $env:USERNAME@localhost" -ForegroundColor Green
}

# ============================================================
# 3. SHARED MODEL CACHE SYMLINK
# ============================================================
if (-not $DaemonOnly -and -not $SSHOnly) {
    Write-Host "`n[3/4] Setting up shared model cache symlink..." -ForegroundColor Yellow

    $SymlinkPath = Join-Path $RepoDir "data\models"
    $TargetPath = "D:\git\NNCONFIDENCE\data\models"

    if (Test-Path $TargetPath) {
        # Remove existing directory/symlink
        if (Test-Path $SymlinkPath) {
            Remove-Item $SymlinkPath -Recurse -Force -ErrorAction SilentlyContinue
        }
        try {
            New-Item -ItemType SymbolicLink -Path $SymlinkPath -Target $TargetPath | Out-Null
            Write-Host "  Symlink created: $SymlinkPath -> $TargetPath" -ForegroundColor Green
        } catch {
            Write-Host "  Symlink failed (needs admin). Using config path reference instead." -ForegroundColor Yellow
            New-Item -ItemType Directory -Path $SymlinkPath -Force | Out-Null
        }
    } else {
        Write-Host "  NNCONFIDENCE model cache not found at $TargetPath" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $SymlinkPath -Force | Out-Null
    }
}

# ============================================================
# 4. TAILSCALE INSTRUCTIONS
# ============================================================
Write-Host "`n[4/4] Tailscale Setup (for remote access)" -ForegroundColor Yellow
Write-Host @"

  To access this machine from outside the firewall:

  1. Download Tailscale: https://tailscale.com/download/windows
  2. Install and sign in on THIS machine (aroma)
  3. Install and sign in on your REMOTE machine
  4. Both machines get stable 100.x.y.z IPs
  5. SSH from remote: ssh $env:USERNAME@<tailscale-ip>

  No port forwarding or firewall changes needed.

"@ -ForegroundColor Gray

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
