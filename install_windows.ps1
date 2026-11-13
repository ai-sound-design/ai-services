# Installs (or updates) the backend and the web UI on Windows without Docker: installs
# what is missing with winget (Python 3.12, Node.js, ffmpeg, Ollama), then sets up every
# service natively with windows\backend.py: PostgreSQL 17 with pgvector, a virtualenv per
# service with PyTorch for CUDA, the models, the services as background processes that
# start at every logon, and the web UI (github.com/ai-sound-design/ai-sound-webui).
# Safe to run again at any time; every step only does what is missing.
#
# For users who never open a terminal: the app "AI Sound Design Installer"
# (github.com/ai-sound-design/installer) asks what to install and runs this script.
#
#   powershell -ExecutionPolicy Bypass -File install_windows.ps1
#   powershell -ExecutionPolicy Bypass -File install_windows.ps1 -DataDir "D:\AI Sound Design" -Sessions "D:\Pro Tools Sessions"
#
# The parameters mean what they mean for install.ps1 (-DataDir, -BbcArchive, -OwnSounds,
# -ModelsDir, -NoPrompt, -NoBrowser), plus -Sessions, the folder of Pro Tools sessions the
# web UI opens, -WebUI on|off, and -Only webui, which sets up only the web UI (beside a
# Docker stack, say). Everything decided goes into .env, the file the Docker stack reads
# as well.
#
# Afterwards the services run in the background and start at every logon. Next to the
# installation are "Web UI", "Start backend", "Stop backend", "Backend status" and
# "Setup page" to double-click (also in the Start menu); on the command line:
#   python windows\backend.py status | stop | start | restart | logs <service>

param(
    [string]$DataDir = "",
    [string]$BbcArchive = "",
    [string]$OwnSounds = "",
    [string]$ModelsDir = "",
    [string]$Sessions = "",
    [ValidateSet("", "on", "off")][string]$WebUI = "",
    [ValidateSet("", "webui")][string]$Only = "",
    [switch]$NoPrompt,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
[Console]::OutputEncoding = [Text.Encoding]::UTF8

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "   $text" -ForegroundColor Yellow }
function Fail($text, $link) {
    Write-Host ""
    Write-Host "   $text" -ForegroundColor Red
    if ($link) { Write-Host "   $link" -ForegroundColor White }
    Write-Host ""
    Write-Host "   Run the installer again afterwards." -ForegroundColor Gray
    exit 1
}
function Quote([string[]]$arguments) {
    return (($arguments | ForEach-Object { if ($_ -eq "" -or $_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join " ")
}
function Native-Run($exe, [string[]]$arguments) {
    # The program writes to the console (or the installer app's pipes) itself: through 2>&1 every
    # stderr line would come wrapped in a PowerShell error record, and "Stop" would end the script
    $line = Quote $arguments
    if ($line) { $process = Start-Process -FilePath $exe -ArgumentList $line -NoNewWindow -Wait -PassThru }
    else { $process = Start-Process -FilePath $exe -NoNewWindow -Wait -PassThru }
    return $process.ExitCode
}
function Capture($exe, [string[]]$arguments) {
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $out = & $exe @arguments 2>$null
    $ErrorActionPreference = $eap
    return ($out | Out-String).Trim()
}
function Refresh-Path {
    # A winget install extends the PATH in the registry, not in this process
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user;$env:Path"
}
function Winget-Install($id, $what) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { Fail "$what is not installed, and winget is missing to install it. Install $what by hand, then run this again." }
    Warn "$what is not installed; installing it with winget"
    $code = Native-Run "winget" @("install", "-e", "--id", $id, "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity")
    Refresh-Path
    return $code
}

Step "Windows"
$os = Get-CimInstance Win32_OperatingSystem
if ([int]$os.BuildNumber -lt 19041) { Fail "Windows 10 version 2004 (build 19041) or newer is needed; this is build $($os.BuildNumber)." }
$memory = [math]::Round($os.TotalVisibleMemorySize / 1MB)
Ok "$($os.Caption) build $($os.BuildNumber), $memory GB memory"
if ($memory -lt 32) { Warn "MMAudio and the vision-language model together want about 24 GB; 32 GB or more is recommended." }
Refresh-Path

if ($Only -ne "webui") {
    Step "NVIDIA driver"
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "No NVIDIA driver found (nvidia-smi is missing): the models run on an NVIDIA GPU. Install the current driver for your GPU:" "https://www.nvidia.com/Download/index.aspx" }
    $gpu = Capture "nvidia-smi" @("--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
    if (-not $gpu) { Fail "nvidia-smi does not answer; reinstall the NVIDIA driver:" "https://www.nvidia.com/Download/index.aspx" }
    Ok "$gpu"
}

Step "Python 3.12"
function Find-Python {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $path = Capture "py" @("-3.12", "-c", "import sys; print(sys.executable)")
        if ($path -and (Test-Path -LiteralPath $path)) { return $path }
    }
    foreach ($candidate in (Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"), "C:\Python312\python.exe", (Join-Path $env:ProgramFiles "Python312\python.exe")) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}
$python = Find-Python
if (-not $python) {
    # Python's winget package installs for this account, with the py launcher
    $null = Winget-Install "Python.Python.3.12" "Python 3.12"
    $python = Find-Python
    if (-not $python) { Fail "Python 3.12 could not be installed with winget. Install it by hand (tick 'py launcher'), then run this again:" "https://www.python.org/downloads/windows/" }
}
Ok "$python ($(Capture $python @('--version')))"

Step "Programs (ffmpeg, Node.js, Ollama)"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    # Gyan's full build: ffprobe, and the drawtext filter the spotting burns frame times with
    $null = Winget-Install "Gyan.FFmpeg" "ffmpeg"
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Fail "ffmpeg could not be installed with winget. Install it by hand (on the PATH), then run this again:" "https://www.gyan.dev/ffmpeg/builds/" }
}
Ok "ffmpeg: $((Get-Command ffmpeg).Source)"
$wantWebui = ($WebUI -eq "on") -or ($Only -eq "webui") -or (($WebUI -eq "") -and ((Get-Content ".env" -Raw -ErrorAction SilentlyContinue) -notmatch "(?m)^\s*WINDOWS_WEBUI\s*=\s*0"))
if ($wantWebui) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        # Node.js only builds the web UI's frontend; its installer asks for administrator rights
        $null = Winget-Install "OpenJS.NodeJS.LTS" "Node.js"
        if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Fail "Node.js could not be installed with winget. Install it by hand, then run this again:" "https://nodejs.org/" }
    }
    Ok "Node.js: $(Capture 'node' @('--version'))"
}
if ($Only -ne "webui") {
    # Ollama on this PC, unless .env points the spotting at another server (as windowsackend.py reads it)
    $envText = Get-Content ".env" -Raw -ErrorAction SilentlyContinue
    $vlmApi = "ollama"; $vlmUrl = ""
    if ($envText -match "(?m)^\s*VLM_API\s*=\s*(\S+)") { $vlmApi = $Matches[1].ToLower() }
    if ($envText -match "(?m)^\s*VLM_URL\s*=\s*(\S+)") { $vlmUrl = $Matches[1] }
    $localOllama = ($vlmApi -eq "ollama") -and (($vlmUrl -eq "") -or ($vlmUrl -match "^https?://(127\.0\.0\.1|localhost|ollama|host\.docker\.internal)(:\d+)?/?$"))
    if ($localOllama) {
        $ollama = Join-Path $env:LocalAppData "Programs\Ollama\ollama.exe"
        $found = Get-Command ollama -ErrorAction SilentlyContinue
        if ($found) { $ollama = $found.Source }
        if (-not (Test-Path -LiteralPath $ollama)) {
            $null = Winget-Install "Ollama.Ollama" "Ollama"
            $ollama = Join-Path $env:LocalAppData "Programs\Ollama\ollama.exe"
            if (-not (Test-Path -LiteralPath $ollama)) { Fail "Ollama could not be installed with winget. Install it by hand, then run this again:" "https://ollama.com/download/windows" }
        }
        Ok "Ollama: $ollama"
    } else {
        Ok "vision-language model from $vlmUrl ($vlmApi, from .env): Ollama is not needed on this PC"
    }

    Step "Docker stack"
    if ((Get-Command docker -ErrorAction SilentlyContinue) -and ((Capture "docker" @("ps", "--format", "{{.Names}}")) -match "mmaudio-api|sound-search-api|hybrid-api")) {
        Warn "Containers of the Docker stack are running and hold the same ports. Stop them first:"
        Warn "docker compose stop"
        Fail "The native services cannot start beside them."
    }
    Ok "no Docker stack running"
}

$arguments = @((Join-Path $root "windows\backend.py"))
if ($Only -eq "webui") { $arguments += "webui" } else { $arguments += "install" }
if ($DataDir) { $arguments += @("--data-dir", $DataDir) }
if ($BbcArchive) { $arguments += @("--bbc-archive", $BbcArchive) }
if ($OwnSounds) { $arguments += @("--own-sounds", $OwnSounds) }
if ($ModelsDir) { $arguments += @("--models-dir", $ModelsDir) }
if ($Sessions) { $arguments += @("--sessions", $Sessions) }
if ($WebUI) { $arguments += @("--webui", $WebUI) }
if ($NoPrompt) { $arguments += "--no-prompt" }
if ($NoBrowser) { $arguments += "--no-browser" }
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$code = Native-Run $python $arguments
exit $code
