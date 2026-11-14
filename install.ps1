# Installs the backend on Windows: checks what the stack needs, builds and starts it,
# and opens the setup page, which fetches the models and fills the sound index with
# progress shown. Safe to run again at any time; every step only does what is missing.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -DataDir "D:\AI Sound Design" -BbcArchive "D:\BBC Sound Effects" -OwnSounds "D:\Sounds"
#
# Docker Desktop and the NVIDIA driver are not installed by this script: it stops
# with the link to their installers. The folders are optional: the data folder takes
# what the stack writes (fetched audio, libraries; default .\data, worth a bigger
# drive for the 60 GB of the BBC archive), two are only read: an existing copy of
# the BBC archive is registered instead of downloaded, and a folder of your own
# sounds becomes a library of the search. The models (about 21 GB) live in Docker
# volumes, where they load in seconds; -ModelsDir puts them in a folder instead,
# visible as files but minutes to load. What is already there follows the choice.
# Without the parameters the script asks for them when it runs in a console.
# Everything it decides goes into .env; docker-compose.yml is never changed. On the
# first run it also gives the database a random password in place of the example's.
#
# -Ollama docker|host: where the vision-language model runs. docker (the default of a
# fresh .env) is the stack's own Ollama container; host is an Ollama installed on
# Windows (installed with winget when missing, OLLAMA_HOST=0.0.0.0 set for this account
# so the containers reach it as host.docker.internal), which then also serves other
# programs on this PC. Without the parameter an existing choice in .env is kept.
# -NoBrowser leaves the setup page closed at the end (the installer app opens it itself).

param(
    [string]$DataDir = "",
    [string]$BbcArchive = "",
    [string]$OwnSounds = "",
    [string]$ModelsDir = "",
    [ValidateSet("", "docker", "host")][string]$Ollama = "",
    [switch]$NoPrompt,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "   $text" -ForegroundColor Yellow }
function Fail($text, $link) {
    Write-Host ""
    Write-Host "   $text" -ForegroundColor Red
    if ($link) { Write-Host "   $link" -ForegroundColor White }
    Write-Host ""
    Write-Host "   Run this script again afterwards." -ForegroundColor Gray
    exit 1
}
# Native programs write warnings to stderr, which Windows PowerShell 5.1 turns into
# errors under "Stop"; they run through cmd, which merges the streams, and their
# exit code is what counts. Output goes to the console, not into the return value.
function Quiet($command) { cmd /c "$command >nul 2>&1"; return $LASTEXITCODE }
function Run($command) { cmd /c "$command 2>&1" | Out-Host; return $LASTEXITCODE }
function Capture($command) { $out = cmd /c "$command 2>&1"; return ($out | Out-String).Trim() }
function Set-EnvValue($key, $value) {
    # Replace or append KEY=value in .env; folders with forward slashes, which Compose takes on Windows
    $text = Get-Content ".env" -Raw
    $line = "$key=$value"
    if ($text -match "(?m)^\s*$key\s*=") { $text = $text -replace "(?m)^\s*$key\s*=.*$", $line }
    else { $text = $text.TrimEnd() + "`r`n$line`r`n" }
    Set-Content -Path ".env" -Value $text -Encoding UTF8
}
function Ask-Folder($prompt) {
    while ($true) {
        $answer = (Read-Host $prompt).Trim().Trim('"')
        if (-not $answer) { return "" }
        if (Test-Path -LiteralPath $answer -PathType Container) { return (Resolve-Path -LiteralPath $answer).Path }
        Warn "$answer is not a folder"
    }
}

Step "Windows"
$os = Get-CimInstance Win32_OperatingSystem
if ([int]$os.BuildNumber -lt 19041) { Fail "Windows 10 version 2004 (build 19041) or newer is needed for WSL 2; this is build $($os.BuildNumber)." }
Ok "$($os.Caption) build $($os.BuildNumber)"

Step "NVIDIA driver"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "No NVIDIA driver found (nvidia-smi is missing). Install the current driver for your GPU:" "https://www.nvidia.com/Download/index.aspx" }
$gpu = Capture "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
if ($LASTEXITCODE -ne 0 -or -not $gpu) { Fail "nvidia-smi does not answer; reinstall the NVIDIA driver:" "https://www.nvidia.com/Download/index.aspx" }
Ok "$gpu"

Step "Docker Desktop"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "Docker Desktop is not installed. Install it (with the WSL 2 backend, the default), start it once, then run this script again:" "https://docs.docker.com/desktop/setup/install/windows-install/" }
if ((Quiet "docker info") -ne 0) {
    $app = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $app) {
        Warn "Docker Desktop is installed but not running; starting it"
        Start-Process $app
        $deadline = (Get-Date).AddMinutes(3)
        do { Start-Sleep -Seconds 5 } while ((Quiet "docker info") -ne 0 -and (Get-Date) -lt $deadline)
    }
    if ((Quiet "docker info") -ne 0) { Fail "Docker Desktop does not answer. Start it from the Start menu, wait until its whale icon is steady, then run this script again." "https://docs.docker.com/desktop/setup/install/windows-install/" }
}
$version = Capture "docker version --format {{.Server.Version}}"
Ok "Docker $version is running"
if ((Quiet "docker compose version") -ne 0) { Fail "Docker Compose v2 is missing; update Docker Desktop:" "https://docs.docker.com/desktop/release-notes/" }

Step "GPU inside containers"
$seen = Capture "docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L"
if ($LASTEXITCODE -ne 0 -or $seen -notmatch "GPU 0") { Fail "Containers do not see the GPU. In Docker Desktop, Settings > General, use the WSL 2 based engine; update the NVIDIA driver; then restart Docker Desktop. The test said: $seen" "https://docs.docker.com/desktop/features/gpu/" }
Ok ($seen -split "`n" | Where-Object { $_ -match "GPU 0" } | Select-Object -First 1)

Step "Memory for the Docker VM"
$wslconfig = Join-Path $env:UserProfile ".wslconfig"
$needsRestart = $false
$content = ""
if (Test-Path $wslconfig) { $content = Get-Content $wslconfig -Raw }
$memory = 0
if ($content -match "(?m)^\s*memory\s*=\s*(\d+)\s*GB") { $memory = [int]$Matches[1] }
if ($memory -ge 24) {
    Ok "$wslconfig gives the VM $memory GB"
} else {
    $total = [math]::Round($os.TotalVisibleMemorySize / 1MB)
    if ($total -lt 32) { Warn "This machine has $total GB; MMAudio needs about 10 GB inside the VM, 24 GB are recommended for the VM and the machine should have 32 GB." }
    if ($content -notmatch "(?m)^\s*\[wsl2\]") { $content = $content.TrimEnd() + "`r`n[wsl2]`r`n" }
    if ($content -match "(?m)^\s*memory\s*=") { $content = $content -replace "(?m)^\s*memory\s*=.*$", "memory=24GB" }
    else { $content = $content -replace "(?m)^\s*\[wsl2\]\s*$", "[wsl2]`r`nmemory=24GB" }
    Set-Content -Path $wslconfig -Value $content -Encoding ASCII
    $needsRestart = $true
    Ok "wrote memory=24GB into $wslconfig"
}

Step "Docker's disk"
# The database and the images live inside Docker Desktop's VM disk; where that is,
# is a Docker Desktop setting (Settings > Resources > Advanced > Disk image location)
$dockerDisk = Join-Path $env:LocalAppData "Docker\wsl"
foreach ($settings in (Join-Path $env:AppData "Docker\settings-store.json"), (Join-Path $env:AppData "Docker\settings.json"), (Join-Path $env:LocalAppData "Docker\settings.json")) {
    if (Test-Path $settings) {
        try {
            $json = Get-Content $settings -Raw | ConvertFrom-Json
            foreach ($key in "DataFolder", "dataFolder") { if ($json.PSObject.Properties[$key] -and $json.$key) { $dockerDisk = $json.$key } }
        } catch { }
    }
}
# The disk file itself: the WSL 2 engine keeps it under %LocalAppData%\Docker\wsl unless moved
$vhdx = $null
foreach ($candidate in (Join-Path $dockerDisk "DockerDesktop.vhdx"), (Join-Path $dockerDisk "docker_data.vhdx"),
                       (Join-Path $env:LocalAppData "Docker\wsl\disk\docker_data.vhdx"), (Join-Path $env:LocalAppData "Docker\wsl\data\ext4.vhdx")) {
    if (Test-Path $candidate) { $vhdx = $candidate; break }
}
if ($vhdx) { $dockerDisk = $vhdx }
$diskDrive = (Split-Path -Qualifier $dockerDisk)
$diskFree = [math]::Round((Get-PSDrive $diskDrive.TrimEnd(":")).Free / 1GB)
$diskSize = ""
if ($vhdx) { $diskSize = ", $([math]::Round((Get-Item $vhdx).Length / 1GB)) GB so far" }
Ok "images and the database live in $dockerDisk ($diskFree GB free on $diskDrive$diskSize)"
Write-Host "   To put them on another drive: Docker Desktop > Settings > Resources > Advanced > Disk image location" -ForegroundColor Gray

Step "Configuration"
$interactive = -not $NoPrompt -and -not [Console]::IsInputRedirected -and [Environment]::UserInteractive
$fresh = -not (Test-Path ".env")
if ($fresh) {
    Copy-Item ".env.example" ".env"
    # The example's password is public; the database port is published on this machine
    $password = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ })
    Set-EnvValue "POSTGRES_PASSWORD" $password
    Ok "created .env from .env.example with a random database password (edit it to change ports or folders)"
    if (-not $DataDir -and $interactive) {
        Write-Host ""
        Write-Host "   The data folder takes what the stack writes: fetched or downloaded audio (the BBC archive is" -ForegroundColor Gray
        Write-Host "   about 60 GB) and your libraries. Default: $root\data" -ForegroundColor Gray
        $DataDir = (Read-Host "   Data folder, Enter for the default").Trim().Trim('"')
    }
} else {
    Ok ".env exists (kept; delete it to start from the example again)"
}
if ($DataDir) {
    if (-not (Test-Path -LiteralPath $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }
    $data = (Resolve-Path -LiteralPath $DataDir).Path
    foreach ($sub in "audio", "libraries", "archive", "own-sounds") {
        $folder = Join-Path $data $sub
        if (-not (Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder | Out-Null }
    }
    $slashes = $data -replace "\\", "/"
    Set-EnvValue "AUDIO_HOST_DIR" "$slashes/audio"
    Set-EnvValue "LIBRARIES_HOST_DIR" "$slashes/libraries"
    Set-EnvValue "BBC_ARCHIVE_HOST_DIR" "$slashes/archive"
    Set-EnvValue "OWN_SOUNDS_HOST_DIR" "$slashes/own-sounds"
    Ok "data folder: $data"
}

# Where the models live: the Docker volumes "models" (MMAudio's weights, the search
# models) and "ollama-models" (the vision-language model), or folders when asked for.
# Models that are already somewhere else (an earlier version's data folder or its four
# separate volumes, or the other kind of store after a change of mind) are copied once.
function Env-Value($key) {
    if ((Get-Content ".env" -Raw) -match "(?m)^\s*$key\s*=\s*(.*?)\s*$") { return $Matches[1].Trim('"') }
    return ""
}
function Remove-EnvValue($key) {
    $text = Get-Content ".env" -Raw
    Set-Content -Path ".env" -Value ($text -replace "(?m)^\s*$key\s*=.*(\r?\n|$)", "") -Encoding UTF8
}
# docker called directly for arguments with quotes or spaces, which the cmd wrappers
# above cannot pass through
function Docker-Run([string[]]$arguments) {
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & docker @arguments | Out-Host
    $ErrorActionPreference = $eap
    return $LASTEXITCODE
}
function Docker-Capture([string[]]$arguments) {
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $out = & docker @arguments 2>$null
    $ErrorActionPreference = $eap
    return ($out | Out-String).Trim()
}
function Volume-Exists($name) { return [bool]((Docker-Capture @("volume", "ls", "-q")) -split "`n" | Where-Object { $_.Trim() -eq $name }) }
function Store-HasContent($store) {
    # $store is "vol:<name>" or a folder; a volume is never created just to look into it
    if ($store.StartsWith("vol:")) {
        $name = $store.Substring(4)
        if (-not (Volume-Exists $name)) { return $false }
        return [bool](Docker-Capture @("run", "--rm", "-v", "${name}:/v", "alpine", "sh", "-c", "ls -A /v | head -1"))
    }
    return (Test-Path -LiteralPath $store) -and [bool](Get-ChildItem -LiteralPath $store -Force | Select-Object -First 1)
}
function Mount-Arg($store) {
    if ($store.StartsWith("vol:")) { return $store.Substring(4) }
    New-Item -ItemType Directory -Force -Path $store | Out-Null
    return ((Resolve-Path -LiteralPath $store).Path -replace "\\", "/")
}
function Ensure-Volume($store) {
    if (-not $store.StartsWith("vol:")) { return }
    $name = $store.Substring(4); $short = $name.Substring($project.Length + 1)
    if (-not (Volume-Exists $name)) {
        Docker-Run @("volume", "create", "--label", "com.docker.compose.project=$project", "--label", "com.docker.compose.volume=$short", $name) | Out-Null
    }
}
function Copy-Store($from, $to, $what, $subfolder = "") {
    Ensure-Volume $to
    Warn "copying $what from $from to $to (once; several GB)"
    $code = Docker-Run @("run", "--rm", "-v", "$(Mount-Arg $from):/from:ro", "-v", "$(Mount-Arg $to):/to", "alpine", "sh", "-c", "mkdir -p /to/$subfolder && cp -a /from/. /to/$subfolder/")
    if ($code -ne 0) { Fail "Could not copy $what; see the messages above." }
}
function Copy-OllamaModel($from, $to, $model) {
    # Only the model the stack uses, taken by its manifest: an Ollama store may hold many
    # more (an existing C:\Users\you\.ollama), and its blobs are plain files named by digest.
    if (-not $model.Contains(":")) { $model = "${model}:latest" }
    $path = if ($model.Contains("/")) { $model -replace ":", "/" } else { "registry.ollama.ai/library/" + ($model -replace ":", "/") }
    $parent = $path.Substring(0, $path.LastIndexOf("/"))
    $script = @"
set -e
m="/from/models/manifests/$path"
if [ ! -f "`$m" ]; then exit 3; fi
mkdir -p /to/models/blobs "/to/models/manifests/$parent"
for d in `$(grep -o 'sha256:[0-9a-f]*' "`$m" | sort -u); do
    f="sha256-`${d#sha256:}"
    [ -f "/to/models/blobs/`$f" ] || cp "/from/models/blobs/`$f" "/to/models/blobs/`$f"
done
cp "`$m" "/to/models/manifests/$path"
"@
    $file = Join-Path $env:TEMP "ai-sound-design-copy-model.sh"
    [IO.File]::WriteAllText($file, $script.Replace("`r`n", "`n"))
    Ensure-Volume $to
    Warn "copying the vision-language model $model from $from to $to (once; several GB)"
    $code = Docker-Run @("run", "--rm", "-v", "$(Mount-Arg $from):/from:ro", "-v", "$(Mount-Arg $to):/to", "-v", "$($file.Replace('\', '/')):/copy.sh:ro", "alpine", "sh", "/copy.sh")
    Remove-Item $file -ErrorAction SilentlyContinue
    if ($code -eq 3) { Warn "$model is not in $from; Ollama fetches it on the first start" }
    elseif ($code -ne 0) { Fail "Could not copy $model; see the messages above." }
}
# The compose project, as Compose picks it: COMPOSE_PROJECT_NAME in .env, else the name in
# docker-compose.yml, else the folder's name; it prefixes the volumes
$project = Env-Value "COMPOSE_PROJECT_NAME"
if (-not $project -and ((Get-Content "docker-compose.yml" -Raw) -match "(?m)^name:\s*([A-Za-z0-9_-]+)")) { $project = $Matches[1] }
if (-not $project) { $project = (Split-Path -Leaf $root).ToLower() -replace "[^a-z0-9_-]", "" }
if (-not $ModelsDir -and $interactive -and $fresh) {
    Write-Host ""
    Write-Host "   The models (about 21 GB) live in Docker volumes, where a model loads in seconds. A folder" -ForegroundColor Gray
    Write-Host "   keeps them visible as files but loads them in minutes through Docker Desktop's file sharing." -ForegroundColor Gray
    $answer = (Read-Host "   Folder for the models, Enter to keep them in Docker volumes (recommended)").Trim().Trim('"')
    if ($answer) { $ModelsDir = $answer }
}
if ($ModelsDir) {
    if (-not (Test-Path -LiteralPath $ModelsDir)) { New-Item -ItemType Directory -Path $ModelsDir | Out-Null }
    $ModelsDir = (Resolve-Path -LiteralPath $ModelsDir).Path
}
$vlm = Env-Value "VLM_MODEL"
if (-not $vlm) { $vlm = "gemma4:e4b-it-qat" }
$data = "./data"
if ((Env-Value "AUDIO_HOST_DIR") -match "^(.*)/audio/?$") { $data = $Matches[1] }
if (-not [System.IO.Path]::IsPathRooted($data)) { $data = Join-Path $root $data }
foreach ($store in @(
        @{ Key = "MODELS_STORE"; Volume = "models"; Sub = "models"; OldKey = "MODELS_HOST_DIR"; What = "the MMAudio and search models" },
        @{ Key = "OLLAMA_MODELS_STORE"; Volume = "ollama-models"; Sub = "ollama"; OldKey = "OLLAMA_MODELS_HOST_DIR"; What = "the vision-language model" })) {
    $volume = "vol:${project}_$($store.Volume)"
    $current = Env-Value $store.Key
    if ($ModelsDir) { $target = Join-Path $ModelsDir $store.Sub }
    elseif ($current -and ($current -match "[/\\]")) { $target = $current }
    else { $target = $volume }
    if (-not $target.StartsWith("vol:")) {
        if (-not [System.IO.Path]::IsPathRooted($target)) { $target = Join-Path $root $target }
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        $target = (Resolve-Path -LiteralPath $target).Path
    }
    # Where the models may be now: the other kind of store, an earlier version's folder
    $sources = @()
    if ($target -ne $volume) { $sources += $volume }
    foreach ($old in @((Env-Value $store.OldKey), (Join-Path $data $store.Sub))) {
        if ($old -and ($old -match "[/\\]")) {
            if (-not [System.IO.Path]::IsPathRooted($old)) { $old = Join-Path $root $old }
            if ((Test-Path -LiteralPath $old) -and ((Resolve-Path -LiteralPath $old).Path -ne $target)) { $sources += (Resolve-Path -LiteralPath $old).Path }
        }
    }
    if (-not (Store-HasContent $target)) {
        $from = $sources | Where-Object { Store-HasContent $_ } | Select-Object -First 1
        if ($from -and $store.Volume -eq "ollama-models") { Copy-OllamaModel $from $target $vlm }
        elseif ($from) { Copy-Store $from $target $store.What }
    }
    if ($store.Volume -eq "models" -and -not (Store-HasContent $target)) {
        # the four volumes of versions before the single models store
        foreach ($pair in @(@("hf-cache", "huggingface"), @("torch-cache", "torch"), @("mmaudio-weights", "mmaudio/weights"), @("mmaudio-ext-weights", "mmaudio/ext_weights"))) {
            $old = "vol:${project}_$($pair[0])"
            if (Store-HasContent $old) { Copy-Store $old $target "$($pair[1]) (old volume)" $pair[1] }
        }
    }
    if ($target.StartsWith("vol:")) {
        Remove-EnvValue $store.Key
        Ok "$($store.What): Docker volume $($target.Substring(4)) (Docker Desktop > Volumes)"
    } else {
        Set-EnvValue $store.Key ($target -replace "\\", "/")
        Ok "$($store.What): folder $target"
    }
    Remove-EnvValue $store.OldKey
}
if (-not $BbcArchive -and $interactive) {
    Write-Host ""
    Write-Host "   The BBC Sound Effects archive (free for research use under the BBC's terms) can be" -ForegroundColor Gray
    Write-Host "   downloaded later from the setup page, or an existing copy on disk is used, read-only." -ForegroundColor Gray
    $BbcArchive = Ask-Folder "   Folder of an existing BBC archive (<id>.wav or .mp3 files), Enter for none"
}
if (-not $OwnSounds -and $interactive) {
    $OwnSounds = Ask-Folder "   Folder with your own sounds to index (read-only), Enter for none"
}
if ($BbcArchive) {
    if (-not (Test-Path -LiteralPath $BbcArchive -PathType Container)) { Fail "The BBC archive folder does not exist: $BbcArchive" }
    Set-EnvValue "BBC_ARCHIVE_HOST_DIR" ((Resolve-Path -LiteralPath $BbcArchive).Path -replace "\\", "/")
    Ok "BBC archive: $BbcArchive (read-only; registered instead of downloaded)"
}
if ($OwnSounds) {
    if (-not (Test-Path -LiteralPath $OwnSounds -PathType Container)) { Fail "The sounds folder does not exist: $OwnSounds" }
    $name = ((Split-Path -Leaf $OwnSounds).ToLower() -replace "[^a-z0-9_-]+", "-").Trim("-")
    if (-not $name) { $name = "own" }
    Set-EnvValue "OWN_SOUNDS_HOST_DIR" ((Resolve-Path -LiteralPath $OwnSounds).Path -replace "\\", "/")
    Set-EnvValue "OWN_SOUNDS_NAME" $name
    Ok "own sounds: $OwnSounds (read-only; library '$name')"
}
$drive = (Get-Item $root).PSDrive
$free = [math]::Round((Get-PSDrive $drive.Name).Free / 1GB)
if ($free -lt 90) { Warn "Only $free GB free on drive $($drive.Name): images, models and the index take about 80 GB, a downloaded BBC archive another 60 GB." } else { Ok "$free GB free on drive $($drive.Name):" }

# Where the vision-language model runs: the stack's own Ollama container, or an Ollama
# installed on Windows, which the containers reach as host.docker.internal.
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
function Reachable($url) {
    try { $null = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 4; return $true } catch { return $false }
}
if ($Ollama -eq "host") {
    Step "Vision-language model: the Ollama on this PC"
    $exe = Join-Path $env:LocalAppData "Programs\Ollama\ollama.exe"
    $found = Get-Command ollama -ErrorAction SilentlyContinue
    if ($found) { $exe = $found.Source }
    if (-not (Test-Path -LiteralPath $exe)) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { Fail "Ollama is not installed, and winget is missing to install it. Install Ollama for Windows, then run this again:" "https://ollama.com/download/windows" }
        Warn "Ollama is not installed; installing it with winget (about 1 GB)"
        $code = Native-Run "winget" @("install", "-e", "--id", "Ollama.Ollama", "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity")
        $exe = Join-Path $env:LocalAppData "Programs\Ollama\ollama.exe"
        if ($code -ne 0 -or -not (Test-Path -LiteralPath $exe)) { Fail "Ollama could not be installed with winget. Install it by hand, then run this again:" "https://ollama.com/download/windows" }
    }
    Ok "Ollama: $exe"
    # Ollama listens on localhost only unless OLLAMA_HOST says otherwise: a setting of
    # this Windows account, read by the Ollama app when it starts
    if ([Environment]::GetEnvironmentVariable("OLLAMA_HOST", "User") -ne "0.0.0.0") {
        [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")
        Ok "OLLAMA_HOST=0.0.0.0 set for this Windows account, so the containers can reach it"
        Get-Process "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    $env:OLLAMA_HOST = "0.0.0.0"
    if (-not (Reachable "http://127.0.0.1:11434/api/version")) {
        $app = Join-Path (Split-Path -Parent $exe) "ollama app.exe"
        if (Test-Path -LiteralPath $app) { Start-Process -FilePath $app } else { Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden }
        $deadline = (Get-Date).AddMinutes(1)
        do { Start-Sleep -Seconds 2 } while (-not (Reachable "http://127.0.0.1:11434/api/version") -and (Get-Date) -lt $deadline)
        if (-not (Reachable "http://127.0.0.1:11434/api/version")) { Fail "Ollama does not answer on http://127.0.0.1:11434. Start it from the Start menu, then run this again." }
    }
    Ok "Ollama answers on port 11434 (it starts with Windows)"
    Ok "fetching the model $vlm if it is missing (about 6 GB the first time)"
    if ((Native-Run $exe @("pull", $vlm)) -ne 0) { Fail "Ollama could not fetch $vlm; see the messages above." }
    Set-EnvValue "BUNDLED_OLLAMA" "0"
    Set-EnvValue "VLM_URL" "http://host.docker.internal:11434"
    Set-EnvValue "VLM_API" "ollama"
    Ok "the stack uses it: BUNDLED_OLLAMA=0, VLM_URL=http://host.docker.internal:11434"
} elseif ($Ollama -eq "docker") {
    Step "Vision-language model: Ollama in the stack"
    Set-EnvValue "BUNDLED_OLLAMA" "1"
    if ((Env-Value "VLM_URL") -match "host\.docker\.internal") { Set-EnvValue "VLM_URL" "http://ollama:11434"; Set-EnvValue "VLM_API" "ollama" }
    Ok "the stack runs its own Ollama container (BUNDLED_OLLAMA=1); the model is fetched on the first start"
}

Step "Building the images (the first time pulls the PyTorch base image: several GB)"
if ((Run "docker compose build") -ne 0) { Fail "The build failed; see the messages above." }

Step "Starting the services"
if ((Run "docker compose up -d") -ne 0) { Fail "The services did not start; see the messages above." }

$port = 8020
if ((Get-Content ".env" -Raw) -match "(?m)^\s*SETUP_PORT\s*=\s*(\d+)") { $port = [int]$Matches[1] }
$url = "http://localhost:$port/"
Step "Setup page"
$deadline = (Get-Date).AddMinutes(2)
$up = $false
do {
    Start-Sleep -Seconds 3
    try { $null = Invoke-WebRequest -UseBasicParsing -Uri "${url}status" -TimeoutSec 4; $up = $true } catch { }
} while (-not $up -and (Get-Date) -lt $deadline)
if ($up) { Ok "$url"; if (-not $NoBrowser) { Start-Process $url } } else { Warn "The page does not answer yet; open $url in a minute." }
if ((Env-Value "BUNDLED_OLLAMA") -eq "0" -and (Env-Value "VLM_URL") -match "host\.docker\.internal") {
    # The host's Ollama from inside a container: Windows' firewall may stand in between
    $seen = Docker-Capture @("run", "--rm", "alpine", "wget", "-qO-", "-T", "5", "http://host.docker.internal:11434/api/version")
    if ($seen -match '"version"') { Ok "the containers reach the Ollama on this PC ($seen)" }
    else {
        Warn "The containers cannot reach the Ollama on this PC (http://host.docker.internal:11434), so the spotting"
        Warn "will fail. Allow ollama.exe through the Windows firewall (Windows Security > Firewall & network"
        Warn "protection > Allow an app through firewall), or run this again with -Ollama docker."
    }
}

Write-Host ""
Write-Host "The services fetch their models on this first start (MMAudio about 10 GB, the vision-language model about 6 GB);" -ForegroundColor Gray
Write-Host "the setup page shows the progress. There, 'Set everything up' fills the sound index." -ForegroundColor Gray
if ($needsRestart) {
    Write-Host ""
    Warn "The memory setting takes effect when Docker Desktop restarts: quit it from the tray icon and start it again"
    Warn "once the images are built. The services start on their own afterwards."
}
