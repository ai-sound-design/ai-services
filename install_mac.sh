#!/bin/bash
# Installs (or updates) the backend and the web UI on a Mac with Apple Silicon:
# installs what is missing (Homebrew with Apple's command line tools, then ffmpeg,
# PostgreSQL 17 with pgvector, Python 3.12, Ollama and Node.js), sets up every service
# natively (no Docker: Docker on macOS cannot reach the Apple GPU), fetches the models,
# builds the web UI (github.com/ai-sound-design/ai-sound-webui) and opens it with the
# setup page.
# Safe to run again at any time; every step only does what is missing.
#
# For users who never open Terminal: the app "AI Sound Design Installer"
# (github.com/ai-sound-design/installer) asks what to install (these services, the web UI, the Pro Tools
# plugin, the sound database) and runs this script for the services.
#
#   bash install_mac.sh
#   bash install_mac.sh --data-dir "/Volumes/Data/AI Sound Design" --bbc-archive "/Volumes/Data/BBC" \
#                       --own-sounds ~/Sounds --sessions "/Volumes/Data/Pro Tools Sessions"
#
# Where it runs: the services are background agents, and macOS keeps those out of the
# Desktop, Documents and Downloads folders and iCloud Drive. A copy in one of them
# (a downloaded ZIP lands in Downloads) copies itself to ~/AI Sound Design/ai-services
# first and carries on there; an earlier installation there keeps its .env, its data
# and its environments. Elsewhere (a git checkout in ~/code, say) it runs in place, unless
# AISD_COPY=1 (the installer app sets it for the code it downloads to a temporary folder).
#
# The folders are optional and mean what they mean for install.ps1: the data folder
# takes what the backend writes (database, models, fetched audio, libraries; default
# ./data), an existing BBC archive, a folder of your own sounds and the folder of Pro
# Tools sessions the web UI opens are only read.
# Without the parameters the script asks for them when it runs in a terminal and has
# not been run before. Everything it decides goes into .env, the file the Docker stack
# reads as well.
#
# Afterwards the services run as launchd agents and start at every login. Next to the
# installation, in ~/AI Sound Design, are "Web UI", "Start backend", "Stop backend",
# "Backend status" and "Setup page" to double-click; on the command line:
#   python3 mac/backend.py status | stop | start | restart | logs <service>

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

step() { printf '\n\033[36m== %s\033[0m\n' "$1"; }
ok()   { printf '   \033[32m%s\033[0m\n' "$1"; }
warn() { printf '   \033[33m%s\033[0m\n' "$1"; }
fail() {
    printf '\n   \033[31m%s\033[0m\n' "$1"
    [ -n "${2:-}" ] && printf '   %s\n' "$2"
    printf '\n   Run the installer again afterwards.\n'
    exit 1
}

# One password prompt for everything that needs administrator rights (Homebrew, the
# Xcode licence), kept valid while the long downloads run
SUDO_KEEPER=""
SUDO_ARGS=()
need_sudo() {
    [ -n "$SUDO_KEEPER" ] && return 0
    if ! groups | grep -q -w admin; then
        fail "Installing Homebrew needs an administrator account, and $(id -un) is not one." \
             "Log in as an administrator once and run the installer there, or ask your administrator to install Homebrew (https://brew.sh)."
    fi
    if [ -t 0 ]; then
        echo "   macOS asks for the password of this Mac's account ($(id -un)); typing it shows nothing, then press Return."
        sudo -v || fail "Without the password the installation cannot go on."
    else
        # The installer app: no terminal to type into; SUDO_ASKPASS shows a dialog
        # (Homebrew's own installer uses it too)
        [ -n "${SUDO_ASKPASS:-}" ] || fail "Administrator rights are needed and there is no terminal to ask in."
        SUDO_ARGS=(-A)
        sudo -A -v || fail "Without the password the installation cannot go on."
    fi
    ( while true; do sudo -n true 2>/dev/null; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) &
    SUDO_KEEPER=$!
    trap '[ -n "$SUDO_KEEPER" ] && kill "$SUDO_KEEPER" 2>/dev/null' EXIT
}

step "macOS"
[ "$(uname -s)" = "Darwin" ] || fail "This script is for macOS; on Windows use install.ps1."
[ "$(uname -m)" = "arm64" ] || fail "An Apple Silicon Mac (M1 or later) is needed: the models run on its GPU."
MACOS="$(sw_vers -productVersion)"
[ "${MACOS%%.*}" -ge 14 ] || fail "macOS 14 (Sonoma) or newer is needed; this is $MACOS."
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
ok "macOS $MACOS on $(sysctl -n machdep.cpu.brand_string), $MEM_GB GB memory"
[ "$MEM_GB" -ge 32 ] || warn "MMAudio and the vision-language model together want about 24 GB; 32 GB or more is recommended."

step "Installation folder"
HOME_DIR="${AISD_HOME:-$HOME/AI Sound Design}"
TARGET="$HOME_DIR/ai-services"
PROTECTED=0
for protected in "$HOME/Desktop" "$HOME/Documents" "$HOME/Downloads" "$HOME/Library/Mobile Documents"; do
    case "$ROOT/" in "$protected"/*) PROTECTED=1 ;; esac
done
if { [ "$PROTECTED" = 1 ] || [ "${AISD_COPY:-0}" = 1 ]; } && [ "$ROOT" != "$TARGET" ]; then
    # Copied without what belongs to an installation, so an update keeps its settings,
    # its data and its environments; a file the new version dropped stays, unused.
    ok "copying the backend from $ROOT to $TARGET"
    mkdir -p "$TARGET"
    rsync -a --exclude '/.env' --exclude '/data/' --exclude '/.mac/' --exclude '/.git/' "$ROOT/" "$TARGET/" \
        || fail "Could not copy the backend to $TARGET."
    # A downloaded ZIP marks every file as coming from the internet; the copy is ours
    xattr -dr com.apple.quarantine "$TARGET" 2>/dev/null || true
    chmod +x "$TARGET/install_mac.sh" 2>/dev/null || true
    [ "$PROTECTED" = 1 ] && ok "the copy in $(dirname "$ROOT") can be deleted afterwards"
    exec bash "$TARGET/install_mac.sh" "$@"
fi
ok "$ROOT"

step "Homebrew"
BREW=/opt/homebrew/bin/brew
if [ ! -x "$BREW" ]; then
    warn "Homebrew is not installed; installing it with Apple's command line tools (several minutes, about 1 GB)"
    need_sudo
    # NONINTERACTIVE: no "Press RETURN"; it uses the sudo rights obtained above
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
        || fail "Homebrew could not be installed (see above)." "https://brew.sh"
    [ -x "$BREW" ] || fail "Homebrew reported success but $BREW is missing."
fi
if ! "$BREW" config >/dev/null 2>&1; then
    if "$BREW" config 2>&1 | grep -q -i "xcode license"; then
        # The full Xcode is installed and its licence was never accepted; Homebrew refuses until it is
        warn "Xcode's licence has not been accepted on this Mac; accepting it"
        need_sudo
        sudo ${SUDO_ARGS[@]+"${SUDO_ARGS[@]}"} xcodebuild -license accept || fail "Could not accept the Xcode licence." "sudo xcodebuild -license accept"
    fi
    "$BREW" config >/dev/null 2>&1 || fail "Homebrew does not work; in Terminal: $BREW doctor"
fi
if ! xcode-select -p >/dev/null 2>&1; then
    # Normally Homebrew's installer brings them; this opens Apple's own dialog
    xcode-select --install >/dev/null 2>&1 || true
    fail "Apple's command line tools are missing. A dialog asks to install them: click Install, wait until it is done."
fi
ok "$("$BREW" --version | head -1)"

step "Packages (ffmpeg, PostgreSQL 17 with pgvector, Python 3.12, Ollama, Node.js)"
# Homebrew is brought up to date first: a formula may need a newer Homebrew than the
# installed one, and its post-install step then fails half-way (PostgreSQL did)
"$BREW" update --quiet >/dev/null 2>&1 || warn "brew update failed; carrying on with the installed Homebrew"
# Node.js only builds the web UI's frontend
PACKAGES=(ffmpeg postgresql@17 pgvector python@3.12 ollama node git)
MISSING=()
for p in "${PACKAGES[@]}"; do "$BREW" list --versions "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
if [ ${#MISSING[@]} -gt 0 ]; then
    "$BREW" install "${MISSING[@]}" || fail "Homebrew could not install: ${MISSING[*]}"
fi
# PostgreSQL's files under /opt/homebrew/share come from its post-install step
if [ ! -f /opt/homebrew/share/postgresql@17/postgres.bki ]; then
    "$BREW" postinstall postgresql@17 >/dev/null 2>&1 || true
    [ -f /opt/homebrew/share/postgresql@17/postgres.bki ] || fail "PostgreSQL 17 is incompletely installed. In Terminal:" \
        "brew update && brew reinstall postgresql@17 pgvector"
fi
[ -f /opt/homebrew/share/postgresql@17/extension/vector.control ] || fail "pgvector is missing for PostgreSQL 17." "brew reinstall pgvector"
for p in "${PACKAGES[@]}"; do ok "$("$BREW" list --versions "$p")"; done

step "Docker stack"
if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -q -E "mmaudio-api|sound-search-api|hybrid-api"; then
    warn "Containers of the Docker stack are running and hold the same ports. Stop them first:"
    warn "docker compose down"
    fail "The native services cannot start beside them."
fi
ok "no Docker stack running"

# The rest needs no administrator rights
[ -n "$SUDO_KEEPER" ] && { kill "$SUDO_KEEPER" 2>/dev/null || true; SUDO_KEEPER=""; sudo -k; }
exec /opt/homebrew/opt/python@3.12/bin/python3.12 "$ROOT/mac/backend.py" install "$@"
