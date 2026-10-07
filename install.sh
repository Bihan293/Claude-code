#!/usr/bin/env bash
# opus-agent installer for Android Termux (also works on regular Linux/macOS).
#   curl -fsSL https://raw.githubusercontent.com/Bihan293/Claude-code/main/install.sh | bash
# or, from a clone:  bash install.sh
set -euo pipefail

REPO_URL="${OPUS_REPO_URL:-https://github.com/Bihan293/Claude-code.git}"
SRC_DIR="${OPUS_SRC_DIR:-$HOME/.opus-agent-src}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

if [ -n "${PREFIX:-}" ] && [[ "$PREFIX" == *com.termux* ]]; then
  say "Termux detected – installing system packages"
  pkg update -y || true
  pkg install -y python git ripgrep openssh gh termux-api which nano curl || \
    pkg install -y python git ripgrep openssh termux-api which nano curl
  # storage access (optional): termux-setup-storage
fi

for bin in python3 git; do
  command -v "$bin" >/dev/null || { echo "missing $bin"; exit 1; }
done

# Determine source: current dir if it is the repo, otherwise clone/update
if [ -f "./pyproject.toml" ] && grep -q 'name = "opus-agent"' ./pyproject.toml; then
  SRC_DIR="$(pwd)"
else
  if [ -d "$SRC_DIR/.git" ]; then
    say "Updating $SRC_DIR"; git -C "$SRC_DIR" pull --ff-only
  else
    say "Cloning $REPO_URL"; git clone --depth 1 "$REPO_URL" "$SRC_DIR"
  fi
fi

say "Installing python package (user site)"
PIP_FLAGS=""
python3 -m pip --version >/dev/null 2>&1 || { say "pip missing"; exit 1; }
# Termux python is not "externally managed"; on other distros fall back to --user/--break-system-packages
python3 -m pip install --upgrade "$SRC_DIR" 2>/dev/null || \
  python3 -m pip install --user --upgrade "$SRC_DIR" 2>/dev/null || \
  python3 -m pip install --user --break-system-packages --upgrade "$SRC_DIR"

if ! command -v opus >/dev/null; then
  BIN_DIR="${PREFIX:-$HOME/.local}/bin"
  mkdir -p "$BIN_DIR"
  cat > "$BIN_DIR/opus" <<EOF
#!/usr/bin/env sh
exec python3 -m opus_agent "\$@"
EOF
  chmod +x "$BIN_DIR/opus"
  say "Created launcher $BIN_DIR/opus (make sure it is on PATH)"
fi

mkdir -p "$HOME/projects"
say "Installed. Run:  opus setup   (first run asks for the API key)"
say "Then:           cd ~/projects/<repo> && opus"
