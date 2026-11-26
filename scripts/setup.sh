#!/usr/bin/env bash
# Set up the Audio Ops Framework from a checkout: virtual environment, PyTorch,
# the framework itself, runtime folders, a .env file and demo audio.
#
#   ./scripts/setup.sh              # training + serving, demo data included
#   ./scripts/setup.sh --dev        # contributors: adds tests, lint and docs tools
#   ./scripts/setup.sh --help       # all options
#
# Safe to run again: existing parts are kept and only missing parts are added.
# Works on Linux, macOS and Windows through WSL2.

set -euo pipefail

VENV=".venv"
EXTRAS="train,serve"
TORCH="auto"
PYTHON=""
DEMO=1
DEV=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [options]

  --dev              install contributor tools too (pytest, ruff, docs, pre-commit)
  --extras LIST      framework extras (default: train,serve). Choices: train,
                     serve, ast, s3, metrics, all
  --torch MODE       auto (default), cpu, cuda or skip
                       auto: CUDA build if an NVIDIA GPU is found (nvidia-smi),
                             the standard build on macOS, otherwise CPU-only
                       skip: keep the torch that is already installed
  --python PATH      Python to use (default: the newest python3.10 to 3.14 found)
  --venv DIR         virtual environment folder (default: .venv)
  --no-demo          don't generate the synthetic demo audio
  -h, --help         show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dev) DEV=1 ;;
    --extras) EXTRAS="${2:?--extras needs a value}"; shift ;;
    --torch) TORCH="${2:?--torch needs a value}"; shift ;;
    --python) PYTHON="${2:?--python needs a value}"; shift ;;
    --venv) VENV="${2:?--venv needs a value}"; shift ;;
    --no-demo) DEMO=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$TORCH" in auto|cpu|cuda|skip) ;; *) echo "--torch must be auto, cpu, cuda or skip" >&2; exit 2 ;; esac
[ "$DEV" = 1 ] && EXTRAS="dev,docs"

# Run from the repository root, wherever the script is called from.
cd "$(dirname "$0")/.."

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=; G=; Y=; R=; N=; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '%s  ok%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s  warning:%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

# ── 1. Python ────────────────────────────────────────────────────────────────
step "Looking for Python 3.10 to 3.14"
py_ok() { "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)' 2>/dev/null; }
if [ -n "$PYTHON" ]; then
  py_ok "$PYTHON" || die "$PYTHON is not Python 3.10 to 3.14."
else
  for cand in python3.13 python3.12 python3.11 python3.14 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then PYTHON="$cand"; break; fi
  done
fi
[ -n "$PYTHON" ] || die "No Python 3.10 to 3.14 found. Install one (for example 3.12) and run
       this script again, or point to it with --python /path/to/python3.12."
ok "$("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ── 2. Virtual environment ───────────────────────────────────────────────────
step "Virtual environment in $VENV"
if [ -x "$VENV/bin/python" ] && py_ok "$VENV/bin/python"; then
  ok "already exists, reusing it"
else
  rm -rf "$VENV"
  err="${TMPDIR:-/tmp}/audio-ops-venv.$$.err"
  if ! "$PYTHON" -m venv "$VENV" 2>"$err"; then
    cat "$err" >&2
    die "Could not create the virtual environment. On Debian or Ubuntu install the
       venv module first, for example: sudo apt install python3-venv
       (or python3.12-venv for a specific version)."
  fi
  ok "created"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip || die "Could not upgrade pip. Check your internet connection."

# ── 3. PyTorch ───────────────────────────────────────────────────────────────
step "PyTorch"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
if [ "$TORCH" = skip ] || "$PY" -c 'import torch' 2>/dev/null; then
  ok "using the installed torch $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '(none)')"
else
  if [ "$TORCH" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then TORCH=cuda
    elif [ "$(uname -s)" = Darwin ]; then TORCH=cuda   # the standard build; it includes MPS
    else TORCH=cpu; fi
  fi
  if [ "$TORCH" = cpu ]; then
    echo "  installing the CPU-only build (no NVIDIA GPU found)"
    if ! "$PY" -m pip install --quiet --timeout 30 --retries 1 torch --index-url "$CPU_INDEX" >/dev/null 2>&1; then
      warn "the CPU package index could not be reached, using the standard build from PyPI (larger download)"
      "$PY" -m pip install --quiet torch || die "Could not install PyTorch."
    fi
  else
    echo "  installing the standard build from PyPI (CUDA on Linux, MPS on macOS)"
    "$PY" -m pip install --quiet torch || die "Could not install PyTorch."
  fi
  ok "torch $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
fi

# ── 4. The framework ─────────────────────────────────────────────────────────
step "Audio Ops Framework [$EXTRAS] (editable install of this checkout)"
"$PY" -m pip install --quiet -e ".[$EXTRAS]" || die "Installing the framework failed (see the pip output above)."
ok "audio-ops $("$VENV/bin/audio-ops" version 2>/dev/null | head -n1)"
if [ "$DEV" = 1 ] && [ -d .git ]; then
  "$VENV/bin/pre-commit" install >/dev/null 2>&1 && ok "pre-commit hooks installed" || warn "pre-commit hooks not installed"
fi

# ── 5. Folders and .env ──────────────────────────────────────────────────────
step "Runtime folders and .env"
mkdir -p storage audio_input dagster_home mlruns mediamtx/test_audio
if [ -f .env ]; then ok ".env already exists, left unchanged"; else cp env.example .env; ok ".env created from env.example"; fi

# ── 6. System tools ──────────────────────────────────────────────────────────
step "System tools"
if command -v ffmpeg >/dev/null 2>&1; then ok "ffmpeg found"
else warn "ffmpeg not found. It is only needed for live RTSP streams. Install it with
       'sudo apt install ffmpeg' (Linux/WSL) or 'brew install ffmpeg' (macOS)."; fi

# ── 7. Demo data ─────────────────────────────────────────────────────────────
if [ "$DEMO" = 1 ]; then
  step "Demo audio for example_classifier"
  "$VENV/bin/audio-ops" demo-data --stream-demo mediamtx/test_audio/demo.wav >/dev/null \
    && ok "audio_input/example_classifier/ and mediamtx/test_audio/demo.wav" \
    || warn "could not generate demo data (run: $VENV/bin/audio-ops demo-data)"
fi

# ── 8. Check ─────────────────────────────────────────────────────────────────
step "Environment check (audio-ops doctor)"
"$VENV/bin/audio-ops" doctor || true
echo "  Unchecked extras are optional; add them later with --extras, e.g. --extras train,serve,s3"

cat <<EOF

${G}${B}Setup complete.${N}

Activate the environment in each new terminal:
    source $VENV/bin/activate

Then, for example:
    audio-ops run example_classifier       train, evaluate and score the demo task
    audio-ops serve                        inference API on http://localhost:8000/docs
    audio-ops dev                          Dagster UI on http://localhost:3000
    audio-ops new-task my_task --config-only --classes a,b

The make targets (make dev, make serve, make test, ...) use $VENV automatically.
EOF
