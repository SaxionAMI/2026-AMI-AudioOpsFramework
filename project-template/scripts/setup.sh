#!/usr/bin/env bash
# Set up a project repo built on the Audio Ops Framework: a virtual
# environment, PyTorch, the framework and the other dependencies from
# pyproject.toml, runtime folders and a .env file. Then it checks the tasks and
# reports how much training audio it finds.
#
#   ./scripts/setup.sh                 # automatic: works out how to get the framework
#   ./scripts/setup.sh --help          # all options
#
# Safe to run again. Works on Linux, macOS and Windows through WSL2.
# This script is the same in every project repo; it comes from the framework's
# project-template/ folder.

set -euo pipefail

VENV=".venv"
TORCH="auto"
PYTHON=""
FRAMEWORK="${FRAMEWORK:-auto}"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [options]

  --framework SOURCE  where to get the Audio Ops Framework (default: auto)
                        auto   try in order: the URL in pyproject.toml over HTTPS,
                               the same repository over SSH, then a framework
                               checkout next to this repo
                        https  only the URL in pyproject.toml
                        ssh    the same repository over SSH (needs your SSH key
                               on GitHub; use this while the repo is private)
                        PATH   a local framework checkout, installed editable
  --torch MODE        auto (default), cpu, cuda or skip
                        auto: CUDA build if an NVIDIA GPU is found (nvidia-smi),
                              the standard build on macOS, otherwise CPU-only
  --python PATH       Python to use (default: the newest python3.10 to 3.14 found)
  --venv DIR          virtual environment folder (default: .venv)
  -h, --help          show this help

The FRAMEWORK environment variable works like --framework.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --framework) FRAMEWORK="${2:?--framework needs a value}"; shift ;;
    --torch) TORCH="${2:?--torch needs a value}"; shift ;;
    --python) PYTHON="${2:?--python needs a value}"; shift ;;
    --venv) VENV="${2:?--venv needs a value}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
case "$TORCH" in auto|cpu|cuda|skip) ;; *) echo "--torch must be auto, cpu, cuda or skip" >&2; exit 2 ;; esac

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "error: no pyproject.toml in $(pwd)" >&2; exit 1; }

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=; G=; Y=; R=; N=; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '%s  ok%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s  warning:%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

case "$FRAMEWORK" in
  auto|https|ssh) ;;
  *) [ -d "$FRAMEWORK" ] || die "--framework must be auto, https, ssh or an existing folder (got '$FRAMEWORK')." ;;
esac

# Never wait for a password prompt; fail instead and try the next option.
export GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new}"

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
  ok "already exists, reusing it ($("$VENV/bin/python" -c 'import sys; print(sys.version.split()[0])'))"
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

# ── Read pyproject.toml ──────────────────────────────────────────────────────
# Prints shell assignments: FW_EXTRAS, FW_URL (git URL without git+ and @ref),
# FW_REF, FW_SPEC (the requirement as written), OTHER_DEPS, TORCH_EXTRA.
PARSED="$("$PY" - <<'PYEOF'
import shlex
import sys
try:
    import tomllib
except ModuleNotFoundError:                       # Python 3.10
    from pip._vendor import tomli as tomllib
from pip._vendor.packaging.requirements import Requirement

deps = tomllib.load(open("pyproject.toml", "rb")).get("project", {}).get("dependencies", [])
fw, others, torch_extra = None, [], []
for spec in deps:
    req = Requirement(spec)
    name = req.name.lower().replace("_", "-")
    if name == "audio-ops-framework":
        fw = (spec, req)
    elif name in ("torch", "torchvision", "torchaudio"):
        torch_extra.append(spec)
    else:
        others.append(spec)
if fw is None:
    sys.exit("pyproject.toml has no audio-ops-framework entry in [project] dependencies")
spec, req = fw
url, ref = "", ""
if req.url:
    u = req.url[4:] if req.url.startswith("git+") else req.url
    if "@" in u.rsplit("/", 1)[-1]:
        u, ref = u.rsplit("@", 1)
    url = u
if "ast" in req.extras or "all" in req.extras:
    torch_extra.append("torchaudio")
print(f"FW_SPEC={shlex.quote(spec)}")
print(f"FW_EXTRAS={shlex.quote(','.join(sorted(req.extras)))}")
print(f"FW_URL={shlex.quote(url)}")
print(f"FW_REF={shlex.quote(ref)}")
print(f"OTHER_DEPS={shlex.quote(' '.join(shlex.quote(d) for d in others))}")
print(f"TORCH_EXTRA={shlex.quote(' '.join(sorted({Requirement(d).name for d in torch_extra})))}")
PYEOF
)" || die "Could not read the dependencies from pyproject.toml (see above)."
eval "$PARSED"

# ── 3. PyTorch ───────────────────────────────────────────────────────────────
step "PyTorch"
TORCH_PKGS="torch $TORCH_EXTRA"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
missing=""
for pkg in $TORCH_PKGS; do "$PY" -c "import $pkg" 2>/dev/null || missing="$missing $pkg"; done
if [ "$TORCH" = skip ] || [ -z "$missing" ]; then
  ok "using the installed torch $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '(none)')"
else
  if [ "$TORCH" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then TORCH=cuda
    elif [ "$(uname -s)" = Darwin ]; then TORCH=cuda   # the standard build; it includes MPS
    else TORCH=cpu; fi
  fi
  # shellcheck disable=SC2086  # word splitting of the package list is intended
  if [ "$TORCH" = cpu ]; then
    echo "  installing the CPU-only build of:$missing (no NVIDIA GPU found)"
    if ! "$PY" -m pip install --quiet --timeout 30 --retries 1 $TORCH_PKGS --index-url "$CPU_INDEX" >/dev/null 2>&1; then
      warn "the CPU package index could not be reached, using the standard build from PyPI (larger download)"
      "$PY" -m pip install --quiet $TORCH_PKGS || die "Could not install PyTorch."
    fi
  else
    echo "  installing the standard build from PyPI (CUDA on Linux, MPS on macOS):$missing"
    "$PY" -m pip install --quiet $TORCH_PKGS || die "Could not install PyTorch."
  fi
  ok "torch $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
fi

# ── 4. The framework ─────────────────────────────────────────────────────────
step "Audio Ops Framework [${FW_EXTRAS}]${FW_REF:+ at $FW_REF}"

reachable() {
  git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=20 ls-remote --exit-code "$1" >/dev/null 2>&1
}
ssh_url() {   # https://github.com/OWNER/REPO(.git) -> ssh://git@github.com/OWNER/REPO.git
  local rest="${1#https://}"; rest="${rest%.git}"
  echo "ssh://git@${rest%%/*}/${rest#*/}.git"
}
repo_name() { local n="${FW_URL%.git}"; echo "${n##*/}"; }

install_url() {   # $1 = git URL without git+ / @ref
  "$PY" -m pip install --quiet "audio-ops-framework[${FW_EXTRAS}] @ git+$1${FW_REF:+@$FW_REF}"
}
install_path() {  # $1 = local checkout
  [ -f "$1/pyproject.toml" ] || die "$1 is not a framework checkout (no pyproject.toml)."
  local have
  have="$(sed -n 's/^version = "\(.*\)"/\1/p' "$1/pyproject.toml" | head -n1)"
  if [ -n "$FW_REF" ] && [ "v$have" != "$FW_REF" ] && [ "$have" != "$FW_REF" ]; then
    warn "the checkout in $1 is version $have; pyproject.toml asks for $FW_REF"
  fi
  "$PY" -m pip install --quiet -e "$1[${FW_EXTRAS}]"
}

if ! command -v git >/dev/null 2>&1 && [ -n "$FW_URL" ] && [ ! -d "$FRAMEWORK" ]; then
  die "git is not installed; it is needed to download the framework.
       Install it (sudo apt install git, or brew install git) and run this script again."
fi

if [ -z "$FW_URL" ] && [ "$FRAMEWORK" = auto ]; then
  echo "  installing from the package index: $FW_SPEC"
  "$PY" -m pip install --quiet "$FW_SPEC" || die "Installing the framework failed."
elif [ -d "$FRAMEWORK" ]; then
  echo "  installing your local checkout $FRAMEWORK (editable)"
  install_path "$FRAMEWORK" || die "Installing the framework from $FRAMEWORK failed."
else
  SSH_URL=""
  case "$FW_URL" in https://*) SSH_URL="$(ssh_url "$FW_URL")" ;; esac
  SIBLING=""
  for d in "../$(repo_name)" "../audio-ops-framework"; do
    [ -f "$d/pyproject.toml" ] && grep -q '^name = "audio-ops-framework"' "$d/pyproject.toml" && { SIBLING="$d"; break; }
  done
  done_fw=0
  case "$FRAMEWORK" in
    auto|https)
      echo "  trying $FW_URL"
      if reachable "$FW_URL"; then install_url "$FW_URL" && done_fw=1 || die "Installing the framework failed (see above)."
      elif [ "$FRAMEWORK" = https ]; then
        die "Cannot reach $FW_URL. If the repository is private, use --framework ssh."
      else echo "  not reachable over HTTPS (private repository or no network)"; fi ;;
  esac
  if [ "$done_fw" = 0 ] && { [ "$FRAMEWORK" = auto ] || [ "$FRAMEWORK" = ssh ]; } && [ -n "$SSH_URL" ]; then
    echo "  trying $SSH_URL"
    if reachable "$SSH_URL"; then install_url "$SSH_URL" && done_fw=1 || die "Installing the framework failed (see above)."
    elif [ "$FRAMEWORK" = ssh ]; then
      die "Cannot reach $SSH_URL over SSH. Check that 'ssh -T git@github.com' greets you
       by name and that your GitHub account can read the repository."
    else echo "  not reachable over SSH (no SSH key on GitHub, or no access)"; fi
  fi
  if [ "$done_fw" = 0 ] && [ "$FRAMEWORK" = auto ] && [ -n "$SIBLING" ]; then
    echo "  using the framework checkout next to this repo: $SIBLING (editable)"
    install_path "$SIBLING" && done_fw=1
  fi
  [ "$done_fw" = 1 ] || die "Could not get the framework. Do one of the following and run this script again:
       - get read access to ${FW_URL:-the framework repository} and add an SSH key to
         your GitHub account (https://github.com/settings/keys), then: --framework ssh
       - or clone the framework next to this repo:
           git clone ${SSH_URL:-$FW_URL} ../$(repo_name)
       - or point to a checkout: ./scripts/setup.sh --framework /path/to/framework
       If your SSH key has a passphrase, load it first: eval \"\$(ssh-agent -s)\" && ssh-add"
fi
ok "audio-ops $("$VENV/bin/audio-ops" version 2>/dev/null | head -n1)"

# ── 5. Other dependencies ────────────────────────────────────────────────────
if [ -n "$OTHER_DEPS" ] || [ -d tests ]; then
  step "Other dependencies"
  extra=""; [ -d tests ] && extra="pytest"
  # shellcheck disable=SC2086
  eval "\"$PY\" -m pip install --quiet $OTHER_DEPS $extra" || die "Installing the dependencies failed."
  ok "${OTHER_DEPS:-none}${extra:+ (+ $extra for the tests)}"
fi

# ── 6. Folders and .env ──────────────────────────────────────────────────────
step "Runtime folders and .env"
mkdir -p storage audio_input dagster_home mlruns tasks
if [ -f .env ]; then ok ".env already exists, left unchanged"
elif [ -f .env.example ]; then cp .env.example .env; ok ".env created from .env.example"
else ok "no .env.example, nothing to copy"; fi

# ── 7. System tools ──────────────────────────────────────────────────────────
step "System tools"
if command -v ffmpeg >/dev/null 2>&1; then ok "ffmpeg found"
else warn "ffmpeg not found. It is only needed for live RTSP streams. Install it with
       'sudo apt install ffmpeg' (Linux/WSL) or 'brew install ffmpeg' (macOS)."; fi

# ── 8. Tasks and data ────────────────────────────────────────────────────────
step "Tasks and training audio"
export AUDIO_OPS_DISABLE_BUILTIN_TASKS="${AUDIO_OPS_DISABLE_BUILTIN_TASKS:-true}"
"$VENV/bin/audio-ops" validate || warn "a task config has errors (see above)"
"$PY" - <<'PYEOF'
from pathlib import Path
import yaml
from audio_ops_dagster.core.runtime import AUDIO_EXTENSIONS as AUDIO
tasks = sorted(p.parent for p in Path("tasks").glob("*/config.yaml"))
if not tasks:
    print("  no tasks yet. Create one with: audio-ops new-task my_task --config-only --classes a,b")
for t in tasks:
    cfg = yaml.safe_load(t.joinpath("config.yaml").read_text()) or {}
    if not {"train", "evaluate"} & set(cfg.get("capabilities", [])) and \
            cfg.get("archetype", "trainable_model") not in ("trainable_model", "eval_only"):
        continue
    root = Path(cfg.get("input_dir") or f"audio_input/{t.name}")
    root.mkdir(parents=True, exist_ok=True)
    counts = {d.name: sum(1 for f in d.rglob("*") if f.suffix.lower() in AUDIO)
              for d in sorted(root.iterdir()) if d.is_dir()}
    labels = cfg.get("class_labels") or list(counts)
    total = sum(counts.values())
    if total == 0:
        print(f"  {t.name}: no audio yet. Put files in {root}/<label>/ "
              f"(labels: {', '.join(map(str, labels)) or 'one folder per label'})")
    else:
        print(f"  {t.name}: {total} files in {root} "
              f"({', '.join(f'{k} {v}' for k, v in counts.items())})")
        empty = [str(lbl) for lbl in labels if counts.get(str(lbl), 0) == 0]
        if empty:
            print(f"    warning: no files for {', '.join(empty)}")
PYEOF

if [ "$VENV" = .venv ]; then how="The make targets use .venv automatically:"
else how="Activate $VENV first (the make targets only find .venv on their own):"; fi
cat <<EOF

${G}${B}Setup complete.${N}

$how
    make train              train, evaluate and score
    make predict FILE=x.wav score a file
    make serve              inference API on http://localhost:8000/docs
    make dev                Dagster UI on http://localhost:3000
    make test               check the tasks against the installed framework

Or activate the environment and use audio-ops directly:
    source $VENV/bin/activate
EOF
