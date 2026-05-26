#!/usr/bin/env bash
# gamma-compile-server -- install / setup the Stable Diffusion stack used
# by the /sprite-gen route (SpriteCreator-2 sprint).
#
# What this does:
#   1. Creates a Python venv at  models/sd-venv/         (next to the model)
#   2. Installs Python deps      torch, diffusers, mlx, etc.
#   3. Downloads model weights   Z-Image-Turbo  → models/z-image-turbo/
#   4. Downloads LoRA            pixel-art      → models/loras/pixel-art/
#   5. Self-tests the install    runs the worker with a dummy prompt
#
# Targets macOS (Apple Silicon recommended; uses MPS / MLX where possible)
# and Linux. Windows is not supported by this script -- WSL2 is the path.
#
# Adding more models later (e.g. Flux.2 klein):
#   - Add a new branch in pull_model() with the HF repo name + cache dir.
#   - Add a corresponding entry in src/sd-route.js's MODELS map.
#   - Re-run this script with the model name as an argument:
#       ./scripts/install-sd.sh flux2-klein
#
# Default behavior (no args) installs Z-Image-Turbo + Pixel Art LoRA.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"
# Honor $GAMMA_MODELS_DIR for installs OUTSIDE the server checkout (e.g.
# server lives in iCloud/OneDrive but model weights need a separate
# drive or non-synced location). Default: <server>/models.
if [[ -n "${GAMMA_MODELS_DIR:-}" ]]; then
  MODELS_DIR="$GAMMA_MODELS_DIR"
  echo "[install-sd] using GAMMA_MODELS_DIR=$MODELS_DIR"
else
  MODELS_DIR="$SERVER_DIR/models"
fi
VENV_DIR="$MODELS_DIR/sd-venv"
LORA_DIR="$MODELS_DIR/loras"

# ── Color helpers (skip if not a TTY) ────────────────────────────────
if [[ -t 1 ]]; then
  G='\033[0;32m'; Y='\033[0;33m'; R='\033[0;31m'; B='\033[0;36m'; N='\033[0m'
else
  G=''; Y=''; R=''; B=''; N=''
fi
say()  { printf "${B}[install-sd]${N} %s\n" "$*"; }
ok()   { printf "${G}  ✓${N} %s\n" "$*"; }
warn() { printf "${Y}  ⚠${N} %s\n" "$*"; }
die()  { printf "${R}  ✗${N} %s\n" "$*" >&2; exit 1; }

# ── Pre-flight ───────────────────────────────────────────────────────
say "checking prerequisites"
OS="$(uname -s)"
ARCH="$(uname -m)"
say "  OS=$OS  arch=$ARCH"
if [[ "$OS" == "Darwin" && "$ARCH" != "arm64" ]]; then
  warn "Apple Silicon (arm64) strongly recommended; Intel Mac will be slow."
fi
if [[ "$OS" != "Darwin" && "$OS" != "Linux" ]]; then
  die "unsupported OS: $OS (use WSL2 on Windows)"
fi

command -v python3 >/dev/null || die "python3 not found; install Python 3.10+"
PY_VER="$(python3 -c 'import sys; print(".".join(str(x) for x in sys.version_info[:2]))')"
say "  python=$PY_VER"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER##*.}"
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
  die "python 3.10+ required (you have $PY_VER)"
fi

command -v curl >/dev/null || die "curl not found"

# huggingface_hub provides huggingface-cli; we install it INTO the venv so
# users don't need a global install. The pip path inside the venv handles it.

mkdir -p "$MODELS_DIR" "$LORA_DIR"

# ── 1. Python venv ───────────────────────────────────────────────────
say "step 1/5  Python venv at $VENV_DIR"
if [[ -d "$VENV_DIR" ]]; then
  ok "venv already exists -- reusing"
else
  python3 -m venv "$VENV_DIR"
  ok "created"
fi

# Activate (subshell so we don't pollute caller's env beyond this script)
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel --quiet

# ── 2. Python dependencies ───────────────────────────────────────────
say "step 2/5  installing Python deps (this can take a few minutes)"
# Apple Silicon path: torch with MPS support is the default wheel.
# We pin nothing strictly; latest versions of diffusers + torch are
# what works best on M-series. Add safetensors + accelerate for fast
# model loading + transformers for tokenizer support.
PY_PKGS=(
  "torch"
  "torchvision"
  # Z-Image-Turbo's pipeline lives in diffusers main (added Jan 2026 per
  # the model card). pypi release may lag; pull from git to be safe.
  "git+https://github.com/huggingface/diffusers.git"
  "transformers"
  "safetensors"
  "accelerate"
  "huggingface_hub[cli]"
  "Pillow"
  "sentencepiece"
)
# MLX is Apple-only and currently optional (Z-Image-Turbo runs fine on
# diffusers+MPS without it). Skip on Linux.
if [[ "$OS" == "Darwin" ]]; then
  PY_PKGS+=("mlx" "mlx-lm")
fi
python -m pip install --quiet "${PY_PKGS[@]}" || die "pip install failed (see output above)"
ok "deps installed"

# ── 3. Model weights ─────────────────────────────────────────────────
MODEL_NAME="${1:-z-image-turbo}"
say "step 3/5  downloading model: $MODEL_NAME"

pull_model() {
  case "$1" in
    z-image-turbo)
      MODEL_REPO="Tongyi-MAI/Z-Image-Turbo"
      MODEL_CACHE="$MODELS_DIR/z-image-turbo"
      ;;
    flux2-klein)
      MODEL_REPO="black-forest-labs/FLUX.2-klein"
      MODEL_CACHE="$MODELS_DIR/flux2-klein"
      ;;
    sdxl)
      MODEL_REPO="stabilityai/stable-diffusion-xl-base-1.0"
      MODEL_CACHE="$MODELS_DIR/sdxl"
      ;;
    *)
      die "unknown model: $1  (known: z-image-turbo, flux2-klein, sdxl)"
      ;;
  esac
  mkdir -p "$MODEL_CACHE"
  # huggingface-cli download writes into a HF cache layout under
  # $MODEL_CACHE; the worker loads via from_pretrained pointing at this
  # dir, so the cache layout doesn't matter as long as it stays stable.
  say "  HF repo: $MODEL_REPO"
  say "  cache:   $MODEL_CACHE"
  if [[ -f "$MODEL_CACHE/.installed" ]]; then
    ok "model already downloaded -- skipping (delete $MODEL_CACHE/.installed to force)"
    return 0
  fi
  if ! huggingface-cli download "$MODEL_REPO" \
        --local-dir "$MODEL_CACHE" \
        --local-dir-use-symlinks False; then
    warn "HF download failed. Common causes:"
    warn "  - need to accept the model license: visit https://huggingface.co/$MODEL_REPO"
    warn "  - need to run 'huggingface-cli login' if the model is gated"
    die "model download failed"
  fi
  touch "$MODEL_CACHE/.installed"
  ok "downloaded"
}
pull_model "$MODEL_NAME"

# ── 4. Pixel Art LoRA ────────────────────────────────────────────────
say "step 4/5  downloading Pixel Art LoRA"
LORA_REPO="nerijs/pixel-art-xl"          # SDXL-trained; works with Z-Image base too in practice.
LORA_FILE="pixel-art-xl-v1.1.safetensors"
LORA_PATH="$LORA_DIR/$LORA_FILE"
if [[ -f "$LORA_PATH" ]]; then
  ok "LoRA already present -- skipping"
else
  huggingface-cli download "$LORA_REPO" "$LORA_FILE" \
      --local-dir "$LORA_DIR" \
      --local-dir-use-symlinks False \
    || die "LoRA download failed"
  ok "downloaded to $LORA_PATH"
fi

# ── 5. Self-test ─────────────────────────────────────────────────────
say "step 5/5  self-test (loads model + generates a 256x256 test image)"
TEST_OUT="$MODELS_DIR/self-test.png"
rm -f "$TEST_OUT"
if python "$SERVER_DIR/src/sd-worker.py" \
      --self-test \
      --model "$MODEL_NAME" \
      --out "$TEST_OUT"; then
  if [[ -f "$TEST_OUT" ]]; then
    SZ=$(wc -c < "$TEST_OUT")
    ok "self-test passed -- wrote $TEST_OUT ($SZ bytes)"
  else
    warn "self-test returned 0 but wrote no file"
  fi
else
  warn "self-test failed -- diffusion may still work, but inspect the output above"
fi

say ""
say "${G}done${N}"
say "next:"
say "  - start the compile-server normally:  npm start"
say "  - the /sprite-gen route will spawn the worker on first request"
say "  - SpriteCreator's 'Compile-server SD' backend in the editor will hit it"
