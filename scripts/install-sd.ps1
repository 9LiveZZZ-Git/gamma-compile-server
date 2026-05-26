# gamma-compile-server -- install / setup the Stable Diffusion stack used
# by the /sprite-gen route. Windows / PowerShell version of install-sd.sh.
#
# What this does:
#   1. Creates a Python venv at  models\sd-venv\        (next to the model)
#   2. Installs Python deps      torch, diffusers, etc.
#   3. Downloads model weights   Z-Image-Turbo  -> models\z-image-turbo\
#   4. Downloads LoRA            pixel-art      -> models\loras\pixel-art\
#   5. Self-tests the install    runs the worker with a dummy prompt
#
# Usage (from an elevated PowerShell in the gamma-compile-server checkout):
#   .\scripts\install-sd.ps1                   # default: z-image-turbo
#   .\scripts\install-sd.ps1 sdxl              # SDXL + Pixel Art XL LoRA
#   .\scripts\install-sd.ps1 flux2-klein       # Flux.2 klein (no LoRA yet)
#
# Pre-requisites (install once):
#   - Python 3.10+ (https://www.python.org/downloads/windows/)
#       During install, check "Add Python to PATH".
#   - Git for Windows (https://git-scm.com/download/win)  -- needed by
#       `pip install git+...` for the diffusers main branch.
#   - 15 GB free disk space (model + venv).
#
# GPU notes:
#   - NVIDIA: pytorch with CUDA support will be installed automatically
#     if you have a recent NVIDIA driver. Generation will be much faster
#     than CPU (a few seconds vs minutes).
#   - AMD / Intel iGPU: CPU fallback. SDXL on CPU is impractical; pick
#     z-image-turbo or use WSL2 + ROCm.
#   - Apple Silicon: this script is the wrong tool. Use install-sd.sh
#     under WSL2... or actually just use a Mac. :)

$ErrorActionPreference = "Stop"

# ----------------------------------------------------------------------
$Model = if ($args.Count -ge 1) { $args[0] } else { "z-image-turbo" }

$ScriptDir = Split-Path -Parent $PSCommandPath
$ServerDir = Split-Path -Parent $ScriptDir
# Honor $env:GAMMA_MODELS_DIR for installs OUTSIDE the server checkout
# (e.g. when the checkout lives in OneDrive but model weights need to
# live on a different drive). Default: <server>/models.
if ($env:GAMMA_MODELS_DIR) {
  $ModelsDir = $env:GAMMA_MODELS_DIR
  Write-Host "[install-sd] using GAMMA_MODELS_DIR=$ModelsDir" -ForegroundColor Cyan
} else {
  $ModelsDir = Join-Path $ServerDir "models"
}
$VenvDir   = Join-Path $ModelsDir "sd-venv"
$LoraDir   = Join-Path $ModelsDir "loras"

function Say($msg)  { Write-Host "[install-sd] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  v $msg"          -ForegroundColor Green }
function Warn($msg) { Write-Host "  ! $msg"          -ForegroundColor Yellow }
function Die($msg)  { Write-Host "  x $msg"          -ForegroundColor Red; exit 1 }

# ---------- Pre-flight ----------
Say "checking prerequisites"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $python) { Die "Python not found in PATH. Install Python 3.10+ from python.org and re-run." }

$pyVer = & $python.Path -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))"
Say "  python = $pyVer  ($($python.Path))"
$verParts = $pyVer.Split('.')
if ([int]$verParts[0] -lt 3 -or ([int]$verParts[0] -eq 3 -and [int]$verParts[1] -lt 10)) {
  Die "Python 3.10+ required (you have $pyVer)"
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Warn "git not in PATH. diffusers main-branch install will fail. Install Git for Windows."
}

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LoraDir   | Out-Null

# ---------- 1. venv ----------
Say "step 1/5  Python venv at $VenvDir"
if (Test-Path $VenvDir) {
  Ok "venv already exists -- reusing"
} else {
  & $python.Path -m venv $VenvDir
  if ($LASTEXITCODE -ne 0) { Die "venv creation failed" }
  Ok "created"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip    = Join-Path $VenvDir "Scripts\pip.exe"
$VenvHf     = Join-Path $VenvDir "Scripts\huggingface-cli.exe"
if (-not (Test-Path $VenvPython)) { Die "venv python missing at $VenvPython" }

& $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { Die "pip upgrade failed" }

# ---------- 2. Python deps ----------
Say "step 2/5  installing Python deps (this can take 5-15 minutes)"
# NVIDIA users: this picks up CUDA wheels automatically when the
# corresponding CUDA toolkit is detected.
$pyPkgs = @(
  "torch",
  "torchvision",
  # Z-Image-Turbo lives in diffusers main. Pull from git so the
  # ZImagePipeline class is available. Requires git in PATH.
  "git+https://github.com/huggingface/diffusers.git",
  "transformers",
  "safetensors",
  "accelerate",
  "huggingface_hub[cli]",
  "Pillow",
  "sentencepiece"
)
& $VenvPip install --quiet @pyPkgs
if ($LASTEXITCODE -ne 0) { Die "pip install failed (see output above)" }
Ok "deps installed"

# ---------- 3. Model weights ----------
Say "step 3/5  downloading model: $Model"
switch ($Model) {
  "z-image-turbo" {
    $modelRepo  = "Tongyi-MAI/Z-Image-Turbo"
    $modelCache = Join-Path $ModelsDir "z-image-turbo"
  }
  "flux2-klein" {
    $modelRepo  = "black-forest-labs/FLUX.2-klein"
    $modelCache = Join-Path $ModelsDir "flux2-klein"
  }
  "sdxl" {
    $modelRepo  = "stabilityai/stable-diffusion-xl-base-1.0"
    $modelCache = Join-Path $ModelsDir "sdxl"
  }
  default { Die "unknown model: $Model  (known: z-image-turbo, flux2-klein, sdxl)" }
}
New-Item -ItemType Directory -Force -Path $modelCache | Out-Null
Say "  HF repo: $modelRepo"
Say "  cache:   $modelCache"
$installedMarker = Join-Path $modelCache ".installed"
if (Test-Path $installedMarker) {
  Ok "model already downloaded -- skipping (delete $installedMarker to force)"
} else {
  & $VenvHf download $modelRepo --local-dir $modelCache --local-dir-use-symlinks False
  if ($LASTEXITCODE -ne 0) {
    Warn "HF download failed. Common causes:"
    Warn "  - need to accept the model license: visit https://huggingface.co/$modelRepo"
    Warn "  - need to run 'huggingface-cli login' if the model is gated"
    Die "model download failed"
  }
  New-Item -ItemType File -Path $installedMarker | Out-Null
  Ok "downloaded"
}

# ---------- 4. Pixel Art LoRA ----------
Say "step 4/5  downloading Pixel Art LoRA"
$loraRepo = "nerijs/pixel-art-xl"
$loraFile = "pixel-art-xl-v1.1.safetensors"
$loraPath = Join-Path $LoraDir $loraFile
if (Test-Path $loraPath) {
  Ok "LoRA already present -- skipping"
} else {
  & $VenvHf download $loraRepo $loraFile --local-dir $LoraDir --local-dir-use-symlinks False
  if ($LASTEXITCODE -ne 0) { Die "LoRA download failed" }
  Ok "downloaded to $loraPath"
}

# ---------- 5. Self-test ----------
Say "step 5/5  self-test (loads model + generates a 256x256 test image)"
$workerPy = Join-Path $ServerDir "src\sd-worker.py"
$testOut  = Join-Path $ModelsDir "self-test.png"
if (Test-Path $testOut) { Remove-Item $testOut }
& $VenvPython $workerPy --self-test --model $Model --out $testOut
if ($LASTEXITCODE -ne 0) {
  Warn "self-test failed -- diffusion may still work, but inspect the output above"
} else {
  if (Test-Path $testOut) {
    $sz = (Get-Item $testOut).Length
    Ok "self-test passed -- wrote $testOut ($sz bytes)"
  } else {
    Warn "self-test returned 0 but wrote no file"
  }
}

Say ""
Say "done"
Say "next:"
Say "  - start the compile-server normally:  npm start"
Say "  - the /sprite-gen route will spawn the worker on first request"
Say "  - SpriteCreator's 'Compile-server SD' backend in the editor will hit it"
