#!/usr/bin/env python3
"""
gamma-compile-server -- persistent Stable Diffusion worker.

Loads the model ONCE on startup, then serves generation requests over a
Unix socket (or TCP fallback on Windows). The compile-server's
/sprite-gen route is the only client.

Protocol (line-delimited JSON over the socket):
  request:  {"prompt": str, "negative": str, "width": int, "height": int,
             "steps": int, "guidance": float, "seed": int|null,
             "lora": str|null, "lora_strength": float}
  response: {"ok": true, "png_b64": str, "elapsed_ms": int}
        or  {"ok": false, "error": str}

Self-test mode (set --self-test) loads the model, runs one tiny gen,
writes the PNG to --out, exits 0 on success / non-zero on failure.

Adding a model later (e.g. flux2-klein):
  - extend MODEL_REGISTRY with the cache dir + pipeline factory
  - re-run install-sd.sh with the new --model arg
  - no worker code changes needed
"""
import argparse
import base64
import io
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

# ── Model registry ───────────────────────────────────────────────────
# Maps the --model arg to (HF-style local_dir, pipeline factory). Add new
# entries here when expanding to Flux.2 etc.
SCRIPT_DIR = Path(__file__).resolve().parent
# Honor GAMMA_MODELS_DIR so model weights can live off the server
# checkout (matches install-sd.sh / install-sd.ps1 / sd-route.js).
_env_models = os.environ.get("GAMMA_MODELS_DIR")
MODELS_DIR = Path(_env_models) if _env_models else (SCRIPT_DIR.parent / "models")
LORA_DIR   = MODELS_DIR / "loras"

def _make_z_image(local_dir, device):
    # Z-Image-Turbo uses its own pipeline class (added to diffusers main
    # branch Jan-2026). Requires `pip install git+https://github.com/huggingface/diffusers.git`
    # for now -- bake into install-sd.sh.
    # bfloat16 is recommended per the model card; MPS doesn't always
    # support bf16 cleanly so fall back to float16 there.
    import torch
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        # Older diffusers without ZImagePipeline → fall back to AutoPipeline
        # which may still work via the model's pipeline_class hint.
        from diffusers import AutoPipelineForText2Image as ZImagePipeline
    dtype = torch.bfloat16 if device != "mps" else torch.float16
    pipe = ZImagePipeline.from_pretrained(
        str(local_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def _make_sdxl(local_dir, device):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(local_dir),
        torch_dtype=_torch_dtype_for(device)
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def _make_flux2(local_dir, device):
    # Flux uses its own pipeline class; this branch is wired so installing
    # Flux.2 later is a one-line registry add, not a worker rewrite.
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        str(local_dir),
        torch_dtype=_torch_dtype_for(device)
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

MODEL_REGISTRY = {
    "z-image-turbo": (MODELS_DIR / "z-image-turbo", _make_z_image),
    "sdxl":          (MODELS_DIR / "sdxl",          _make_sdxl),
    "flux2-klein":   (MODELS_DIR / "flux2-klein",   _make_flux2),
}

def _torch_dtype_for(device):
    import torch
    # MPS on macOS likes float16 for SDXL-class models; CPU stays float32.
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.float16
    return torch.float32

def _detect_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ── Per-process model state ──────────────────────────────────────────
class Worker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = _detect_device()
        self.pipe = None
        self.current_lora = None
        self.current_lora_strength = 1.0

    def load(self):
        if self.pipe is not None:
            return
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model: {self.model_name} (known: {list(MODEL_REGISTRY)})")
        local_dir, factory = MODEL_REGISTRY[self.model_name]
        if not local_dir.exists():
            raise FileNotFoundError(
                f"model dir missing: {local_dir}\n"
                f"  run scripts/install-sd.sh first (with arg '{self.model_name}' for non-default models)"
            )
        t0 = time.time()
        print(f"[sd-worker] loading {self.model_name} from {local_dir} on {self.device}…",
              flush=True)
        self.pipe = factory(local_dir, self.device)
        print(f"[sd-worker] model ready in {time.time()-t0:.1f}s", flush=True)

    def apply_lora(self, lora_name, strength):
        if self.pipe is None:
            return
        # Only re-load when the requested LoRA differs from what's bound.
        if self.current_lora == lora_name and abs(self.current_lora_strength - strength) < 1e-3:
            return
        # Unload any previous LoRA (some diffusers versions expose this).
        if hasattr(self.pipe, "unload_lora_weights") and self.current_lora is not None:
            try: self.pipe.unload_lora_weights()
            except Exception: pass
        if not lora_name:
            self.current_lora = None
            return
        # File-based LoRA -- look up in models/loras/.
        lora_path = LORA_DIR / lora_name
        if not lora_path.exists() and not lora_name.endswith(".safetensors"):
            lora_path = LORA_DIR / (lora_name + ".safetensors")
        if not lora_path.exists():
            print(f"[sd-worker] LoRA not found: {lora_path} (skipping)", flush=True)
            self.current_lora = None
            return
        self.pipe.load_lora_weights(str(lora_path))
        try:
            # Newer diffusers: set adapter strength via fuse_lora.
            self.pipe.fuse_lora(lora_scale=strength)
        except Exception:
            pass
        self.current_lora = lora_name
        self.current_lora_strength = strength
        print(f"[sd-worker] LoRA loaded: {lora_name} @ {strength:.2f}", flush=True)

    def generate(self, req):
        self.load()
        prompt    = req.get("prompt", "")
        negative  = req.get("negative", "blurry, smooth, anti-aliased, low quality, watermark")
        width     = int(req.get("width",  512))
        height    = int(req.get("height", 512))
        steps     = int(req.get("steps",  20))
        guidance  = float(req.get("guidance", 7.0))
        seed      = req.get("seed", None)
        lora      = req.get("lora", None)
        lora_str  = float(req.get("lora_strength", 1.0))

        # Apply / swap LoRA before generation.
        self.apply_lora(lora, lora_str)

        import torch
        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        t0 = time.time()
        with torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
        img = out.images[0]
        elapsed_ms = int((time.time() - t0) * 1000)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return {"ok": True, "png_b64": b64, "elapsed_ms": elapsed_ms,
                "device": self.device, "model": self.model_name}

# ── Socket server ────────────────────────────────────────────────────
def serve(socket_path, model_name):
    worker = Worker(model_name)
    # Pre-load so the first /sprite-gen request doesn't pay the 30-60s
    # model-load cost. The HTTP client probably sets a longer timeout
    # but we still want the warm-start path to be the common case.
    try:
        worker.load()
    except Exception as e:
        print(f"[sd-worker] FATAL during preload: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # Bind socket. Unix on Mac/Linux; TCP fallback on Windows (untested).
    is_unix = not socket_path.startswith("tcp:")
    if is_unix:
        if os.path.exists(socket_path):
            os.remove(socket_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(socket_path)
        os.chmod(socket_path, 0o600)
    else:
        host, port = socket_path[len("tcp:"):].split(":")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((host, int(port)))
    srv.listen(4)
    print(f"[sd-worker] listening on {socket_path}", flush=True)

    while True:
        conn, _ = srv.accept()
        try:
            # Read one line of JSON.
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(8192)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 16 * 1024:
                    raise ValueError("request header too long")
            req = json.loads(buf.decode("utf-8").strip())
            try:
                resp = worker.generate(req)
            except Exception as e:
                resp = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
            payload = (json.dumps(resp) + "\n").encode("utf-8")
            conn.sendall(payload)
        except Exception as e:
            try:
                conn.sendall((json.dumps({"ok": False, "error": str(e)}) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()

# ── Self-test ────────────────────────────────────────────────────────
def self_test(model_name, out_path):
    worker = Worker(model_name)
    worker.load()
    req = {
        "prompt": "pixel art, small red fox, simple, side view",
        "negative": "blurry, photo",
        "width": 256, "height": 256,
        "steps": 6, "guidance": 4.0, "seed": 1234,
        "lora": "pixel-art-xl.safetensors", "lora_strength": 1.0
    }
    print(f"[sd-worker] self-test: {req['prompt']}", flush=True)
    resp = worker.generate(req)
    if not resp.get("ok"):
        print(f"[sd-worker] self-test FAILED: {resp.get('error')}", flush=True)
        sys.exit(2)
    Path(out_path).write_bytes(base64.b64decode(resp["png_b64"]))
    print(f"[sd-worker] self-test OK in {resp['elapsed_ms']}ms -> {out_path}", flush=True)

# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/gamma-sd-worker.sock",
                    help="Unix socket path, or 'tcp:HOST:PORT' for TCP")
    ap.add_argument("--model", default="z-image-turbo",
                    choices=list(MODEL_REGISTRY))
    ap.add_argument("--self-test", action="store_true",
                    help="run a one-shot generation and exit")
    ap.add_argument("--out", default="self-test.png",
                    help="output path for --self-test")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.model, args.out)
        return

    serve(args.socket, args.model)

if __name__ == "__main__":
    main()
