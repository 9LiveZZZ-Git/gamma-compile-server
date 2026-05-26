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
import atexit
import base64
import io
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

def _pid_file_for(socket_path):
    """Where the worker writes its PID. Adjacent to the socket so
    sd-route.js can find it via the same path convention. On Windows
    (TCP) the socket_path isn't a file, so we use a fixed temp path."""
    if socket_path.startswith("tcp:"):
        return os.path.join(
            os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp",
            "gamma-sd-worker.pid"
        )
    return socket_path + ".pid"

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
    # bfloat16 across the board: Z-Image was trained in bf16 and its
    # DiT transformer is fragile in fp16 (attention Q@K^T accumulates
    # NaN over many steps -> black output). PyTorch 2.5+ supports bf16
    # on MPS so the old "use fp16 on MPS" workaround is obsolete.
    import torch
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        # Older diffusers without ZImagePipeline → fall back to AutoPipeline
        # which may still work via the model's pipeline_class hint.
        from diffusers import AutoPipelineForText2Image as ZImagePipeline
    dtype = torch.bfloat16
    pipe = ZImagePipeline.from_pretrained(
        str(local_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _upcast_vae_for_stability(pipe, device)
    return pipe

def _make_sdxl(local_dir, device):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(local_dir),
        torch_dtype=_torch_dtype_for(device)
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _upcast_vae_for_stability(pipe, device)
    return pipe

def _upcast_vae_for_stability(pipe, device):
    """Force the VAE decoder to float32 on MPS.

    The MPS backend's float16 ops produce NaN in the VAE decode path
    on Apple Silicon (most visible at the final image_processor cast:
    `RuntimeWarning: invalid value encountered in cast` followed by a
    flat black/uniform output). The transformer/UNet can stay fp16
    for speed; only the final decode needs fp32, and it's a one-shot
    pass so the perf hit is small.

    Also try to set force_upcast on the VAE config when available;
    some pipelines (SDXL especially) honor that flag and upcast just
    for the decode pass even if the VAE module itself is fp16. On
    other pipelines that flag is a no-op, so we cast the module too.
    """
    import torch
    if device != "mps":
        return  # CUDA + CPU don't have the MPS-fp16 VAE NaN issue
    if not hasattr(pipe, "vae") or pipe.vae is None:
        return
    try:
        pipe.vae = pipe.vae.to(dtype=torch.float32)
        if hasattr(pipe.vae, "config") and hasattr(pipe.vae.config, "force_upcast"):
            pipe.vae.config.force_upcast = True
        print("[sd-worker] VAE upcast to float32 (MPS NaN workaround)", flush=True)
    except Exception as e:
        print(f"[sd-worker] VAE upcast failed (will rely on nan_to_num): {e}", flush=True)

def _make_flux2(local_dir, device):
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        from diffusers import AutoPipelineForText2Image as Flux2KleinPipeline
    import torch
    dtype = torch.bfloat16 if device != "mps" else torch.float16
    pipe = Flux2KleinPipeline.from_pretrained(
        str(local_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _upcast_vae_for_stability(pipe, device)
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
        # Wrap both the load AND fuse in try/except so an install missing
        # the `peft` package (required by diffusers since v0.27 for LoRA
        # ops) still produces a sprite -- just without the LoRA. The
        # user sees a one-line warning instead of a 500.
        try:
            self.pipe.load_lora_weights(str(lora_path))
        except Exception as e:
            print(f"[sd-worker] LoRA load failed for {lora_name}: {e}", flush=True)
            print(f"[sd-worker]   tip: run `pip install peft` inside the sd-venv", flush=True)
            self.current_lora = None
            return
        try:
            # Newer diffusers: set adapter strength via fuse_lora.
            self.pipe.fuse_lora(lora_scale=strength)
        except Exception as e:
            # fuse_lora is optional; the LoRA may still apply via the
            # default adapter weight even if fuse fails.
            print(f"[sd-worker] fuse_lora warning (LoRA still bound): {e}", flush=True)
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
        # Step-by-step progress callback. Prints a single line to stdout
        # for each step; sd-route.js parses these and exposes them via
        # GET /sprite-gen/progress so the browser can drive a progress bar.
        # Two pipeline APIs in the wild:
        #   - newer: callback_on_step_end(pipe, step, timestep, kwargs)
        #   - legacy: callback(step, timestep, latents)
        # Detect by inspecting the first positional arg.
        print(f"[progress] step=0 total={steps} elapsed_ms=0", flush=True)
        # *args because the two diffusers callback APIs differ in arity:
        #   - newer: (pipe, step_index, timestep, callback_kwargs)  -> 4 positional
        #   - legacy: (step_index, timestep, latents)                -> 3 positional
        # The step index is args[1] in the newer API (after pipe) or
        # args[0] in the legacy one. Detect by length, not by attribute
        # probing -- the 4-arg case may pass `pipe=None` in some edge
        # paths (custom pipelines) so hasattr() isn't reliable.
        def progress_cb(*args, **kwargs):
            if len(args) >= 4:
                step_idx = int(args[1]) if args[1] is not None else 0
            elif len(args) >= 1:
                step_idx = int(args[0]) if args[0] is not None else 0
            else:
                step_idx = 0
            elapsed_ms = int((time.time() - t0) * 1000)
            print(f"[progress] step={step_idx + 1} total={steps} elapsed_ms={elapsed_ms}",
                  flush=True)
            return kwargs or {}

        pipe_kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if not self.model_name.startswith("flux"):
            pipe_kwargs["negative_prompt"] = negative
        with torch.inference_mode():
            # Try newer API first; fall back to legacy if the pipeline
            # rejects the kwarg. Some custom pipelines don't accept either,
            # in which case the user just sees no per-step updates (the
            # bar still shows elapsed-time via the browser timer).
            # Ask the pipeline for raw numpy output (float [0,1]) so we
            # can sanitize NaN/Inf BEFORE the (images*255).astype(uint8)
            # step that triggers the "invalid value encountered in cast"
            # RuntimeWarning + produces a black square. This is a safety
            # net on top of the bf16 + VAE upcast fixes -- if either of
            # those slips, the user still gets a usable (if noisy)
            # image instead of garbage.
            pipe_kwargs["output_type"] = "np"
            try:
                out = self.pipe(callback_on_step_end=progress_cb, **pipe_kwargs)
            except TypeError as e1:
                if "callback_on_step_end" not in str(e1):
                    raise
                try:
                    out = self.pipe(callback=progress_cb, callback_steps=1, **pipe_kwargs)
                except TypeError as e2:
                    if "callback" not in str(e2):
                        raise
                    out = self.pipe(**pipe_kwargs)
        # out.images is a numpy array of shape (N, H, W, 3) in [0,1].
        import numpy as np
        from PIL import Image
        arr = np.asarray(out.images[0], dtype=np.float32)
        bad = int(np.sum(~np.isfinite(arr)))
        if bad > 0:
            total = int(arr.size)
            print(f"[sd-worker] WARN: {bad}/{total} non-finite output values "
                  f"({100.0*bad/total:.1f}%); clamping to mid-gray", flush=True)
            arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255.0).round().astype(np.uint8))
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
        # SO_REUSEADDR so a fresh server can bind even if the kernel
        # still has the old socket in TIME_WAIT after an abrupt exit.
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, int(port)))
    srv.listen(4)
    # PID file lets sd-route.js kill any stale worker on its own
    # restart. We register atexit cleanup so the file goes away on a
    # clean exit; the new server will overwrite it on dirty exit.
    pid_file = _pid_file_for(socket_path)
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        def _cleanup_pid():
            try: os.remove(pid_file)
            except Exception: pass
            if is_unix:
                try: os.remove(socket_path)
                except Exception: pass
        atexit.register(_cleanup_pid)
    except Exception as e:
        print(f"[sd-worker] WARN pid-file write failed: {e}", flush=True)
    print(f"[sd-worker] listening on {socket_path} (pid={os.getpid()})", flush=True)

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
    use_lora = not model_name.startswith("flux")
    req = {
        "prompt": "pixel art, small red fox, simple, side view",
        "negative": "blurry, photo",
        "width": 256, "height": 256,
        "steps": 6, "guidance": 4.0, "seed": 1234,
        "lora": "pixel-art-xl.safetensors" if use_lora else None,
        "lora_strength": 1.0
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
