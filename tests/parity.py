"""Replay a reference capture (dev/capture.py of the research clone) through the pack and compare every tensor.

    CUDA_VISIBLE_DEVICES=1 python tests/parity.py <capture dir> [models dir]

RECAVSR_ATTENTION=default uses the attention backend ComfyUI would pick (xformers/sage...) instead of PyTorch SDPA.
RECAVSR_KERNELS=fast runs the fast kernels; RECAVSR_TRANSFORMER names another transformer file (the int8 variant);
RECAVSR_KV_OFFLOAD=1 keeps the attention history in RAM (must not change the output).
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFYUI", os.path.dirname(os.path.dirname(PACK)))
sys.path.insert(0, COMFY)
from comfy.cli_args import args  # noqa: E402

# the reference's cross-attention is plain SDPA; other backends differ by a bf16 ulp
args.use_pytorch_cross_attention = os.environ.get("RECAVSR_ATTENTION", "pytorch") == "pytorch"
import comfy.model_management  # noqa: E402

spec = importlib.util.spec_from_file_location("recavsr_pack", os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK])
pack = importlib.util.module_from_spec(spec)
sys.modules["recavsr_pack"] = pack
spec.loader.exec_module(pack)
from recavsr_pack.nodes import activation_bytes  # noqa: E402
from recavsr_pack.recavsr import loader  # noqa: E402
from recavsr_pack.recavsr.session import DecoderStream, VideoStream, wavelet_color_fix  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

capture = sys.argv[1]
models = sys.argv[2] if len(sys.argv) > 2 else os.path.join(COMFY, "models", "recavsr")
meta = json.load(open(os.path.join(capture, "meta.json")))
steps = load_file(os.path.join(capture, "steps.safetensors"))
reference_output = np.load(os.path.join(capture, "output.npy"))
reference_wavelet = np.load(os.path.join(capture, "wavelet.npy"))
frames = np.load(os.path.join(capture, "input.npy"))

kernels = os.environ.get("RECAVSR_KERNELS", "exact")
model = loader.load(*(os.path.join(models, f) for f in (os.environ.get("RECAVSR_TRANSFORMER", "transformer.safetensors"), "flashdecoder.safetensors", "prompt.safetensors")))
height, width, size = meta["H"], meta["W"], (meta["height"], meta["width"])
kv_offload = bool(os.environ.get("RECAVSR_KV_OFFLOAD"))
# the node's activation estimate, so ComfyUI leaves room for it and streams the weights on a small card
comfy.model_management.load_models_gpu(model.patchers(), memory_required=activation_bytes(height, width, kv_offload))
device = model.transformer.load_device


def report(name, ours, theirs):
    ours, theirs = ours.float().cpu(), theirs.float().cpu()
    diff = (ours - theirs).abs()
    print(f"{name}: max|diff| {diff.max():.3e}  mean {diff.mean():.3e}  exact {(diff == 0).float().mean() * 100:.2f}%  ref|max| {theirs.abs().max():.3e}")


with torch.inference_mode(), comfy.model_management.cuda_device_context(device):
    stream = VideoStream(model.transformer.model, model.prompt, height, width, 42, device, kernels,
                         torch.device("cpu") if kv_offload else None)
    decode = DecoderStream(model.decoder.model, model.latent_format, device, torch.bfloat16, kernels)
    output = np.empty_like(reference_output)
    wavelet = np.empty_like(reference_output)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    offset = 0
    for i in range(len([k for k in steps if k.endswith(".latent")])):
        lq = steps[f"step{i}.lq"].to(device)
        latent = stream.step(lq, noise=steps[f"step{i}.noise"].to(device))
        report(f"step {i} latent", latent, steps[f"step{i}.latent"])
        decoded = torch.cat(list(decode.step(latent, lq)), 2)
        report(f"step {i} decoded", decoded, steps[f"step{i}.decoded"])
        take = min(decoded.shape[2], len(frames) - offset)
        rgb = decoded[0, :, :take, :size[0], :size[1]].transpose(0, 1)
        output[offset:offset + take] = ((rgb.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        wavelet[offset:offset + take] = wavelet_color_fix(rgb, frames[offset:offset + take]).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        offset += take
    torch.cuda.synchronize()
    print(f"{time.perf_counter() - t0:.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (reference {meta['seconds']:.1f} s, {meta['peak_gib']:.2f} GiB)")
for name, ours, theirs in (("uint8 output", output, reference_output), ("wavelet output", wavelet, reference_wavelet)):
    diff = ours.astype(np.float64) - theirs.astype(np.float64)
    mse = (diff ** 2).mean()
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    print(f"{name}: max {np.abs(diff).max():.0f}  mean {np.abs(diff).mean():.4f}  identical pixels {(diff == 0).mean() * 100:.3f}%  PSNR {psnr:.1f} dB")
