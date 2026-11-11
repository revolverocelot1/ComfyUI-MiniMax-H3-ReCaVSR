from __future__ import annotations

import gc
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.sd
import comfy.utils
import comfy.model_management as model_management
import folder_paths


TAP = "down.1.block.1"
DEFAULT_TILE = 256
DEFAULT_OVERLAP = 64

# LRU cache for encoded reference images & reference videos in V2V (avoids re-encoding 100+ video frames on prompt/seed edits)
_REF_ENCODE_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_MAX_REF_CACHE_ITEMS = 6


def _die(msg):
    raise RuntimeError(str(msg))


def _empty_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tensor_of(o):
    if torch.is_tensor(o):
        return o
    if isinstance(o, (tuple, list)):
        for x in o:
            if torch.is_tensor(x):
                return x
    return None


def _fast_tensor_fingerprint(t: torch.Tensor) -> tuple:
    """Fast O(1) fingerprint of an image/video tensor without copying the full video."""
    shape = tuple(t.shape)
    dtype = str(t.dtype)
    try:
        flat = t.reshape(-1)
        n = flat.numel()
        if n == 0:
            return (shape, dtype, 0.0)
        # Sample 16 evenly spaced points across the video/image tensor
        idx = torch.linspace(0, n - 1, steps=min(16, n), device=flat.device).long()
        vals = tuple(round(float(x), 5) for x in flat[idx].detach().float().cpu().tolist())
        return (shape, dtype, vals)
    except Exception:
        return (shape, dtype, id(t))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class Down1DetailBranch(nn.Module):
    """Exact successful B32 v1 branch."""
    def __init__(self, bottleneck=32, width=64, max_delta=0.12):
        super().__init__()
        self.max_delta = float(max_delta)
        self.compress = nn.Conv2d(256, bottleneck, 1)
        self.inp = nn.Conv2d(bottleneck, width, 3, padding=1)
        self.b0 = ResBlock(width)
        self.b1 = ResBlock(width)
        self.b2 = ResBlock(width)
        self.b3 = ResBlock(width)
        self.out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, down1):
        x = F.silu(self.compress(down1))
        x = F.silu(self.inp(x))
        x = self.b0(x)
        x = self.b1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.b2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.b3(x)
        return torch.tanh(self.out(F.silu(x))) * self.max_delta


def _split_checkpoint(sd):
    e34 = {}
    detail = {}
    meta = {}
    for k, v in sd.items():
        if k.startswith("detail_b32.meta."):
            meta[k[len("detail_b32.meta."):]] = v
        elif k.startswith("detail_b32."):
            detail[k[len("detail_b32."):]] = v
        else:
            e34[k] = v
    return e34, detail, meta


def _bootstrap_x2_vae(e34_sd):
    """
    Build stock H3 architecture from the packaged checkpoint itself:
    temporarily present stock-sized proj_out, instantiate comfy.sd.VAE,
    replace proj_out with the real 12288-row X2 head, then load E34 exactly.
    """
    if "decoder.proj_out.weight" not in e34_sd:
        _die("Checkpoint has no decoder.proj_out.weight")
    w = e34_sd["decoder.proj_out.weight"]
    b = e34_sd.get("decoder.proj_out.bias")
    if tuple(w.shape) != (12288, 2048):
        _die(f"Expected X2 proj_out (12288,2048), got {tuple(w.shape)}")

    bootstrap = dict(e34_sd)
    bootstrap["decoder.proj_out.weight"] = w[:3072].contiguous()
    if b is not None:
        bootstrap["decoder.proj_out.bias"] = b[:3072].contiguous()

    vae = comfy.sd.VAE(sd=bootstrap)
    inner = vae.first_stage_model
    old = inner.decoder.proj_out
    inner.decoder.proj_out = nn.Linear(
        2048, 12288, bias=True,
        device=old.weight.device, dtype=old.weight.dtype
    )
    inner.decoder.out_channels = 12

    incompatible = inner.load_state_dict(e34_sd, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        _die(f"E34 state mismatch: missing={missing} unexpected={unexpected}")
    return vae


def _load_runtime(checkpoint_path):
    sd = comfy.utils.load_torch_file(str(checkpoint_path), safe_load=True)
    e34_sd, detail_sd, meta = _split_checkpoint(sd)

    if not detail_sd:
        # If user selected normal VAE (minimax_h3_video_vae_fp16.safetensors) in the Detailed Upscale node,
        # automatically check if MiniMax-H3-X2-Detail-v1.safetensors is available in models/vae/
        fallback = folder_paths.get_full_path("vae", "MiniMax-H3-X2-Detail-v1.safetensors")
        if fallback and Path(fallback).exists() and Path(fallback).resolve() != Path(checkpoint_path).resolve():
            print("[H3 X2 Detail] Normal VAE selected in Detailed Upscale node; auto-routing to MiniMax-H3-X2-Detail-v1.safetensors", flush=True)
            sd = comfy.utils.load_torch_file(str(fallback), safe_load=True)
            e34_sd, detail_sd, meta = _split_checkpoint(sd)
        if not detail_sd:
            _die("This node requires MiniMax-H3-X2-Detail-v1.safetensors (detail_b32.* tensors missing in selected VAE).")

    bottleneck = int(meta.get("bottleneck", torch.tensor([32])).flatten()[0].item())
    width = int(meta.get("width", torch.tensor([64])).flatten()[0].item())
    max_delta = float(meta.get("max_delta", torch.tensor([0.12])).flatten()[0].item())

    vae = _bootstrap_x2_vae(e34_sd)
    detail = Down1DetailBranch(bottleneck=bottleneck, width=width, max_delta=max_delta)
    detail.load_state_dict(detail_sd, strict=True)
    detail.eval()
    for p in detail.parameters():
        p.requires_grad_(False)

    return vae, detail


def _resolve_down1(vae):
    enc = vae.first_stage_model.encoder
    named = dict(enc.named_modules())
    if TAP not in named:
        _die(f"Cannot resolve encoder tap {TAP}")
    return named[TAP]


def _tile_starts(n, tile, overlap):
    if n <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, max(1, n - tile + 1), stride))
    last = n - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _prepare_encoder_once(vae):
    probe_shape = (1, 3, 1, DEFAULT_TILE, DEFAULT_TILE)
    memory_used = vae.memory_used_encode(probe_shape, vae.vae_dtype)
    model_management.load_models_gpu(
        [vae.patcher],
        memory_required=memory_used,
        force_full_load=vae.disable_offload,
    )


@torch.no_grad()
def _capture_one_tile_direct(vae, img_bchw, mod):
    """Capture down.1.block.1 without re-entering comfy.sd.VAE.encode()."""
    cap = []

    def hook(_m, _i, o):
        t = _tensor_of(o)
        if t is not None:
            cap.append(t.detach().float().cpu())

    inner = vae.first_stage_model
    h = mod.register_forward_hook(hook)
    try:
        pixels = img_bchw.unsqueeze(2)
        pixels = vae.process_input(pixels).to(device=vae.device, dtype=vae.vae_dtype)
        normalized = inner._normalize_pixels(pixels)
        _ = inner.encoder(normalized)
    finally:
        h.remove()

    if len(cap) != 1:
        _die(f"Expected one {TAP} capture for explicit tile, got {len(cap)}")
    f = cap[0]
    if f.ndim == 5:
        f = f[:, :, 0]
    if f.ndim != 4 or f.shape[1] != 256:
        _die(f"Unexpected down1 tile shape {tuple(f.shape)}")
    return f


@torch.no_grad()
def _stitched_down1(vae, img_bchw, mod, tile=256, overlap=64):
    B, C, H, W = img_bchw.shape
    if B != 1:
        _die("Internal frame path expects batch=1")
    if H < tile or W < tile:
        return _capture_one_tile_direct(vae, img_bchw, mod)

    ys = _tile_starts(H, tile, overlap)
    xs = _tile_starts(W, tile, overlap)
    out = torch.zeros((1, 256, H // 2, W // 2), dtype=torch.float32)
    wgt = torch.zeros((1, 1, H // 2, W // 2), dtype=torch.float32)

    for y in ys:
        for x in xs:
            patch = img_bchw[:, :, y:y + tile, x:x + tile]
            f = _capture_one_tile_direct(vae, patch, mod)
            fy, fx = y // 2, x // 2
            fh, fw = f.shape[-2:]
            out[:, :, fy:fy + fh, fx:fx + fw] += f
            wgt[:, :, fy:fy + fh, fx:fx + fw] += 1.0

    if float(wgt.min()) <= 0:
        _die("Feature stitching left uncovered pixels")
    return out / wgt


def _expand_rgb_stat(stat, packed_channels):
    if packed_channels == 3 or any(int(s) == packed_channels for s in stat.shape):
        return stat
    if packed_channels % 3:
        _die(f"Cannot expand RGB stat to {packed_channels}")
    dims = [i for i, n in enumerate(stat.shape) if int(n) == 3]
    if not dims:
        _die(f"RGB stat has no size-3 dimension: {tuple(stat.shape)}")
    d = 1 if stat.ndim > 1 and int(stat.shape[1]) == 3 else dims[0]
    return stat.repeat_interleave(packed_channels // 3, dim=d)


@torch.no_grad()
def _packed_to_images(packed, upscale, work_device):
    """
    packed [B, C, T, H, W] (CPU) -> ComfyUI IMAGE [B*T, H*up, W*up, C/up^2] float32 (CPU).

    Supports both:
    - upscale=2 (12-channel PixelShuffle 2X Detail VAE)
    - upscale=1 (3-channel Standard fp16 Video VAE)
    """
    B, C, T, H, W = packed.shape
    oc = C // (upscale * upscale)
    out = torch.empty((B * T, H * upscale, W * upscale, oc), dtype=torch.float32, device="cpu")
    bytes_per_frame = C * H * W * 4
    step = max(1, int((512 * 1024 * 1024) // max(1, bytes_per_frame)))
    for b in range(B):
        t0 = 0
        while t0 < T:
            t1 = min(T, t0 + step)
            chunk = packed[b, :, t0:t1]  # [C, t, H, W]
            if upscale == 1:
                # Standard 1x H3 Video VAE: direct chunked layout transpose [C, t, H, W] -> [t, H, W, C]
                out[b * T + t0:b * T + t1].copy_(chunk.permute(1, 2, 3, 0).to(dtype=torch.float32))
                del chunk
            else:
                try:
                    q = chunk.to(work_device, dtype=torch.float32)
                except Exception:
                    q = chunk.to(dtype=torch.float32)
                q = F.pixel_shuffle(q.permute(1, 0, 2, 3), upscale)   # t, oc, H*up, W*up
                out[b * T + t0:b * T + t1].copy_(q.permute(0, 2, 3, 1))
                del q, chunk
            t0 = t1
    return out


@torch.no_grad()
def _decode_e34_images(vae, z):
    """
    Unified Zero-Leak Tiled FP16 Decoder for BOTH:
    - MiniMax-H3-X2-Detail-v1 (packed_channels=12, upscale=2 -> [B*T, 2H, 2W, 3])
    - minimax_h3_video_vae_fp16 (packed_channels=3, upscale=1 -> [B*T, H, W, 3])

    Fixes the 32-bit cuDNN index overflow / corrupted output frames that happened
    when using the normal VAE on 0.6MP / 20s videos without tiling.
    """
    inner = vae.first_stage_model
    d = inner.decoder
    packed_channels = int(d.proj_out.weight.shape[0]) // (
        int(d.patch_size_t) * int(d.patch_size) * int(d.patch_size)
    )
    upscale = int(round(math.sqrt(max(1, packed_channels // 3))))
    if (packed_channels, upscale) not in ((12, 2), (3, 1)):
        _die(f"Expected packed12/PixelShuffle2 or packed3/standard1, got packed={packed_channels}, upscale={upscale}")

    saved = (
        inner.decoder.out_channels,
        inner.pixel_mean.clone() if torch.is_tensor(inner.pixel_mean) else inner.pixel_mean,
        inner.pixel_std.clone() if torch.is_tensor(inner.pixel_std) else inner.pixel_std,
        inner.tiling,
        inner.tile_size,
        inner.tile_overlap_min,
    )
    packed = None
    try:
        inner.decoder.out_channels = packed_channels
        inner.tiling = True
        inner.tile_size = 256
        inner.tile_overlap_min = 64

        with model_management.cuda_device_context(vae.device):
            mult = 2 if upscale == 2 else 1
            memory_used = vae.memory_used_decode(z.shape, vae.vae_dtype) * mult
            model_management.load_models_gpu(
                [vae.patcher],
                memory_required=memory_used,
                force_full_load=vae.disable_offload,
            )
            if packed_channels == 12:
                inner.pixel_mean = _expand_rgb_stat(saved[1], packed_channels)
                inner.pixel_std = _expand_rgb_stat(saved[2], packed_channels)
            else:
                # Guarantee 3-channel pixel_mean / pixel_std for normal VAE
                if torch.is_tensor(inner.pixel_mean) and inner.pixel_mean.shape[1] == 12:
                    inner.pixel_mean = inner.pixel_mean[:, ::4].contiguous()
                if torch.is_tensor(inner.pixel_std) and inner.pixel_std.shape[1] == 12:
                    inner.pixel_std = inner.pixel_std[:, ::4].contiguous()

            packed = torch.empty(
                inner.decode_output_shape(z.shape),
                device=model_management.intermediate_device(),
                dtype=torch.float16,
            )
            samples = z.to(device=vae.device, dtype=vae.vae_dtype)
            with torch.inference_mode():
                inner.decode(samples, output_buffer=packed)
            del samples
            images = _packed_to_images(packed, upscale, vae.device)
        return images
    finally:
        del packed
        (
            inner.decoder.out_channels,
            inner.pixel_mean,
            inner.pixel_std,
            inner.tiling,
            inner.tile_size,
            inner.tile_overlap_min,
        ) = saved
        model_management.soft_empty_cache()


@torch.no_grad()
def _decode_e34(vae, z):
    """X2 decode -> [B, 3, T, 2H, 2W] (used by the single-frame Detailed Upscale node)."""
    B = int(z.shape[0])
    images = _decode_e34_images(vae, z)
    return images.reshape(B, -1, images.shape[1], images.shape[2], images.shape[3]).permute(0, 4, 1, 2, 3).contiguous()


def _extract_video_latent(samples):
    z = samples["samples"] if isinstance(samples, dict) and "samples" in samples else samples
    if getattr(z, "is_nested", False) and hasattr(z, "unbind"):
        tensors = z.unbind()
        if tensors and len(tensors) > 0:
            z = tensors[0]
    elif hasattr(z, "unbind") and not isinstance(z, torch.Tensor):
        try:
            tensors = list(z.unbind())
            if tensors:
                z = tensors[0]
        except Exception:
            pass
    elif isinstance(z, (list, tuple)) and len(z) > 0:
        z = z[0]

    if hasattr(z, "ndim") and z.ndim == 4:
        z = z.unsqueeze(2)
    return z


_orig_comfy_vae_init = comfy.sd.VAE.__init__

@torch.no_grad()
def _decode_normal_h3_images(vae, video: torch.Tensor) -> torch.Tensor:
    """
    Exact native H3 tiled decode path (from ComfyUI-MiniMaxH3_LatentUpscaler) for
    minimax_h3_video_vae_fp16.safetensors. Uses H3's native 256px / 64px overlap
    spatial tiler and vae.process_output(pixel_samples), preventing ComfyUI's
    broken generic 3D OOM tiler fallback from corrupting frames.
    """
    inner = vae.first_stage_model
    out_dev = model_management.intermediate_device()
    saved = (inner.tiling, inner.tile_size, inner.tile_overlap_min)
    try:
        inner.tiling = True
        inner.tile_size = 256
        inner.tile_overlap_min = 64
        with model_management.cuda_device_context(vae.device):
            memory_used = vae.memory_used_decode(video.shape, vae.vae_dtype)
            model_management.load_models_gpu(
                [vae.patcher],
                memory_required=memory_used,
                force_full_load=vae.disable_offload,
            )
            pixel_samples = torch.empty(
                inner.decode_output_shape(video.shape),
                device=out_dev,
                dtype=vae.vae_output_dtype(),
            )
            samples = video.to(device=vae.device, dtype=vae.vae_dtype)
            with torch.inference_mode():
                inner.decode(samples, output_buffer=pixel_samples)
            del samples
            vae.process_output(pixel_samples)
        images = pixel_samples.to(out_dev).movedim(1, -1)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images
    finally:
        inner.tiling, inner.tile_size, inner.tile_overlap_min = saved
        model_management.soft_empty_cache()


def _apply_fast_decode_patch(cls):
    try:
        if not isinstance(cls, type):
            return
        if getattr(cls, "__module__", "").startswith("torch"):
            return
        if "decode" not in cls.__dict__ and not hasattr(cls, "decode"):
            return
        orig_fast_decode = getattr(cls, "decode", None)
        if orig_fast_decode is None or getattr(orig_fast_decode, "_is_x2_patched", False):
            return

        def patched_fast_decode(self, samples, vae, *args, **kwargs):
            if getattr(vae, "_is_x2_h3_vae", False):
                video = _extract_video_latent(samples)
                return (_decode_e34_images(vae, video),)
            if getattr(vae, "_is_h3_video_vae", False):
                video = _extract_video_latent(samples)
                return (_decode_normal_h3_images(vae, video),)
            return orig_fast_decode(self, samples, vae, *args, **kwargs)

        patched_fast_decode._is_x2_patched = True
        cls.decode = patched_fast_decode
        print("[ComfyUI-H3-X2-Detailed] [OK] Patched MiniMaxH3VAEDecodeFast for Both 2X Detail & Normal H3 Video VAE!", flush=True)
    except Exception as e:
        print(f"[ComfyUI-H3-X2-Detailed] Note: Fast decode patch skipped: {e}", flush=True)

def _patch_fast_decode_if_available():
    import sys
    try:
        import nodes
        if hasattr(nodes, "NODE_CLASS_MAPPINGS") and "MiniMaxH3VAEDecodeFast" in nodes.NODE_CLASS_MAPPINGS:
            _apply_fast_decode_patch(nodes.NODE_CLASS_MAPPINGS["MiniMaxH3VAEDecodeFast"])
            return
    except Exception:
        pass

    for mod_name, mod in list(sys.modules.items()):
        if not mod_name or "torch" in mod_name or mod_name.startswith("_"):
            continue
        try:
            if hasattr(mod, "__dict__") and "MiniMaxH3VAEDecodeFast" in mod.__dict__:
                cls = mod.__dict__["MiniMaxH3VAEDecodeFast"]
                _apply_fast_decode_patch(cls)
                return
        except Exception:
            continue


def _attach_h3_vae_hooks(vae_obj, is_x2: bool):
    vae_obj._is_x2_h3_vae = is_x2
    vae_obj._is_h3_video_vae = not is_x2

    def _h3_vae_decode(samples_in, *a, **kw):
        video = _extract_video_latent(samples_in)
        if is_x2:
            return _decode_e34_images(vae_obj, video)
        return _decode_normal_h3_images(vae_obj, video)

    vae_obj.decode = _h3_vae_decode
    vae_obj.decode_tiled = _h3_vae_decode


def _patched_comfy_vae_init(self, sd=None, *args, **kwargs):
    if sd is not None and "decoder.proj_out.weight" in sd:
        w = sd["decoder.proj_out.weight"]
        w_shape = tuple(w.shape)

        if w_shape == (12288, 2048):
            # 2X Detail H3 VAE checkpoint (MiniMax-H3-X2-Detail-v1)
            e34_sd, _, _ = _split_checkpoint(sd)
            b = e34_sd.get("decoder.proj_out.bias")
            bootstrap_sd = dict(e34_sd)
            bootstrap_sd["decoder.proj_out.weight"] = w[:3072].contiguous()
            if b is not None:
                bootstrap_sd["decoder.proj_out.bias"] = b[:3072].contiguous()

            _orig_comfy_vae_init(self, sd=bootstrap_sd, *args, **kwargs)

            inner = self.first_stage_model
            old_linear = inner.decoder.proj_out
            inner.decoder.proj_out = nn.Linear(
                2048, 12288, bias=True,
                device=old_linear.weight.device, dtype=old_linear.weight.dtype
            )
            inner.decoder.out_channels = 12
            inner.load_state_dict(e34_sd, strict=False)

            _attach_h3_vae_hooks(self, is_x2=True)
            _patch_fast_decode_if_available()

            print("[ComfyUI-H3-X2-Detailed] [OK] Loaded 2X Detail VAE (12-channel PixelShuffle2 + Ref2VA Encode Cache enabled)!", flush=True)
            return

        elif w_shape == (3072, 2048):
            # Standard 1x H3 Video VAE checkpoint (minimax_h3_video_vae_fp16.safetensors)
            _orig_comfy_vae_init(self, sd=sd, *args, **kwargs)
            inner = getattr(self, "first_stage_model", None)
            if inner is not None and hasattr(inner, "decoder") and hasattr(inner.decoder, "patch_size_t"):
                _attach_h3_vae_hooks(self, is_x2=False)
                _patch_fast_decode_if_available()
                print("[ComfyUI-H3-X2-Detailed] [OK] Loaded Standard H3 Video VAE (Tiled FP16 Decode + NestedTensor safety + Ref2VA Encode Cache enabled)!", flush=True)
            return

    return _orig_comfy_vae_init(self, sd=sd, *args, **kwargs)

comfy.sd.VAE.__init__ = _patched_comfy_vae_init
_patch_fast_decode_if_available()

# Hook ComfyUI PromptExecutor to re-verify fast decode patch before every graph execution
try:
    import execution
    if hasattr(execution, "PromptExecutor") and hasattr(execution.PromptExecutor, "execute"):
        orig_pe_execute = execution.PromptExecutor.execute
        if not getattr(orig_pe_execute, "_has_x2_hook", False):
            def patched_pe_execute(self, *a, **kw):
                _patch_fast_decode_if_available()
                return orig_pe_execute(self, *a, **kw)
            patched_pe_execute._has_x2_hook = True
            execution.PromptExecutor.execute = patched_pe_execute
except Exception:
    pass


def _center_crop_multiple32(x):
    _, _, H, W = x.shape
    nh = (H // 32) * 32
    nw = (W // 32) * 32
    if nh < 32 or nw < 32:
        _die(f"Input is too small: {H}x{W}")
    if nh == H and nw == W:
        return x
    y = (H - nh) // 2
    z = (W - nw) // 2
    print(f"[H3 X2 Detail] center crop {H}x{W} -> {nh}x{nw} for H3 alignment")
    return x[:, :, y:y + nh, z:z + nw].contiguous()


class MiniMaxH3X2DetailedUpscale:
    _cache_path = None
    _cache_vae = None
    _cache_detail = None

    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("vae")
        preferred = "MiniMax-H3-X2-Detail-v1.safetensors"
        if preferred in names:
            names = [preferred] + [x for x in names if x != preferred]
        return {
            "required": {
                "images": ("IMAGE",),
                "detail_vae": (names,),
                "detail_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("x2_detail", "x2_e34")
    FUNCTION = "upscale"
    CATEGORY = "MiniMax H3/X2 Detail"
    DESCRIPTION = (
        "Validated E34 + B32 Teacher path. Input is an IMAGE/video-frame batch. "
        "B32 is extracted from encoder down.1.block.1, so this is not latent-only T2V decode."
    )

    def _runtime(self, name):
        path = folder_paths.get_full_path("vae", name)
        if path is None:
            _die(f"VAE checkpoint not found: {name}")
        path = str(Path(path))
        if self.__class__._cache_path != path:
            print(f"[H3 X2 Detail v1.2] loading {path}", flush=True)
            self.__class__._cache_vae, self.__class__._cache_detail = _load_runtime(Path(path))
            self.__class__._cache_path = path
        return self.__class__._cache_vae, self.__class__._cache_detail

    def upscale(self, images, detail_vae, detail_strength=1.0):
        if images.ndim != 4 or images.shape[-1] != 3:
            _die(f"Expected ComfyUI IMAGE [N,H,W,3], got {tuple(images.shape)}")

        vae, detail = self._runtime(detail_vae)
        mod = _resolve_down1(vae)
        device = vae.device
        dtype = vae.vae_dtype

        out_detail = []
        out_e34 = []
        n = int(images.shape[0])

        for i in range(n):
            print(f"[H3 X2 Detail v1.2] frame {i+1}/{n}", flush=True)
            img = images[i:i+1].detach().float().cpu().permute(0, 3, 1, 2).contiguous()
            img = _center_crop_multiple32(img)

            with model_management.cuda_device_context(vae.device):
                _prepare_encoder_once(vae)
                feat = _stitched_down1(vae, img, mod, DEFAULT_TILE, DEFAULT_OVERLAP)

            z = vae.encode(img.permute(0, 2, 3, 1).contiguous()).detach().float().cpu()
            if z.ndim == 4:
                z = z.unsqueeze(2)
            if z.ndim != 5:
                _die(f"Unexpected H3 latent shape {tuple(z.shape)}")

            video = _decode_e34(vae, z)
            if video.ndim != 5 or video.shape[1] != 3:
                _die(f"Unexpected E34 decode shape {tuple(video.shape)}")
            base = video[:, :, -1].contiguous()

            detail = detail.to(device=device, dtype=dtype)
            with torch.inference_mode():
                delta = detail(feat.to(device=device, dtype=dtype)).float().cpu()
            final = (base + float(detail_strength) * delta).clamp(0, 1)

            if final.shape[-2:] != (img.shape[-2] * 2, img.shape[-1] * 2):
                _die(
                    f"Output shape mismatch: input={tuple(img.shape[-2:])}, "
                    f"E34={tuple(base.shape[-2:])}, detail={tuple(final.shape[-2:])}"
                )

            out_detail.append(final[0].permute(1, 2, 0).contiguous())
            out_e34.append(base[0].permute(1, 2, 0).contiguous())

            del img, feat, z, video, base, delta, final
            model_management.soft_empty_cache()

        detail_batch = torch.stack(out_detail, dim=0)
        e34_batch = torch.stack(out_e34, dim=0)
        print(f"[H3 X2 Detail v1.2] DONE {tuple(detail_batch.shape)}", flush=True)
        return (detail_batch, e34_batch)


class H3SaveFrozenLatentV4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename": ("STRING", {"default": "H3 VAE detailed.pt"}),
            }
        }
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "saved_path")
    FUNCTION = "save_latent"
    CATEGORY = "MiniMaxH3/Latent"

    def save_latent(self, latent, filename="H3 VAE detailed.pt"):
        save_path = ""
        try:
            output_dir = folder_paths.get_output_directory()
            save_path = str(Path(output_dir) / filename)
            torch.save(latent, save_path)
            print(f"[H3SaveFrozenLatentV4] Successfully saved frozen latent to {save_path}", flush=True)
        except Exception as e:
            print(f"[H3SaveFrozenLatentV4] Note: latent save skipped ({e}), passing latent through.", flush=True)
        return (latent, save_path)


class GetImageSizePlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_target_resolution": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (["original", "custom", "1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "9:16"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "multiple_of": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
            }
        }
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "original_width", "original_height", "actual_megapixels")
    FUNCTION = "get_size"
    CATEGORY = "image"

    def get_size(self, image, use_target_resolution=False, aspect_ratio="9:16", megapixels=1.0, target_width=1024, target_height=1024, multiple_of=32):
        h = int(image.shape[1])
        w = int(image.shape[2])
        orig_w, orig_h = w, h
        if not use_target_resolution:
            final_w = int(round(w / multiple_of) * multiple_of)
            final_h = int(round(h / multiple_of) * multiple_of)
        else:
            final_w = int(round(target_width / multiple_of) * multiple_of)
            final_h = int(round(target_height / multiple_of) * multiple_of)
        actual_mp = float(final_w * final_h) / 1000000.0
        return (final_w, final_h, orig_w, orig_h, actual_mp)


# ==============================================================================
# Interactive Stage 1 -> Stage 2 Approval Gate (Zero Timer) & 2x/4x Scale Option
# ==============================================================================
import time
import json

_APPROVAL_STATE = {}  # node_id -> {"status": "waiting" | "approve" | "approve_2x" | "approve_4x" | "stop"}

def _register_approval_routes():
    try:
        from server import PromptServer
        from aiohttp import web

        if PromptServer.instance is None:
            return
        if getattr(PromptServer.instance, "_h3_approval_routes_registered", False):
            return

        @PromptServer.instance.routes.post("/h3_approval/decision")
        async def h3_approval_decision(request):
            try:
                data = await request.json()
            except Exception:
                data = {}
            node_id = str(data.get("node_id", "latest"))
            decision = str(data.get("decision", "approve"))
            _APPROVAL_STATE[node_id] = {"status": decision}
            _APPROVAL_STATE["latest"] = {"status": decision}
            for k in list(_APPROVAL_STATE.keys()):
                if _APPROVAL_STATE[k].get("status") == "waiting":
                    _APPROVAL_STATE[k]["status"] = decision
            print(f"[H3UpscaleApprovalGate] User decision received for node {node_id}: {decision}", flush=True)
            return web.json_response({"ok": True, "node_id": node_id, "decision": decision})

        @PromptServer.instance.routes.get("/h3_approval/status")
        async def h3_approval_status(request):
            return web.json_response({"states": _APPROVAL_STATE})

        PromptServer.instance._h3_approval_routes_registered = True
        print("[ComfyUI-H3-X2-Detailed] [OK] Registered /h3_approval/decision interactive gate endpoint (No Timer).", flush=True)
    except Exception as e:
        print(f"[ComfyUI-H3-X2-Detailed] Note: Could not register PromptServer routes at import time ({e}).", flush=True)

_register_approval_routes()


class H3UpscaleApprovalGate:
    """
    Interactive Stage 1 -> Stage 2 Approval Gate & 2x/4x Upscale Option (Zero Timer).
    Receives `video` directly from Stage 1's `SaveVideo` output slot 0 so that Stage 1's
    768p draft video is 100% saved and displayed in the ComfyUI player BEFORE this node
    pauses and waits for the user's decision.
    """

    APPROVAL_MODES = [
        "Ask Approval in UI (Wait Indefinitely - No Timer)",
        "Auto-Upscale Immediately (No Pause)",
        "Stop After 768p Draft Only (Do Not Upscale)",
    ]
    UPSCALE_FACTORS = [
        "2x Upscale (1440p / 2K)",
        "4x Upscale (2880p / 4K)",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "approval_mode": (cls.APPROVAL_MODES, {"default": "Ask Approval in UI (Wait Indefinitely - No Timer)"}),
                "upscale_factor": (cls.UPSCALE_FACTORS, {"default": "2x Upscale (1440p / 2K)"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "latent": ("LATENT",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "LATENT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("video", "images", "audio", "latent", "scale_float", "target_width", "target_height")
    FUNCTION = "gate"
    CATEGORY = "MiniMax H3/Upscale Gate"

    @classmethod
    def IS_CHANGED(cls, video, approval_mode, upscale_factor, images=None, audio=None, latent=None, unique_id=None):
        # Always re-evaluate when interactive approval or stop mode is selected
        return float("nan")

    def _stop_after_draft(self, node_id: str, reason: str):
        print(f"[H3UpscaleApprovalGate] {reason} — Stopping cleanly after Stage 1 768p draft.", flush=True)
        try:
            from server import PromptServer
            if PromptServer.instance is not None:
                PromptServer.instance.send_sync("h3_approval_resolved", {"node_id": node_id, "decision": "stop"})
        except Exception:
            pass
        if hasattr(model_management, "InterruptProcessingException"):
            raise model_management.InterruptProcessingException()
        model_management.interrupt_current_processing(True)
        raise RuntimeError("Stopped after Stage 1 768p draft by user choice.")

    def gate(self, video, approval_mode="Ask Approval in UI (Wait Indefinitely - No Timer)",
             upscale_factor="2x Upscale (1440p / 2K)", images=None, audio=None, latent=None, unique_id=None):
        _register_approval_routes()
        node_id = str(unique_id) if unique_id is not None else "latest"

        # Default scale factor from dropdown widget
        scale_float = 4.0 if str(upscale_factor).startswith("4x") else 2.0

        mode_str = str(approval_mode)
        if mode_str.startswith("Stop After"):
            self._stop_after_draft(node_id, "Mode is set to 'Stop After 768p Draft Only'")

        elif mode_str.startswith("Ask Approval"):
            _APPROVAL_STATE[node_id] = {"status": "waiting"}
            _APPROVAL_STATE["latest"] = {"status": "waiting"}
            print(
                f"[H3UpscaleApprovalGate] Stage 1 (768p Draft) is saved and visible in UI! "
                f"Waiting indefinitely (NO TIMER) for user approval on node #{node_id}...",
                flush=True,
            )
            try:
                from server import PromptServer
                if PromptServer.instance is not None:
                    PromptServer.instance.send_sync(
                        "h3_approval_waiting",
                        {
                            "node_id": node_id,
                            "upscale_factor": upscale_factor,
                            "scale_float": scale_float,
                        },
                    )
            except Exception as e:
                print(f"[H3UpscaleApprovalGate] Warning sending WS event: {e}", flush=True)

            # Infinite wait loop — NO TIMER! Waits until user clicks Approve or Stop in UI
            while True:
                if model_management.processing_interrupted():
                    _APPROVAL_STATE.pop(node_id, None)
                    if hasattr(model_management, "InterruptProcessingException"):
                        raise model_management.InterruptProcessingException()
                    break

                st = _APPROVAL_STATE.get(node_id, {}).get("status", "waiting")
                if st == "waiting":
                    st = _APPROVAL_STATE.get("latest", {}).get("status", "waiting")

                if st in ("approve", "approve_2x", "approve_4x"):
                    if st == "approve_2x":
                        scale_float = 2.0
                    elif st == "approve_4x":
                        scale_float = 4.0
                    _APPROVAL_STATE.pop(node_id, None)
                    _APPROVAL_STATE.pop("latest", None)
                    try:
                        from server import PromptServer
                        if PromptServer.instance is not None:
                            PromptServer.instance.send_sync(
                                "h3_approval_resolved",
                                {"node_id": node_id, "decision": st, "scale_float": scale_float},
                            )
                    except Exception:
                        pass
                    print(f"[H3UpscaleApprovalGate] Approved! Proceeding to Stage 2 ({scale_float}x upscale).", flush=True)
                    break

                elif st == "stop":
                    _APPROVAL_STATE.pop(node_id, None)
                    _APPROVAL_STATE.pop("latest", None)
                    self._stop_after_draft(node_id, "User clicked 'Stop Here (Keep 768p Draft)'")

                time.sleep(0.2)

        # Compute target width & height for 2x or 4x
        base_w, base_h = 1280, 720
        if images is not None and hasattr(images, "shape") and images.ndim == 4:
            base_h = int(images.shape[1])
            base_w = int(images.shape[2])
        else:
            try:
                dims = video.get_dimensions()
                if dims and len(dims) == 2:
                    base_w, base_h = int(dims[0]), int(dims[1])
            except Exception:
                pass

        if scale_float == 2.0 and base_w in (1280, 1344) and base_h in (720, 768):
            target_width, target_height = 2560, 1440
        else:
            target_width = max(32, int(round((base_w * scale_float) / 32.0) * 32))
            target_height = max(32, int(round((base_h * scale_float) / 32.0) * 32))

        return (video, images, audio, latent, float(scale_float), int(target_width), int(target_height))


# ==============================================================================
# Open-H3-Regenerate-2K-IC Companion Nodes (Enhanced for Any Valid H3 Frame Count)
# ==============================================================================
_IC_TASK = (
    "Enhance <Video 1> into a high-quality version of the same video. "
    "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
    "Restore fine details and remove degradation. Do not continue or change the scene."
)

_IC_SIGMAS = {
    "up3": [0.834942, 0.696852, 0.468753, 0.0],
    "up4": [0.923077, 0.834942, 0.696852, 0.468753, 0.0],
    "hf8": [1.0, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0.0],
}

_IC_SILENCE_ASSET = Path(__file__).resolve().parent / "assets" / "silence_audio_124f.safetensors"


def _ic_cover_resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("OpenH3ICDraft expects IMAGE [T,H,W,C]")
    x = frames[..., :3].movedim(-1, 1).float()
    h, w = x.shape[-2:]
    scale = max(height / h, width / w)
    sh, sw = max(height, round(h * scale)), max(width, round(w * scale))
    if (sh, sw) != (h, w):
        x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False, antialias=scale < 1.0)
    top, left = (sh - height) // 2, (sw - width) // 2
    x = x[..., top:top + height, left:left + width]
    return x.movedim(1, -1).clamp(0.0, 1.0).contiguous()


def _ic_fit_audio(audio, sample_count: int):
    if audio is None:
        return {"waveform": torch.zeros((1, 2, sample_count), dtype=torch.float32), "sample_rate": 32000}
    waveform = audio["waveform"].float().clone()
    if waveform.ndim != 3:
        raise ValueError("AUDIO waveform must be [B,C,L]")
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    if waveform.shape[-1] < sample_count:
        pad = waveform.new_zeros((waveform.shape[0], waveform.shape[1], sample_count - waveform.shape[-1]))
        waveform = torch.cat((waveform, pad), dim=-1)
    elif waveform.shape[-1] > sample_count:
        waveform = waveform[..., :sample_count]
    return {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}


class OpenH3ICDraft:
    """Prepare a Stage 1 draft for Open-H3-Regenerate-2K-IC (<Video 1>). Supports any valid H3 length (5 + 17*k)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "target_width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "target_height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "frame_count": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "output_fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "fit": (["cover", "stretch"], {"default": "cover"}),
                "audio_mode": (["trim_pad", "passthrough", "silent"], {"default": "trim_pad"}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("draft_frames", "draft_audio", "report")
    FUNCTION = "prepare"
    CATEGORY = "Open-H3/Regenerate 2K IC"

    def prepare(self, images, source_fps, target_width, target_height, frame_count, output_fps,
                fit="cover", audio_mode="trim_pad", audio=None):
        if images.shape[0] < 1:
            raise ValueError("The draft video contains no frames")
        target_width = max(32, int(round(target_width / 32.0) * 32))
        target_height = max(32, int(round(target_height / 32.0) * 32))
        source_fps = float(source_fps)
        source_count = int(images.shape[0])
        frame_count = int(frame_count)
        if frame_count <= 0:
            frame_count = source_count
        # Snap frame_count to nearest valid 5 + 17*k if needed
        if (frame_count - 5) % 17 != 0:
            frame_count = max(5, 5 + round((frame_count - 5) / 17.0) * 17)

        wanted = [int(i * source_fps / output_fps) for i in range(frame_count)]
        valid = sum(1 for idx in wanted if idx < source_count)
        indices = torch.tensor([min(idx, source_count - 1) for idx in wanted], device=images.device)
        picked = images.index_select(0, indices)
        if fit == "cover":
            frames = _ic_cover_resize(picked, int(target_height), int(target_width))
        else:
            x = picked[..., :3].movedim(-1, 1).float()
            x = F.interpolate(x, size=(int(target_height), int(target_width)), mode="bilinear",
                              align_corners=False, antialias=True)
            frames = x.movedim(1, -1).clamp(0.0, 1.0).contiguous()
        if audio_mode == "silent" or audio is None:
            out_audio = _ic_fit_audio(None, int(round(frames.shape[0] / float(output_fps) * 32000)))
        elif audio_mode == "passthrough":
            waveform = audio["waveform"].float().clone()
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)
            out_audio = {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}
        else:
            sample_count = int(round(frames.shape[0] / float(output_fps) * int(audio["sample_rate"])))
            out_audio = _ic_fit_audio(audio, sample_count)
        report = {
            "source_frames": source_count,
            "source_fps": source_fps,
            "real_frames_at_24fps": valid,
            "output_frames": int(frames.shape[0]),
            "output_fps": int(output_fps),
            "canvas": [int(target_width), int(target_height)],
            "audio_samples": int(out_audio["waveform"].shape[-1]),
            "audio_rate": int(out_audio["sample_rate"]),
        }
        return frames, out_audio, json.dumps(report, ensure_ascii=False)


class OpenH3ICPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"caption": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "Open-H3/Regenerate 2K IC"

    def build(self, caption):
        return (_IC_TASK + "\n\nScene description: " + (caption or "").strip(),)


class OpenH3ICSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (["up3", "up4", "hf8"], {"default": "up3"}),
                "custom_sigmas": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "report")
    FUNCTION = "build"
    CATEGORY = "Open-H3/Regenerate 2K IC"

    def build(self, sampler, custom_sigmas=""):
        if custom_sigmas.strip():
            values = [float(part.strip()) for part in custom_sigmas.split(",") if part.strip()]
            if not values or abs(values[-1]) > 1e-12:
                raise ValueError("custom_sigmas must end in 0.0")
        else:
            if sampler not in _IC_SIGMAS:
                raise ValueError("Unknown IC sampler: %s" % sampler)
            values = list(_IC_SIGMAS[sampler])
        report = {"sampler": sampler if not custom_sigmas.strip() else "custom", "sigmas": values,
                  "steps": max(0, len(values) - 1)}
        return torch.tensor(values, dtype=torch.float32), json.dumps(report)


class OpenH3ICComposeLatent:
    """Build the clean target AV latent and pin its audio rows to official silence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "reference_latent": ("LATENT",),
                "pin_audio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "compose"
    CATEGORY = "Open-H3/Regenerate 2K IC"

    @staticmethod
    def _fit_silence(audio: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if audio.shape[:3] != target.shape[:3]:
            return torch.zeros_like(target)
        if audio.shape[-1] > target.shape[-1]:
            audio = audio[..., :target.shape[-1]]
        elif audio.shape[-1] < target.shape[-1]:
            pad = audio.new_zeros(audio.shape[:-1] + (target.shape[-1] - audio.shape[-1],))
            audio = torch.cat((audio, pad), dim=-1)
        return audio

    def compose(self, video_latent, reference_latent, pin_audio=True):
        import comfy.nested_tensor as nested

        video = video_latent["samples"]
        if video.is_nested:
            video = video.tensors[0]
        ref = reference_latent["samples"]
        if not ref.is_nested or len(ref.tensors) != 2:
            raise ValueError("reference_latent must be the joint AV latent from MiniMax H3 Reference to Video")
        video = video.to(dtype=ref.tensors[0].dtype, device=ref.tensors[0].device)
        if tuple(video.shape) != tuple(ref.tensors[0].shape):
            video = F.interpolate(
                video, size=tuple(ref.tensors[0].shape[2:]), mode="trilinear", align_corners=False
            )
        if _IC_SILENCE_ASSET.exists():
            raw = load_file(str(_IC_SILENCE_ASSET), device="cpu")["audio_latents"]
            silence = raw.permute(1, 0, 2).unsqueeze(0).to(device=ref.tensors[1].device, dtype=ref.tensors[1].dtype)
            silence = self._fit_silence(silence, ref.tensors[1])
        else:
            silence = torch.zeros_like(ref.tensors[1])

        out = reference_latent.copy()
        out.update({k: v for k, v in video_latent.items() if k != "samples"})
        out["samples"] = nested.NestedTensor((video, silence))
        if pin_audio:
            out["noise_mask"] = nested.NestedTensor((torch.ones_like(video), torch.zeros_like(silence)))
        else:
            out.pop("noise_mask", None)
        out["open_h3_ic"] = {"audio": "official_silence", "pinned": bool(pin_audio)}
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3X2DetailedUpscale": MiniMaxH3X2DetailedUpscale,
    "H3SaveFrozenLatentV4": H3SaveFrozenLatentV4,
    "GetImageSizePlus": GetImageSizePlus,
    "H3UpscaleApprovalGate": H3UpscaleApprovalGate,
    "OpenH3ICDraft": OpenH3ICDraft,
    "OpenH3ICPrompt": OpenH3ICPrompt,
    "OpenH3ICSigmas": OpenH3ICSigmas,
    "OpenH3ICComposeLatent": OpenH3ICComposeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3X2DetailedUpscale": "MiniMax H3 VAE 2X (Detailed Upscale)",
    "H3SaveFrozenLatentV4": "MiniMax H3 Save Frozen Latent V4",
    "GetImageSizePlus": "Get Image Size Plus",
    "H3UpscaleApprovalGate": "⚡ H3 Stage 1 -> Stage 2 Approval Gate & 2x/4x Option (No Timer)",
    "OpenH3ICDraft": "Open-H3 IC · Prepare Draft (124f / 24fps)",
    "OpenH3ICPrompt": "Open-H3 IC · Official Prompt",
    "OpenH3ICSigmas": "Open-H3 IC · Sigma Grid",
    "OpenH3ICComposeLatent": "Open-H3 IC · Compose Target Latent",
}
