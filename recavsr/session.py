"""Driving one video through the streaming model: block schedule, LQ preparation, the per-video state of the
transformer and of the FlashDecoder, and the colour corrections (StableSR's AdaIN and wavelet matching)."""
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

from .cache import RollingRoPE
from .dit import BODY_FRAMES, BODY_LATENTS, Conditioning, LQProjState, PREFIX_FRAMES, PREFIX_LATENTS

STRIDE = 32  # pixels per token: the VAE's 16 and the 2x2 patch
WINDOW = (22, 40)  # self-attention window in tokens (height, width), the release's default


def scaled_size(height, width, scale):
    """Each dimension scaled and rounded half up, as the reference does."""
    return math.floor(height * scale + 0.5), math.floor(width * scale + 0.5)


def padded_size(height, width):
    return height + (-height) % STRIDE, width + (-width) % STRIDE


def block_schedule(total):
    """(first frame, valid frames, block length): 21 frames, then 8 at a time; the last block is padded."""
    start = 0
    while start < total:
        block = PREFIX_FRAMES if start == 0 else BODY_FRAMES
        valid = min(block, total - start)
        yield start, valid, block
        start += valid


def upscale_frames(frames, size):
    """uint8 [T, h, w, 3] frames bilinear to (height, width) as [T, 3, H, W] float32 in [-1, 1]. OpenCV's resize is
    the reference's; torch's bilinear is the same filter up to 8-bit rounding."""
    if cv2 is None:
        x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False).round_().div_(127.5).sub_(1.0)
    enlarged = np.stack([cv2.resize(f, (size[1], size[0]), interpolation=cv2.INTER_LINEAR) for f in frames])
    return torch.from_numpy(enlarged).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def prepare_block(frames, start, valid, block, size, device):
    """The LQ input of one block, [1, 3, block, H, W] at the padded output size on the device."""
    x = upscale_frames(frames[start:start + valid], size).to(device).permute(1, 0, 2, 3).unsqueeze(0)
    if valid < block:
        x = torch.cat((x, x[:, :, -1:].expand(-1, -1, block - valid, -1, -1)), 2)
    ph, pw = (-size[0]) % STRIDE, (-size[1]) % STRIDE
    if pw:
        x = F.pad(x, (0, pw, 0, 0, 0, 0), mode="reflect" if pw < size[1] else "replicate")
    if ph:
        x = F.pad(x, (0, 0, 0, ph, 0, 0), mode="reflect" if ph < size[0] else "replicate")
    return x


class VideoStream:
    """The transformer's state for one video: conditioning, rolling RoPE, routed KV history, LQ convolution
    history, the recycled latents and the noise generator. One per video and geometry."""

    def __init__(self, model, prompt, height, width, seed, device, kernels="exact", kv_storage=None, window=WINDOW):
        self.model, self.height, self.width, self.window, self.kernels = model, height, width, window, kernels
        self.device, self.dtype = device, model.dtype
        self.conditioning = Conditioning(model, prompt.to(device=device, dtype=model.dtype), kernels)
        self.rope = RollingRoPE(model.dim // model.heads, height // STRIDE, width // STRIDE, device)
        self.caches = model.new_caches(kv_storage)
        self.lq_state = LQProjState()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.offset = 0
        self.previous = None

    def step(self, lq, noise=None):
        """Consume one LQ block [1, 3, frames, H, W] and return its latents [1, C, T, H/16, W/16] float32."""
        frames = PREFIX_LATENTS if self.offset == 0 else BODY_LATENTS
        if noise is None:
            shape = (1, self.model.in_channels, frames, self.height // 16, self.width // 16)
            noise = torch.randn(shape, generator=self.generator, device=self.device, dtype=torch.float32)
        recycle = torch.zeros_like(noise, dtype=self.dtype) if self.previous is None else self.previous
        latent = self.model(noise, lq.to(self.dtype), recycle, self.offset, self.conditioning, self.rope,
                            self.caches, self.lq_state, self.window, self.kernels)
        self.previous = latent[:, :, -BODY_LATENTS:].to(self.dtype).clone()
        self.offset += frames
        return latent


class DecoderStream:
    """The FlashDecoder's state for one video; decodes a block of latents frame by frame."""

    def __init__(self, decoder, latent_format, device, dtype, kernels="exact"):
        self.decoder, self.dtype, self.kernels = decoder, dtype, kernels
        self.state = decoder.new_state()
        self.mean = latent_format.latents_mean.to(device=device, dtype=dtype)
        self.std = latent_format.latents_std.to(device=device, dtype=dtype)

    def step(self, latents, lq):
        """Yield [1, 3, 1 or 4, H, W] RGB in [-1, 1] per latent frame, given the block's LQ frames."""
        offset = 0
        for i in range(latents.shape[2]):
            take = 1 if self.state.frame_index == 0 else 4
            raw = latents[:, :, i].to(self.dtype) * self.std[:, :, 0] + self.mean[:, :, 0]
            yield self.decoder(raw, self.state, lq[:, :, offset:offset + take], self.dtype, self.kernels).clamp(-1, 1)
            offset += take


def adain_color_fix(decoded, reference):
    """Match each frame's per-channel mean and variance to the LQ frame; [T, 3, H, W] in [-1, 1]."""
    content, source = (decoded.float() + 1) * 0.5, (reference.float() + 1) * 0.5
    correction = int(content.shape[-2] * content.shape[-1] > 1)
    content_var, content_mean = torch.var_mean(content, dim=(-2, -1), correction=correction, keepdim=True)
    source_var, source_mean = torch.var_mean(source, dim=(-2, -1), correction=correction, keepdim=True)
    gain = ((source_var + 1e-5) / (content_var + 1e-5)).sqrt()
    return ((content - content_mean) * gain + source_mean) * 2 - 1


def wavelet_color_fix(decoded, frames):
    """Keep the output's high frequencies over the bicubic-upscaled LQ frames' low frequencies; returns [0, 1]."""
    content = decoded.float().add(1).mul(0.5).clamp_(0, 1)
    reference = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).to(device=decoded.device, dtype=torch.float32).div_(255.0)
    reference = F.interpolate(reference, size=content.shape[-2:], mode="bicubic", align_corners=False).clamp_(0, 1)
    taps = content.new_tensor([1.0, 2.0, 1.0])
    kernel = (torch.outer(taps, taps) / 16).expand(3, 1, 3, 3).contiguous()
    high = torch.zeros_like(content)
    for radius in (1, 2, 4, 8, 16):
        padding = (radius,) * 4
        blurred = F.conv2d(F.pad(content, padding, mode="replicate"), kernel, dilation=radius, groups=3)
        high.add_(content - blurred)
        content = blurred
        reference = F.conv2d(F.pad(reference, padding, mode="replicate"), kernel, dilation=radius, groups=3)
    return (high + reference).clamp_(0, 1)
