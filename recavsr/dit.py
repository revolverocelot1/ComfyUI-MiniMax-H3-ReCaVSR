"""ReCaVSR's one-step streaming transformer (Wan2.2 TI2V-5B with the LQ and recycled-latent adapters).

Module and parameter names are those of the released transformer.safetensors (a merged diffusers
WanTransformer3DModel plus lq_proj and sr_proj), so the file loads without a key map. The forward is one
streaming step of the reference runtime: a fixed timestep and prompt for the whole video, a routed KV history
per block, the LQ frames projected by two causal convolutions, and the previous block's latents fed back
through sr_proj. Norms and residual adds run in float32 like the reference; everything else in the model dtype.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

from .attention import full_attention, rotate_pairs, windowed_attention
from .cache import KVCache

TIMESTEP = 1000.0
PATCH = 2  # tokens are 2x2 latent pixels: one per 32x32 RGB pixels
PREFIX_LATENTS, BODY_LATENTS = 6, 2  # latent frames per block; the first block is longer
PREFIX_FRAMES, BODY_FRAMES = 21, 8  # the RGB frames those blocks cover (1 + 4 * (T - 1))

# model_config.json of the release
CONFIG = dict(in_channels=48, dim=3072, ffn_dim=14336, heads=24, layers=30, freq_dim=256, text_dim=4096, eps=1e-6,
              lq_hidden=(2048, 3072))
ROUTER = dict(initial=5, stride=6, anchors=2, actions=[
    "none", "none", "recent_1", "none", "recent_2", "none", "recent_2", "recent_4", "none", "recent_2_anchor",
    "recent_4", "recent_4_anchor", "recent_2", "recent_4_anchor", "anchor", "recent_4_anchor", "recent_2",
    "recent_4_anchor", "recent_2_anchor", "recent_4", "recent_2", "none", "recent_2_anchor", "recent_2", "none",
    "recent_2", "none", "recent_1", "none", "none"])


def timestep_embedding(t, dim):
    """diffusers Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0): [cos, sin]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    angles = t[:, None].float() * freqs[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads, eps, operation_settings):
        super().__init__()
        ops, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.to_q = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.to_k = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.to_v = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.to_out = nn.ModuleList([ops.Linear(dim, dim, device=device, dtype=dtype)])
        self.norm_q = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.norm_k = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)


class GELUProj(nn.Module):
    def __init__(self, dim, inner, operation_settings):
        super().__init__()
        self.proj = operation_settings["operations"].Linear(dim, inner, device=operation_settings["device"], dtype=operation_settings["dtype"])


class FeedForward(nn.Module):
    def __init__(self, dim, inner, operation_settings):
        super().__init__()
        self.net = nn.ModuleList([GELUProj(dim, inner, operation_settings), nn.Identity(),
                                  operation_settings["operations"].Linear(inner, dim, device=operation_settings["device"], dtype=operation_settings["dtype"])])

    def forward(self, x):
        return comfy.ops.linear_input_act(self.net[2], self.net[0].proj(x), "gelu_tanh")


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads, eps, operation_settings):
        super().__init__()
        ops, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.norm1 = ops.LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(dim, heads, eps, operation_settings)
        self.attn2 = Attention(dim, heads, eps, operation_settings)
        self.norm2 = ops.LayerNorm(dim, eps, elementwise_affine=True, device=device, dtype=dtype)
        self.ffn = FeedForward(dim, ffn_dim, operation_settings)
        self.norm3 = ops.LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, dim, device=device, dtype=dtype))

    def forward(self, x, cond, rope, history, positions, window, kernels):
        """One block over the current planes; returns the hidden states and the spatially rotated K and V.

        exact: the reference's op sequence (float32 norms, modulation and residual adds, each rounded separately).
        fast: comfy-kitchen's fused adaln and rope kernels, the gated residual adds in the output projections'
        epilogues, and a bigger attention tile; bf16 modulation, each value rounded once."""
        shift, scale, gate, cshift, cscale, cgate = cond.modulation
        heads = self.attn1.heads
        span = rope.height * rope.width
        attention = windowed_attention if rope.height > window[0] or rope.width > window[1] else full_attention
        if kernels == "fast":
            norm = comfy.quant_ops.ck.adaln(x, scale, shift, self.norm1.eps)
        else:
            norm = (self.norm1(x.float()) * (1 + scale) + shift).to(x.dtype)
        q = self.attn1.norm_q(self.attn1.to_q(norm)).unflatten(2, (heads, -1))
        k = self.attn1.norm_k(self.attn1.to_k(norm)).unflatten(2, (heads, -1))
        v = self.attn1.to_v(norm).unflatten(2, (heads, -1))
        if kernels == "fast":
            rotation = rope.spatial_rotation(q.shape[1] // span)
            q = comfy.quant_ops.ck.apply_rope1(q, rotation)
            k = comfy.quant_ops.ck.apply_rope1(k, rotation)
        else:
            q = rotate_pairs(q.unflatten(1, (-1, span)), rope.spatial_cos, rope.spatial_sin).flatten(1, 2)
            k = rotate_pairs(k.unflatten(1, (-1, span)), rope.spatial_cos, rope.spatial_sin).flatten(1, 2)
        out = attention(q, k, v, *history, positions, rope.temporal_cos, rope.temporal_sin, rope.height, rope.width, *window, kernels).flatten(2)
        if kernels == "fast":
            x = comfy.ops.linear_input_act(self.attn1.to_out[0], out, None, residual=x, residual_scale=gate)
            norm = comfy.quant_ops.ck.adaln(x, *cond.norm2, self.norm2.eps)
        else:
            x = (x.float() + self.attn1.to_out[0](out) * gate).to(x.dtype)
            norm = self.norm2(x.float()).to(x.dtype)
        cq = AttentionTensorContainer(self.attn2.norm_q(self.attn2.to_q(norm)))
        out = optimized_attention(cq, AttentionTensorContainer(cond.cross_k), AttentionTensorContainer(cond.cross_v), heads,
                                  preferred_attention=self.attn2.comfy_attention)
        if kernels == "fast":
            x = comfy.ops.linear_input_act(self.attn2.to_out[0], out, None, residual=x, residual_scale=cond.ones)
            norm = comfy.quant_ops.ck.adaln(x, cscale, cshift, self.norm3.eps)
            x = comfy.ops.linear_input_act(self.ffn.net[2], self.ffn.net[0].proj(norm), "gelu_tanh", residual=x, residual_scale=cgate)
        else:
            x = x + self.attn2.to_out[0](out)
            norm = (self.norm3(x.float()) * (1 + cscale) + cshift).to(x.dtype)
            x = (x.float() + self.ffn(norm).float() * cgate).to(x.dtype)
        return x, k, v


class MLP(nn.Module):
    """diffusers TimestepEmbedding (silu) and PixArtAlphaTextProjection (gelu tanh): linear_1, act, linear_2."""

    def __init__(self, in_dim, dim, act, operation_settings):
        super().__init__()
        ops, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.linear_1 = ops.Linear(in_dim, dim, device=device, dtype=dtype)
        self.linear_2 = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.act = act

    def forward(self, x):
        if self.act == "silu":
            return self.linear_2(F.silu(self.linear_1(x)))
        return comfy.ops.linear_input_act(self.linear_2, self.linear_1(x), self.act)


class ConditionEmbedder(nn.Module):
    def __init__(self, dim, freq_dim, text_dim, operation_settings):
        super().__init__()
        self.freq_dim = freq_dim
        self.time_embedder = MLP(freq_dim, dim, "silu", operation_settings)
        self.time_proj = operation_settings["operations"].Linear(dim, dim * 6, device=operation_settings["device"], dtype=operation_settings["dtype"])
        self.text_embedder = MLP(text_dim, dim, "gelu_tanh", operation_settings)

    def forward(self, timestep, context):
        temb = self.time_embedder(timestep_embedding(timestep, self.freq_dim).unsqueeze(1).to(context.dtype))
        return temb, self.time_proj(F.silu(temb)).unflatten(2, (6, -1)), self.text_embedder(context)


class ConvState:
    """Temporal history of one strided causal convolution over a streamed video."""

    def __init__(self):
        self.history = None
        self.seen = 0


def causal_conv(conv, x, state):
    """Run a strided causal Conv3d on a consecutive chunk, emitting the outputs the chunk completes.
    The first frame is replicated backwards in time, as in the whole-video forward."""
    kernel, stride = conv.kernel_size[0], conv.stride[0]
    next_end = -(-state.seen // stride) * stride
    last_end = (state.seen + x.shape[2] - 1) // stride * stride
    if state.history is None:
        combined, start = x, state.seen
    else:
        combined, start = torch.cat((state.history, x), 2), state.seen - state.history.shape[2]
    needed = next_end - (kernel - 1)
    inp = combined[:, :, max(needed, 0) - start:last_end - start + 1]
    if needed < 0:
        inp = torch.cat((combined[:, :, :1].expand(-1, -1, -needed, -1, -1), inp), 2)
    state.history = combined[:, :, -(kernel - 1):].clone()
    state.seen += x.shape[2]
    return conv(inp) if last_end >= next_end else None  # a short chunk may complete no output yet


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim, eps, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, 1, 1, 1, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=1, keepdim=True) + self.eps)
        return (y * comfy.model_management.cast_to(self.weight, device=x.device)).to(x.dtype)


class LQProjState:
    def __init__(self):
        self.conv1 = ConvState()
        self.conv2 = ConvState()


class LQProj(nn.Module):
    """RGB frames at the output size to one token per 32x32 pixels and 4 frames: a 16x16 pixel unshuffle, two
    causal 4x3x3 convolutions each halving time (the second also halving space), and a linear projection."""

    def __init__(self, dim, hidden1, hidden2, operation_settings):
        super().__init__()
        ops, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.conv1 = ops.Conv3d(3 * 16 * 16, hidden1, kernel_size=(4, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), device=device, dtype=dtype)
        self.norm1 = ChannelRMSNorm(hidden1, 1e-6, device, dtype)
        self.conv2 = ops.Conv3d(hidden1, hidden2, kernel_size=(4, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1), device=device, dtype=dtype)
        self.norm2 = ChannelRMSNorm(hidden2, 1e-6, device, dtype)
        self.out_proj = ops.Linear(hidden2, dim, device=device, dtype=dtype)

    chunk = 0  # input frames per pass, 0 for the whole block; the node sets it when the full-size maps don't fit

    def forward(self, video, state):
        # causal along time, so a few frames at a time give the same tokens with a fraction of the full-size maps
        step = self.chunk or video.shape[2]
        out = [y for i in range(0, video.shape[2], step) if (y := self._forward(video[:, :, i:i + step], state)) is not None]
        return out[0] if len(out) == 1 else torch.cat(out, 1)

    def _forward(self, video, state):
        b, c, t, h, w = video.shape
        x = video.view(b, c, t, h // 16, 16, w // 16, 16).permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * 256, t, h // 16, w // 16)
        x = causal_conv(self.conv1, x, state.conv1)
        if x is None:
            return None
        x = causal_conv(self.conv2, F.silu(self.norm1(x)), state.conv2)
        if x is None:
            return None
        return self.out_proj(F.silu(self.norm2(x)).flatten(2).transpose(1, 2))


class BlockConditioning:
    """What the fixed timestep and prompt give one block every step: modulation, cross-attention K/V and, for the
    fast kernels, the modulation in the model dtype, the affine norm as a modulation and a unit residual scale."""

    def __init__(self, block, timestep_proj, context, kernels):
        device, dtype = context.device, context.dtype
        table = comfy.model_management.cast_to(block.scale_shift_table, dtype=torch.float32, device=device)
        self.modulation = tuple(part.squeeze(2) for part in (table.unsqueeze(0) + timestep_proj.float()).chunk(6, dim=2))
        self.cross_k = block.attn2.norm_k(block.attn2.to_k(context))
        self.cross_v = block.attn2.to_v(context)
        if kernels == "fast":
            self.modulation = tuple(part.to(dtype) for part in self.modulation)
            weight = comfy.model_management.cast_to(block.norm2.weight, dtype=torch.float32, device=device)
            self.norm2 = ((weight - 1).to(dtype), comfy.model_management.cast_to(block.norm2.bias, dtype=dtype, device=device))
            self.ones = torch.ones_like(self.norm2[1])


class Conditioning:
    def __init__(self, model, context, kernels):
        temb, timestep_proj, context = model.condition_embedder(torch.tensor([TIMESTEP], device=context.device), context)
        self.blocks = [BlockConditioning(block, timestep_proj, context, kernels) for block in model.blocks]
        table = comfy.model_management.cast_to(model.scale_shift_table, device=context.device)
        self.head = tuple(part.squeeze(2) for part in (table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2))


class ReCaVSRTransformer(nn.Module):
    def __init__(self, in_channels, dim, ffn_dim, heads, layers, freq_dim, text_dim, eps, lq_hidden, device=None, dtype=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.in_channels, self.dim, self.heads = in_channels, dim, heads
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        self.patch_embedding = operations.Conv3d(in_channels, dim, kernel_size=(1, PATCH, PATCH), stride=(1, PATCH, PATCH), device=device, dtype=dtype)
        self.condition_embedder = ConditionEmbedder(dim, freq_dim, text_dim, operation_settings)
        self.blocks = nn.ModuleList([Block(dim, ffn_dim, heads, eps, operation_settings) for _ in range(layers)])
        self.norm_out = operations.LayerNorm(dim, eps, elementwise_affine=False)
        self.proj_out = operations.Linear(dim, in_channels * PATCH * PATCH, device=device, dtype=dtype)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, dim, device=device, dtype=dtype))
        self.lq_proj = LQProj(dim, *lq_hidden, operation_settings)
        self.sr_proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def new_caches(self, storage=None):
        return [KVCache(action, ROUTER["initial"], ROUTER["stride"], ROUTER["anchors"], storage) for action in ROUTER["actions"]]

    def forward(self, noise, lq, recycle, start, conditioning, rope, caches, lq_state, window, kernels):
        """One block: noise [1, C, T, h, w] float32, lq [1, 3, frames, 16h, 16w] and recycle [1, C, T, h, w] in the
        model dtype. Returns the block's latents in float32."""
        frames, h, w = noise.shape[2:]
        hidden = self.patch_embedding(noise.to(self.dtype)).flatten(2).transpose(1, 2)
        hidden = hidden + self.lq_proj(lq, lq_state)
        hidden = hidden + self.sr_proj(self.patch_embedding(recycle).flatten(2).transpose(1, 2))
        origin, positions = rope.positions(start, frames)
        like = hidden.unflatten(2, (self.heads, -1))
        for block, cache, cond in zip(self.blocks, caches, conditioning.blocks):
            hidden, k, v = block(hidden, cond, rope, cache.history(like, origin), positions, window, kernels)
            cache.commit(k, v, frames, start)
        shift, scale = conditioning.head
        if kernels == "fast":
            hidden = self.proj_out(comfy.quant_ops.ck.adaln(hidden, scale, shift, self.norm_out.eps))
        else:
            hidden = self.proj_out((self.norm_out(hidden.float()) * (1 + scale) + shift).to(self.dtype))
        prediction = hidden.reshape(1, frames, h // PATCH, w // PATCH, 1, PATCH, PATCH, -1)
        prediction = prediction.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return noise - prediction.float()
