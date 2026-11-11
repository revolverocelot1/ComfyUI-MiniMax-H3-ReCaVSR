"""ReCaVSR's LR-conditioned FlashDecoder-S: a small streaming transformer that decodes one Wan2.2 latent frame
at a time, conditioned on the LQ frames at the output size. Ported from models/flashdecoder/ (Apache-2.0),
forward only. The release stores float32 weights and runs them under bfloat16 autocast: linears take the
compute dtype, RMSNorms run in float32, and that is written out here explicitly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention

from .attention import rotate_pairs, triton, tl

ROPE_DIMS = (16, 24, 24)  # temporal, height, width parts of the 64-wide head


def rope_tables(frames, height, width, first_frame, device):
    """cos/sin per rotary pair for every token of `frames` planes starting at `first_frame`, [tokens, 32], and the
    same as rotation matrices."""
    t = torch.arange(first_frame, first_frame + frames, device=device)
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    t, rows, cols = torch.meshgrid(t, rows, cols, indexing="ij")
    cos, sin = [], []
    for positions, dim in zip((t, rows, cols), ROPE_DIMS):
        # on the CPU like the reference's buffers: CUDA's pow rounds two of the three tables differently
        inv_freq = (1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))).to(device)
        angles = positions.flatten().float().unsqueeze(-1) * inv_freq
        cos.append(angles.cos())
        sin.append(angles.sin())
    cos, sin = torch.cat(cos, -1), torch.cat(sin, -1)
    # the third entry is the same rotation as [1, tokens, 1, pairs, 2, 2] matrices for comfy-kitchen's apply_rope1
    return cos, sin, torch.stack((cos, -sin, sin, cos), -1).view(1, cos.shape[0], 1, cos.shape[1], 2, 2)


if triton is not None:
    @triton.jit
    def _local_forward(Q, K, V, OUT, HEIGHT: tl.constexpr, WIDTH: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                       D: tl.constexpr, WH: tl.constexpr, WW: tl.constexpr, COLS: tl.constexpr, QF: tl.constexpr,
                       NF: tl.constexpr, Q_OFFSET: tl.constexpr, TW: tl.constexpr, R: tl.constexpr,
                       BH: tl.constexpr, BW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        n: tl.constexpr = HEIGHT * WIDTH
        tile_y, tile_x, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        qt = tile_y // triton.cdiv(HEIGHT, BH)
        tile_y = tile_y % triton.cdiv(HEIGHT, BH)
        batch, head = bh // HQ, bh % HQ
        kv_head = head // (HQ // HK)
        row = tl.arange(0, BM)
        qy, qx = tile_y * BH + row // BW, tile_x * BW + row % BW
        qi = qt * n + qy * WIDTH + qx
        valid_q = (qy < HEIGHT) & (qx < WIDTH)
        d = tl.arange(0, D)
        q = tl.load(Q + ((batch * HQ + head) * QF * n + qi[:, None]) * D + d[None, :], valid_q[:, None], 0)
        acc = tl.full((BM, D), 0, tl.float32)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        scale: tl.constexpr = D**-0.5 * 1.4426950408889634
        rows: tl.constexpr = min(WH + BH - 1, HEIGHT)
        for source in tl.static_range(TW * R):
            # Block-causal groups of R frames, bidirectional inside a group, TW groups back.
            plane = ((qt + Q_OFFSET) // R - TW + 1) * R + source
            valid_plane = (plane >= 0) & (plane < NF)
            if valid_plane:
                top, left = tl.maximum(tile_y * BH - WH // 2, 0), tl.maximum(tile_x * BW - WW // 2, 0)
                for offset in range(tl.cdiv(rows * COLS, BN)):
                    p = offset * BN + tl.arange(0, BN)
                    ky, kx = top + p // COLS, left + p % COLS
                    valid_k = (p < rows * COLS) & (ky >= 0) & (ky < HEIGHT) & (kx >= 0) & (kx < WIDTH) & valid_plane
                    ki = plane * n + ky * WIDTH + kx
                    k = tl.load(K + ((batch * HK + kv_head) * NF * n + ki[None, :]) * D + d[:, None], valid_k[None, :], 0)
                    scores = tl.dot(q, k).to(tl.float32) * scale
                    allowed = (valid_k[None, :]
                               & (ky[None, :] >= qy[:, None] - WH // 2) & (ky[None, :] < qy[:, None] + WH // 2)
                               & (kx[None, :] >= qx[:, None] - WW // 2) & (kx[None, :] < qx[:, None] + WW // 2))
                    scores = tl.where(allowed, scores, -float("inf"))
                    new_max = tl.maximum(maximum, tl.max(scores, 1))
                    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
                    probabilities = tl.exp2(scores - safe_max[:, None])
                    correction = tl.exp2(maximum - safe_max)
                    v = tl.load(V + ((batch * HK + kv_head) * NF * n + ki[:, None]) * D + d[None, :], valid_k[:, None], 0)
                    acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
                    denominator = denominator * correction + tl.sum(probabilities, 1)
                    maximum = new_max
        tl.store(OUT + ((batch * HQ + head) * QF * n + qi[:, None]) * D + d[None, :], acc / denominator[:, None], valid_q[:, None])


TILES = {"exact": (4, 16), "fast": (4, 8)}  # query tile (rows, columns): the reference's, and -48% measured on the RTX 3090


def local_attention(query, key, value, height, width, window, group_size, temporal_window, kernels="exact"):
    """Spatially windowed grouped-query attention, [B, heads, tokens, 64] in; the queries are the last
    `group_size` planes of the keys."""
    n = height * width
    qf, nf = query.shape[2] // n, key.shape[2] // n
    wh, ww = window
    if triton is not None and query.is_cuda and query.dtype in (torch.float16, torch.bfloat16) and query.shape[-1] == 64:
        if kernels == "fast" and group_size == 1:
            # one query plane over itself and the previous ones: comfy-kitchen's window kernel, cut-off borders
            b, heads, _, dim = query.shape
            planes = lambda t, frames: t.view(b, t.shape[1], frames, height, width, dim).permute(0, 2, 3, 4, 1, 5).contiguous()
            out = comfy.quant_ops.ck.window_attention(planes(query, qf), planes(key, nf), planes(value, nf), [wh, ww],
                                                      temporal_window=temporal_window, clamp=False)
            return out.permute(0, 4, 1, 2, 3, 5).reshape(b, heads, qf * n, dim)
        q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
        out = torch.empty_like(q)
        bh, bw = TILES[kernels]
        grid = (qf * triton.cdiv(height, bh), triton.cdiv(width, bw), q.shape[0] * q.shape[1])
        _local_forward[grid](q, k, v, out, height, width, q.shape[1], k.shape[1], q.shape[-1], wh, ww,
                             min(triton.next_power_of_2(ww + bw - 1), triton.next_power_of_2(width)),
                             qf, nf, nf - qf, temporal_window, group_size, bh, bw, bh * bw, 64, num_warps=4, num_stages=2)
        return out
    # without Triton or CUDA: 64 queries at a time against every key, masked; many times slower
    k = torch.arange(nf * n, device=query.device)
    kg, ky, kx = k // (group_size * n), (k % n) // width, k % width
    result = []
    for start in range(0, qf * n, 64):
        q = torch.arange(start, min(start + 64, qf * n), device=query.device)
        qg = (q // n + nf - qf) // group_size
        dy = ky[None, :] - ((q % n) // width)[:, None]
        dx = kx[None, :] - (q % width)[:, None]
        mask = ((kg[None, :] <= qg[:, None]) & (kg[None, :] > qg[:, None] - temporal_window)
                & (dy >= -wh // 2) & (dy < wh // 2) & (dx >= -ww // 2) & (dx < ww // 2))
        result.append(optimized_attention(query[:, :, start:start + 64], key, value, query.shape[1], mask=mask,
                                          skip_reshape=True, skip_output_reshape=True, enable_gqa=True))
    return torch.cat(result, dim=2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, heads, groups, eps, operation_settings):
        super().__init__()
        ops, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.heads, self.groups, self.head_dim = heads, groups, dim // heads
        self.query = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.key = ops.Linear(dim, groups * self.head_dim, device=device, dtype=dtype)
        self.value = ops.Linear(dim, groups * self.head_dim, device=device, dtype=dtype)
        self.key_norm = ops.RMSNorm(self.head_dim, eps=eps, device=device, dtype=dtype)
        self.value_norm = ops.RMSNorm(self.head_dim, eps=eps, device=device, dtype=dtype)
        self.output = ops.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, cache, query_rope, key_rope, spec, dtype, kernels):
        b, n, _ = x.shape
        x = x.to(dtype)
        query = self.query(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        key = self.key_norm(self.key(x).view(b, n, self.groups, self.head_dim).transpose(1, 2).float())
        value = self.value_norm(self.value(x).view(b, n, self.groups, self.head_dim).transpose(1, 2).float())
        if cache is not None:
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)
        if kernels == "fast":
            # the fused kernel takes [B, tokens, heads, dim] in bf16: fine for the queries, but the cached keys are
            # float32 and rounding them before the rotation costs 1.3 dB, so they keep the float32 path
            query = comfy.quant_ops.ck.apply_rope1(query.transpose(1, 2), query_rope[2]).transpose(1, 2)
        else:
            query = rotate_pairs(query, *query_rope[:2])
        rotated = rotate_pairs(key, *key_rope[:2]).to(query.dtype)
        out = local_attention(query, rotated, value.to(query.dtype), **spec, kernels=kernels)
        return self.output(out.transpose(1, 2).reshape(b, n, -1).to(dtype)), (key, value)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, operation_settings):
        super().__init__()
        ops, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.gate = ops.Linear(dim, hidden, device=device, dtype=dtype)
        self.up = ops.Linear(dim, hidden, device=device, dtype=dtype)
        self.down = ops.Linear(hidden, dim, device=device, dtype=dtype)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class StreamingBlock(nn.Module):
    def __init__(self, dim, heads, groups, hidden, eps, operation_settings):
        super().__init__()
        ops, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.attention_norm = ops.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.attention = GroupedQueryAttention(dim, heads, groups, eps, operation_settings)
        self.feed_forward_norm = ops.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.feed_forward = SwiGLU(dim, hidden, operation_settings)

    def forward(self, x, cache, query_rope, key_rope, spec, dtype, kernels):
        attended, cache = self.attention(self.attention_norm(x.float()), cache, query_rope, key_rope, spec, dtype, kernels)
        x = x + attended
        return x + self.feed_forward(self.feed_forward_norm(x.float()).to(dtype)), cache


class StreamingTransformer(nn.Module):
    """A block stack whose KV caches hold at most `max_frames` planes."""

    def __init__(self, depth, dim, heads, groups, hidden, max_frames, eps, operation_settings):
        super().__init__()
        self.max_frames = max_frames
        self.blocks = nn.ModuleList([StreamingBlock(dim, heads, groups, hidden, eps, operation_settings) for _ in range(depth)])

    def forward(self, x, caches, rope, frames, height, width, window, dtype, kernels):
        keep = (self.max_frames - frames) * height * width
        caches = [None if cache is None or keep == 0 else (cache[0][:, :, -keep:], cache[1][:, :, -keep:]) for cache in caches]
        previous = 0 if caches[0] is None else caches[0][0].shape[2] // (height * width)
        for key in ((previous + frames, 0), (frames, previous)):
            if key not in rope:
                rope[key] = rope_tables(key[0], height, width, key[1], x.device)
        key_rope, query_rope = rope[(previous + frames, 0)], rope[(frames, previous)]
        spec = dict(height=height, width=width, window=window, group_size=frames, temporal_window=self.max_frames // frames)
        new = []
        for block, cache in zip(self.blocks, caches):
            x, cache = block(x, cache, query_rope, key_rope, spec, dtype, kernels)
            new.append(cache)
        return x, new


class DecoderState:
    """Per-video caches of the two block stacks, their RoPE tables, and the count of latent frames decoded."""

    def __init__(self, decoder):
        self.backbone = [None] * len(decoder.backbone.blocks)
        self.refinement = [None] * len(decoder.temporal_refinement.blocks)
        self.rope = {}
        self.frame_index = 0


class FlashDecoder(nn.Module):
    def __init__(self, dim=512, depth=12, heads=8, groups=2, refinement_depth=2, latent_channels=48, window=(24, 24),
                 eps=1e-5, device=None, dtype=None, operations=None):
        super().__init__()
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        hidden = dim * 4
        self.dim, self.window = dim, tuple(window)
        self.latent_projection = operations.Linear(latent_channels, dim, device=device, dtype=dtype)
        self.backbone = StreamingTransformer(depth, dim, heads, groups, hidden, 2, eps, operation_settings)
        self.temporal_projection = operations.Linear(dim, 4 * dim, device=device, dtype=dtype)
        self.temporal_refinement = StreamingTransformer(refinement_depth, dim, heads, groups, hidden, 8, eps, operation_settings)
        self.spatial_projection = nn.Sequential(operations.Linear(dim, hidden, device=device, dtype=dtype), nn.SiLU(),
                                                operations.Linear(hidden, 3 * 16 * 16, device=device, dtype=dtype))
        self.lr_stem = operations.Linear(3 * 16 * 16, latent_channels, device=device, dtype=dtype)
        self.lr_early_projection = operations.Linear(4 * latent_channels, dim, bias=False, device=device, dtype=dtype)
        self.lr_late_projection = operations.Linear(latent_channels, dim, bias=False, device=device, dtype=dtype)

    def encode_lr(self, lr, first, dtype):
        """Per-frame LQ features: one token per 4 frames for the backbone, one per frame for the refinement."""
        b, _, frames, height, width = lr.shape
        pixels = lr.transpose(1, 2).reshape(b * frames, 3, height, width)
        packed = F.pixel_unshuffle(pixels, 16).permute(0, 2, 3, 1)
        features = self.lr_stem(packed.to(dtype))
        height, width = height // 16, width // 16
        features = features.reshape(b, frames, height, width, -1)
        if first:
            features = torch.cat((features[:, :1].expand(-1, 3, -1, -1, -1), features), dim=1)
        channels = features.shape[-1]
        grouped = features.reshape(b, -1, 4, height, width, channels).permute(0, 1, 3, 4, 2, 5).reshape(b, -1, 4 * channels)
        return self.lr_early_projection(grouped), self.lr_late_projection(features.reshape(b, -1, channels))

    def forward(self, latent, state, lr, dtype, kernels="exact"):
        """Decode one raw latent frame [B, 48, h, w] with its LQ frames [B, 3, 1 or 4, 16h, 16w] in [-1, 1].
        The first latent frame gives one RGB frame, every later one gives four."""
        b, _, height, width = latent.shape
        first = state.frame_index == 0
        early, late = self.encode_lr(lr, first, dtype)
        x = self.latent_projection(latent.permute(0, 2, 3, 1).to(dtype)).reshape(b, height * width, self.dim) + early
        x, state.backbone = self.backbone(x, state.backbone, state.rope, 1, height, width, self.window, dtype, kernels)
        x = self.temporal_projection(x).view(b, height, width, 4, self.dim).permute(0, 3, 1, 2, 4).reshape(b, 4 * height * width, self.dim) + late
        x, state.refinement = self.temporal_refinement(x, state.refinement, state.rope, 4, height, width, self.window, dtype, kernels)
        pixels = self.spatial_projection(x).view(b, 4, height, width, -1).permute(0, 1, 4, 2, 3).reshape(b * 4, -1, height, width)
        pixels = F.pixel_shuffle(pixels, 16).view(b, 4, 3, height * 16, width * 16).permute(0, 2, 1, 3, 4)
        state.frame_index += 1
        return pixels[:, :, 3:] if first else pixels
