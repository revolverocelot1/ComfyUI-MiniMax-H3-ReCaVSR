"""Self-attention of the streaming transformer: each query plane sees its own block's planes and the routed
history planes, inside a spatial window, with the temporal RoPE applied relative to the rolling origin.

The Triton kernel is ReCaVSR's kernels/local_attention.py (Apache-2.0, forward only). A window that covers
the whole grid runs through ComfyUI's attention with a slot mask instead."""
import torch

from comfy.ldm.modules.attention import optimized_attention

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def rotate_pairs(x, cos, sin):
    """RoPE on interleaved (even, odd) pairs; cos/sin hold one value per pair and broadcast over x."""
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)


def apply_temporal(x, positions, cos, sin, span):
    """Rotate the temporal head dimensions of whole planes; x is [B, planes * span, heads, dim]."""
    tdim = cos.shape[-1] * 2
    planes = positions.numel()
    framed = x.view(x.shape[0], planes, span, x.shape[2], x.shape[3])
    c = cos[positions.long()].view(1, planes, 1, 1, -1)
    s = sin[positions.long()].view(1, planes, 1, 1, -1)
    return torch.cat((rotate_pairs(framed[..., :tdim], c, s), framed[..., tdim:]), -1).view_as(x)


if triton is not None:
    @triton.jit
    def _paired_local(Q, CK, CV, PK, PV, SLOTS, PPOS, QPOS, COS, SIN, OUT,
                      qs0: tl.constexpr, qs1: tl.constexpr, qs2: tl.constexpr,
                      cs0: tl.constexpr, cs1: tl.constexpr, cs2: tl.constexpr,
                      vs0: tl.constexpr, vs1: tl.constexpr, vs2: tl.constexpr,
                      ps0: tl.constexpr, ps1: tl.constexpr, ps2: tl.constexpr,
                      pv0: tl.constexpr, pv1: tl.constexpr, pv2: tl.constexpr,
                      HEIGHT: tl.constexpr, WIDTH: tl.constexpr, QT: tl.constexpr, CAP: tl.constexpr,
                      HEADS: tl.constexpr, DIM: tl.constexpr, TD: tl.constexpr, WH: tl.constexpr, WW: tl.constexpr,
                      BH: tl.constexpr, BW: tl.constexpr, BT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      ROWS: tl.constexpr, COLS: tl.constexpr):
        spatial_tile, temporal_tile, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        batch, head = bh // HEADS, bh % HEADS
        y0 = spatial_tile // triton.cdiv(WIDTH, BW) * BH
        x0 = spatial_tile % triton.cdiv(WIDTH, BW) * BW
        r = tl.arange(0, BM)
        qt = temporal_tile * BT + r // (BH * BW)
        qy = y0 + r // BW % BH
        qx = x0 + r % BW
        good_q = (qt < QT) & (qy < HEIGHT) & (qx < WIDTH)
        qi = qt * (HEIGHT * WIDTH) + qy * WIDTH + qx
        d = tl.arange(0, DIM)
        q = tl.load(Q + batch * qs0 + qi[:, None] * qs1 + head * qs2 + d[None, :], good_q[:, None], 0).to(tl.float32)
        # Swap rotary partners in registers instead of loading Q a second time.
        qp = tl.gather(q, tl.broadcast_to((d ^ 1)[None, :], (BM, DIM)), axis=1)
        pos = tl.load(QPOS + qt, good_q, 0)
        tc = tl.load(COS + pos[:, None] * (TD // 2) + d[None, :] // 2, good_q[:, None] & (d[None, :] < TD), 1)
        ts = tl.load(SIN + pos[:, None] * (TD // 2) + d[None, :] // 2, good_q[:, None] & (d[None, :] < TD), 0)
        sign = tl.where(d % 2 == 0, -1.0, 1.0)
        q = (q * tc + qp * ts * sign[None, :]).to(Q.dtype.element_ty)
        top = tl.minimum(tl.maximum(qy - WH // 2, 0), HEIGHT - WH)
        left = tl.minimum(tl.maximum(qx - WW // 2, 0), WIDTH - WW)
        base_y = tl.minimum(tl.maximum(y0 - WH // 2, 0), HEIGHT - WH)
        base_x = tl.minimum(tl.maximum(x0 - WW // 2, 0), WIDTH - WW)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        acc = tl.full((BM, DIM), 0, tl.float32)
        for plane in range(CAP + QT):
            if plane < CAP:
                slot = tl.load(SLOTS + plane)
                tpos = tl.load(PPOS + plane)
                kp, vp = PK, PV
                k0, k1, k2 = ps0, ps1, ps2
                v0, v1, v2 = pv0, pv1, pv2
            else:
                slot = plane - CAP
                tpos = tl.load(QPOS + slot)
                kp, vp = CK, CV
                k0, k1, k2 = cs0, cs1, cs2
                v0, v1, v2 = vs0, vs1, vs2
            # An empty physical slot has no visible keys; skip its entire plane.
            if slot >= 0:
                for chunk in range(triton.cdiv(ROWS * COLS, BN)):
                    p = chunk * BN + tl.arange(0, BN)
                    ky, kx = base_y + p // COLS, base_x + p % COLS
                    good_k = (p < ROWS * COLS) & (ky < HEIGHT) & (kx < WIDTH) & (slot >= 0)
                    ki = tl.maximum(slot, 0) * (HEIGHT * WIDTH) + ky * WIDTH + kx
                    k = tl.load(kp + batch * k0 + ki[None, :] * k1 + head * k2 + d[:, None], good_k[None, :], 0).to(tl.float32)
                    pair = tl.gather(k, tl.broadcast_to((d ^ 1)[:, None], (DIM, BN)), axis=0)
                    kc = tl.load(COS + tpos * (TD // 2) + d // 2, d < TD, 1)
                    ks = tl.load(SIN + tpos * (TD // 2) + d // 2, d < TD, 0)
                    k = (k * kc[:, None] + pair * ks[:, None] * sign[:, None]).to(Q.dtype.element_ty)
                    score = tl.dot(q, k).to(tl.float32) * (DIM**-0.5 * 1.4426950408889634)
                    allowed = (good_q[:, None] & good_k[None, :]
                               & (ky[None, :] >= top[:, None]) & (ky[None, :] < top[:, None] + WH)
                               & (kx[None, :] >= left[:, None]) & (kx[None, :] < left[:, None] + WW))
                    score = tl.where(allowed, score, -float("inf"))
                    next_max = tl.maximum(maximum, tl.max(score, 1))
                    safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
                    prob = tl.exp2(score - safe_max[:, None])
                    correction = tl.exp2(maximum - safe_max)
                    v = tl.load(vp + batch * v0 + ki[:, None] * v1 + head * v2 + d[None, :], good_k[:, None], 0)
                    acc = acc * correction[:, None] + tl.dot(prob.to(v.dtype), v)
                    denominator = denominator * correction + tl.sum(prob, 1)
                    maximum = next_max
        output = acc / tl.maximum(denominator[:, None], 1e-20)
        tl.store(OUT + ((batch * (QT * HEIGHT * WIDTH) + qi[:, None]) * HEADS + head) * DIM + d[None, :], output, good_q[:, None])


# query tile (rows, columns, planes) and warps: the reference's, and the faster one measured on the RTX 3090 (-18%)
TILES = {"exact": (4, 8, 2, 4), "fast": (8, 8, 2, 8)}


def windowed_attention(q, ck, cv, pk, pv, slots, past_positions, query_positions, cos, sin, height, width, window_h, window_w, kernels="exact"):
    """Spatially windowed attention over the current planes and the routed history, [B, tokens, heads, dim] in."""
    frames = q.shape[1] // (height * width)
    wh, ww = min(height, window_h), min(width, window_w)
    if triton is None or not q.is_cuda:
        return windowed_attention_torch(q, ck, cv, pk, pv, slots, past_positions, query_positions, cos, sin, height, width, wh, ww)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    bh, bw, bt, warps = TILES[kernels]
    _paired_local[(triton.cdiv(height, bh) * triton.cdiv(width, bw), triton.cdiv(frames, bt), q.shape[0] * q.shape[2])](
        q, ck, cv, pk, pv, slots, past_positions, query_positions, cos, sin, out,
        *q.stride()[:3], *ck.stride()[:3], *cv.stride()[:3], *pk.stride()[:3], *pv.stride()[:3],
        height, width, frames, slots.numel(), q.shape[2], q.shape[3], cos.shape[1] * 2, wh, ww,
        bh, bw, bt, bh * bw * bt, 64, min(height, wh + bh - 1), min(width, ww + bw - 1),
        num_warps=warps, num_stages=2, enable_fp_fusion=False)
    return out


def full_attention(q, ck, cv, pk, pv, slots, past_positions, query_positions, cos, sin, height, width, window_h, window_w, kernels="exact"):
    """Full-grid attention with the router's plane visibility, for grids the window covers."""
    span = height * width
    if slots.numel():
        index = slots.clamp_min(0).long()
        pk = pk.unflatten(1, (-1, span))[:, index].flatten(1, 2)
        pv = pv.unflatten(1, (-1, span))[:, index].flatten(1, 2)
    key = torch.cat((pk, ck), 1)
    value = torch.cat((pv, cv), 1)
    positions = torch.cat((past_positions, query_positions))
    q = apply_temporal(q, query_positions, cos, sin, span)
    key = apply_temporal(key, positions, cos, sin, span)
    valid = torch.cat((slots >= 0, torch.ones_like(query_positions, dtype=torch.bool)))
    mask = valid.repeat_interleave(span).view(1, 1, 1, -1)
    heads = q.shape[2]
    return optimized_attention(q.flatten(2), key.flatten(2), value.flatten(2), heads, mask=mask).unflatten(2, (heads, -1))


def windowed_attention_torch(q, ck, cv, pk, pv, slots, past_positions, query_positions, cos, sin, height, width, wh, ww):
    """The same attention without Triton: 64 queries at a time against every visible key with a mask of the
    clamped window. For a CPU or a GPU without Triton; many times slower than the kernel."""
    span = height * width
    if slots.numel():
        index = slots.clamp_min(0).long()
        pk = pk.unflatten(1, (-1, span))[:, index].flatten(1, 2)
        pv = pv.unflatten(1, (-1, span))[:, index].flatten(1, 2)
    key = apply_temporal(torch.cat((pk, ck), 1), torch.cat((past_positions, query_positions)), cos, sin, span)
    value = torch.cat((pv, cv), 1)
    q = apply_temporal(q, query_positions, cos, sin, span)
    heads = q.shape[2]
    planes = torch.cat((slots >= 0, torch.ones_like(query_positions, dtype=torch.bool)))
    k = torch.arange(key.shape[1], device=q.device)
    ky, kx = (k % span) // width, k % width
    visible = planes[k // span]
    out = []
    for start in range(0, q.shape[1], 64):
        i = torch.arange(start, min(start + 64, q.shape[1]), device=q.device)
        qy, qx = (i % span) // width, i % width
        top = (qy - wh // 2).clamp(0, height - wh)
        left = (qx - ww // 2).clamp(0, width - ww)
        mask = (visible[None] & (ky[None] >= top[:, None]) & (ky[None] < top[:, None] + wh)
                & (kx[None] >= left[:, None]) & (kx[None] < left[:, None] + ww))
        out.append(optimized_attention(q[:, start:start + 64].flatten(2), key.flatten(2), value.flatten(2), heads, mask=mask))
    return torch.cat(out, 1).unflatten(2, (heads, -1))
