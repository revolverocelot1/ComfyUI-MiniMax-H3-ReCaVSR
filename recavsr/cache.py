"""Per-video attention state of the streaming transformer: the routed KV history of each block and the
rolling temporal RoPE. Ported from ReCaVSR's runtime/cache.py and runtime/rope.py (Apache-2.0)."""
import torch

ROPE_WINDOW = 22  # latent frames the checkpoint was trained to see; positions roll inside it

# recent frames kept, periodic anchors kept
ACTIONS = {
    "none": (0, False),
    "recent_1": (1, False),
    "recent_2": (2, False),
    "recent_4": (4, False),
    "anchor": (0, True),
    "recent_2_anchor": (2, True),
    "recent_4_anchor": (4, True),
}


def retained_frames(indices, action, initial, stride, anchors):
    """The chronological union of the newest `recent` frames and the last `anchors` periodic anchors."""
    recent, use_anchor = ACTIONS[action]
    keep = set(indices[-recent:]) if recent else set()
    if use_anchor:
        keep.update([i for i in indices if i >= initial and (i - initial) % stride == 0][-anchors:])
    return tuple(i for i in indices if i in keep)


class KVCache:
    """Bounded KV storage of one block, indexed by absolute latent frame. Retention picks logical frames;
    a frame keeps its physical slot until evicted, so nothing is recomputed when the RoPE origin moves."""

    def __init__(self, action, initial, stride, anchors, storage=None):
        self.action, self.initial, self.stride, self.anchors = action, initial, stride, anchors
        recent, use_anchor = ACTIONS[action]
        self.capacity = recent + (anchors if use_anchor else 0)
        self.storage = storage  # a device to keep the planes on between steps (RAM for low-VRAM cards), or None
        self.key = self.value = None
        self.frame_to_slot = {}
        self.next_frame = 0
        self.metadata = {}

    def _metadata(self, slots, positions, device):
        key = (slots, positions)
        if key not in self.metadata:
            self.metadata[key] = (torch.tensor(slots, dtype=torch.int32, device=device),
                                  torch.tensor(positions, dtype=torch.int32, device=device))
        return self.metadata[key]

    def history(self, like, origin):
        """The physical buffers, their slot ids (-1 = unused) and positions relative to the RoPE origin."""
        batch, _, heads, dim = like.shape
        if self.key is None or not self.frame_to_slot:
            empty = like.new_empty((batch, 0, heads, dim))
            return (empty, empty) + self._metadata((), (), like.device)
        ids = tuple(sorted(self.frame_to_slot))
        positions = tuple(i - origin for i in ids)
        if min(positions) < 0 or max(positions) >= ROPE_WINDOW:
            raise ValueError("retained KV frames fall outside the rolling RoPE window")
        pad = self.capacity - len(ids)
        slots = tuple(self.frame_to_slot[i] for i in ids) + (-1,) * pad
        key, value = self.key, self.value
        if self.storage is not None:
            key, value = key.to(like.device, non_blocking=True), value.to(like.device, non_blocking=True)
        return (key, value) + self._metadata(slots, positions + (0,) * pad, like.device)

    def commit(self, key, value, frames, start):
        """Copy the newly retained planes of this block's spatially rotated keys and values."""
        if start != self.next_frame:
            raise ValueError("KV commits must be consecutive blocks")
        span = key.shape[1] // frames
        candidates = tuple(sorted(self.frame_to_slot)) + tuple(range(start, start + frames))
        selected = retained_frames(candidates, self.action, self.initial, self.stride, self.anchors)
        self.next_frame += frames
        if not self.capacity:
            return
        if self.key is None:
            shape = (key.shape[0], self.capacity * span, *key.shape[2:])
            if self.storage is None:
                self.key, self.value = key.new_empty(shape), value.new_empty(shape)
            else:
                pin = self.storage.type == "cpu" and key.is_cuda
                self.key = torch.empty(shape, dtype=key.dtype, device=self.storage, pin_memory=pin)
                self.value = torch.empty(shape, dtype=key.dtype, device=self.storage, pin_memory=pin)
        self.frame_to_slot = {i: s for i, s in self.frame_to_slot.items() if i in selected}
        free = iter(sorted(set(range(self.capacity)) - set(self.frame_to_slot.values())))
        for i in selected:
            if i in self.frame_to_slot:
                continue
            slot = next(free)
            offset = i - start
            # non-blocking: into pinned RAM when offloaded; the next step's upload follows on the same stream
            self.key[:, slot * span:(slot + 1) * span].copy_(key[:, offset * span:(offset + 1) * span], non_blocking=True)
            self.value[:, slot * span:(slot + 1) * span].copy_(value[:, offset * span:(offset + 1) * span], non_blocking=True)
            self.frame_to_slot[i] = slot


def rotary_table(dim, size, device):
    """cos/sin per rotary pair, [size, dim // 2], from float64 angles like diffusers' get_1d_rotary_pos_embed."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim))
    angles = torch.outer(torch.arange(size, dtype=torch.float64, device=device), freqs)
    return angles.cos().float(), angles.sin().float()


class RollingRoPE:
    """Spatial RoPE tables for the token grid, and temporal tables over the bounded rolling window."""

    def __init__(self, head_dim, height, width, device):
        self.height, self.width = height, width
        spatial = 2 * (head_dim // 6)
        temporal = head_dim - 2 * spatial
        self.temporal_cos, self.temporal_sin = rotary_table(temporal, ROPE_WINDOW, device)
        ch, sh = rotary_table(spatial, height, device)
        cw, sw = rotary_table(spatial, width, device)
        ones = torch.ones((height, width, temporal // 2), device=device)
        self.spatial_cos = torch.cat((ones, ch[:, None].expand(-1, width, -1), cw[None].expand(height, -1, -1)), -1).reshape(1, 1, height * width, 1, head_dim // 2)
        self.spatial_sin = torch.cat((torch.zeros_like(ones), sh[:, None].expand(-1, width, -1), sw[None].expand(height, -1, -1)), -1).reshape(1, 1, height * width, 1, head_dim // 2)
        self.query_positions = {}
        self.rotations = {}

    def spatial_rotation(self, frames):
        """The spatial RoPE as [1, frames * H * W, 1, pairs, 2, 2] rotation matrices for comfy-kitchen's apply_rope1."""
        if frames not in self.rotations:
            cos, sin = self.spatial_cos.flatten(0, 3), self.spatial_sin.flatten(0, 3)
            self.rotations[frames] = torch.stack((cos, -sin, sin, cos), -1).view(1, 1, -1, cos.shape[-1], 2, 2).expand(1, frames, -1, -1, -1, -1).reshape(1, -1, 1, cos.shape[-1], 2, 2)
        return self.rotations[frames]

    def positions(self, start, frames):
        """The RoPE origin of this block and the positions of its frames inside the window."""
        origin = max(0, start + frames - ROPE_WINDOW)
        values = tuple(range(start - origin, start - origin + frames))
        if values not in self.query_positions:
            self.query_positions[values] = torch.tensor(values, dtype=torch.int32, device=self.temporal_cos.device)
        return origin, self.query_positions[values]
