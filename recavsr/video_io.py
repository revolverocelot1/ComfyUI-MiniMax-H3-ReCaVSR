"""Streaming video in and out: the input read one block at a time, the output encoded as each block is decoded, so a
clip's length costs disk, not RAM. Decoded blocks become uint8 on the GPU, are copied on a side stream into pinned
buffers and encoded on a thread while the next block computes."""
import queue
import threading
from fractions import Fraction

import av
import numpy as np
import torch

# name: (PyAV codec, options), in fallback order; the NVENC ones run on the GPU's video encoder, beside the CUDA cores
ENCODERS = {
    "hevc nvenc": ("hevc_nvenc", {"preset": "p5", "tune": "hq", "rc": "vbr", "cq": "19", "b": "0"}),
    "h264 nvenc": ("h264_nvenc", {"preset": "p5", "tune": "hq", "rc": "vbr", "cq": "17", "b": "0"}),
    "h264": ("libx264", {"crf": "14", "preset": "medium"}),
}


def pick_encoder(choice, width, height, fps):
    """The first encoder that opens at this size, from the choice down the fallback order (auto starts at the top):
    NVENC is missing without an NVIDIA GPU or in a PyAV build without it, and its H.264 stops at 4096 pixels a side."""
    names = list(ENCODERS)
    for name in names[0 if choice == "auto" else names.index(choice):]:
        codec, options = ENCODERS[name]
        try:
            context = av.CodecContext.create(codec, "w")
            context.width, context.height, context.pix_fmt = width, height, "yuv420p"
            context.framerate, context.time_base = fps, 1 / fps
            context.options = dict(options)
            context.open()  # released when dropped; PyAV has no close for a bare context
            del context
            return name
        except (av.FFmpegError, ValueError):
            continue
    raise RuntimeError(f"no video encoder opens at {width}x{height}")


class ArrayFrames:
    """IMAGE frames [T, H, W, 3] in [0, 1], as uint8."""

    def __init__(self, images, fps):
        self.frames = images.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        self.total, self.fps, self.audio = len(self.frames), Fraction(fps).limit_denominator(1001), None

    def get(self, start, stop):
        return self.frames[start:stop]


class FileFrames:
    """A file-backed VIDEO decoded forward as rgb24, keeping only the frames a block still needs. Honours the
    VIDEO's trim window as ComfyUI's own decode does."""

    def __init__(self, video):
        self.video, self.total, self.fps = video, video.get_frame_count(), video.get_frame_rate()
        self.container = av.open(video.get_stream_source())
        stream = self.container.streams.video[0]
        stream.thread_type = "AUTO"
        start, duration = video.get_active_trim_window()
        self.start_pts, self.end_pts = int(start / stream.time_base), int((start + duration) / stream.time_base) if duration else None
        if self.start_pts:
            self.container.seek(self.start_pts, stream=stream)
        self.frames = (f for f in self.container.decode(stream) if f.pts is None or f.pts >= self.start_pts)
        self.cache, self.first = [], 0

    def get(self, start, stop):
        while self.first + len(self.cache) < stop:
            frame = next(self.frames, None)
            if frame is None or (self.end_pts is not None and frame.pts is not None and frame.pts >= self.end_pts):
                break
            self.cache.append(frame.to_ndarray(format="rgb24"))
        if start > self.first:
            del self.cache[:start - self.first]
            self.first = start
        return np.stack(self.cache[start - self.first:stop - self.first])

    @property
    def audio(self):
        """(samples [C, S] float32, sample rate) of the first audio stream within the trim window, or None."""
        with av.open(self.video.get_stream_source()) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            start, duration = self.video.get_active_trim_window()
            resampler = av.AudioResampler(format="fltp")
            chunks = [f.to_ndarray() for frame in container.decode(stream) for f in resampler.resample(frame)]
            if not chunks:
                return None
            samples, rate = np.concatenate(chunks, 1), stream.rate
            first = int(start * rate)
            return samples[:, first:first + int(duration * rate)] if duration else samples[:, first:], rate

    def close(self):
        self.container.close()


class Writer:
    """Encodes uint8 frames [n, H, W, 3] given on the GPU into an mp4. On CUDA the device-to-host copy runs on a side
    stream into a pinned slot and a thread encodes it, so neither holds up the next block."""

    def __init__(self, path, width, height, fps, encoder, device, audio=None, chunk=4, slots=3):
        codec, options = ENCODERS[encoder]
        self.container = av.open(path, "w")
        self.stream = self.container.add_stream(codec, rate=fps, options=dict(options))
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, "yuv420p"
        # every stream before the first packet: the header is written then. audio = (samples [C, S] float32, rate)
        self.audio = audio
        if audio is not None:
            self.audio_stream = self.container.add_stream("aac", rate=audio[1], layout="stereo" if audio[0].shape[0] > 1 else "mono")
        self.cuda = device.type == "cuda"
        self.slots = [torch.empty((chunk, height, width, 3), dtype=torch.uint8, pin_memory=self.cuda) for _ in range(slots)]
        self.free, self.work = queue.Queue(), queue.Queue()
        for i in range(slots):
            self.free.put(i)
        self.copy_stream = torch.cuda.Stream(device) if self.cuda else None
        self.error = None
        self.thread = threading.Thread(target=self._encode, daemon=True)
        self.thread.start()

    def put(self, frames):
        if self.error:
            raise self.error
        slot = self.free.get()
        n = frames.shape[0]
        if self.cuda:
            ready = torch.cuda.Event()
            ready.record()
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(ready)
                self.slots[slot][:n].copy_(frames, non_blocking=True)
                done = torch.cuda.Event()
                done.record(self.copy_stream)
            frames.record_stream(self.copy_stream)
        else:
            self.slots[slot][:n].copy_(frames)
            done = None
        self.work.put((slot, n, done))

    def _encode(self):
        while (item := self.work.get()) is not None:
            slot, n, done = item
            try:
                if done is not None:
                    done.synchronize()
                if self.error is None:
                    for frame in self.slots[slot][:n].numpy():
                        self.container.mux(self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
            except Exception as e:  # raised again in the node's thread by put() or close()
                self.error = e
            self.free.put(slot)

    def close(self):
        self.work.put(None)
        self.thread.join()
        if self.error:
            self.container.close()
            raise self.error
        self.container.mux(self.stream.encode())
        if self.audio is not None:
            samples, rate = self.audio
            out = self.audio_stream
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[:2], dtype=np.float32), format="fltp", layout=out.layout.name)
            frame.sample_rate = rate
            self.container.mux(out.encode(frame))
            self.container.mux(out.encode())
        self.container.close()
