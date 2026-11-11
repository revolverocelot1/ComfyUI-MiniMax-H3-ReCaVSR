import logging
import os
import time
import uuid

import torch
from tqdm import tqdm

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, io

from .recavsr import loader
from .recavsr.session import (DecoderStream, VideoStream, adain_color_fix, block_schedule, padded_size, prepare_block,
                              scaled_size, upscale_frames, wavelet_color_fix)
from .recavsr.video_io import ENCODERS, ArrayFrames, FileFrames, Writer, pick_encoder

log = logging.getLogger("ReCaVSR")
FOLDER = "recavsr"
ACTIVATION_BYTES_PER_TOKEN = int(2.6 * 1024 * 1024)  # peak minus weights, measured at 416 and 1612 tokens
KV_BYTES_PER_TOKEN = 66 * 2 * 3072 * 2  # the router keeps 66 planes of bf16 keys and values over the 30 blocks
# the input projection fed 4 frames at a time instead of a whole block: 16.5 -> 6.2 GiB at 5376x3072 (21 frames)
LQ_CHUNK_SAVING_PER_TOKEN = int(0.66 * 1024 * 1024)
LQ_CHUNK_FRAMES = 4
folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))


def _files():
    return [f for f in folder_paths.get_filename_list(FOLDER) if f.endswith(".safetensors")]


class ReCaVSRLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = _files()
        return io.Schema(
            node_id="ReCaVSRLoader",
            display_name="ReCaVSR Model Loader",
            category="ReCaVSR",
            description="Loads ReCaVSR (the Wan2.2-based one-step streaming video super-resolution transformer) with its "
                        "prompt embedding and FlashDecoder from models/recavsr.",
            inputs=[
                io.Combo.Input("transformer", options=files, default="transformer.safetensors" if "transformer.safetensors" in files else None,
                               tooltip="kopper/ReCaVSR transformer.safetensors (bf16), or the int8 ConvRot file made by "
                                       "tools/convert_int8.py (half the size, 1.5x faster, 45.5 dB against bf16; needs an NVIDIA "
                                       "GPU of compute capability 7.5 or newer). prompt.safetensors and flashdecoder.safetensors "
                                       "are read from the same folder."),
            ],
            outputs=[io.Custom("RECAVSR").Output(display_name="model")],
        )

    @classmethod
    def execute(cls, transformer) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise(FOLDER, transformer)
        folder = os.path.dirname(path)
        for name in ("prompt.safetensors", "flashdecoder.safetensors"):
            if not os.path.isfile(os.path.join(folder, name)):
                raise FileNotFoundError(f"{name} is missing next to {path}; download kopper/ReCaVSR into models/{FOLDER}")
        return io.NodeOutput(loader.load(path, os.path.join(folder, "flashdecoder.safetensors"), os.path.join(folder, "prompt.safetensors")))


class ReCaVSRUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ReCaVSRUpscale",
            display_name="ReCaVSR Upscale Video",
            category="ReCaVSR",
            description="One-step streaming video super-resolution: 21 frames, then 8 at a time, each block decoded and "
                        "encoded into the output video as it is generated, so a clip's length costs disk, not memory.",
            inputs=[
                io.Custom("RECAVSR").Input("model"),
                io.Video.Input("video", optional=True, tooltip="The video to upscale, read a block at a time; its audio and "
                                                               "frame rate carry over. Connect this or images."),
                io.Image.Input("images", optional=True, tooltip="Or the frames as images, in order; they are held in memory "
                                                                 "as images are. The output gets the fps below and no audio."),
                io.Float.Input("scale", default=4.0, min=0.5, max=8.0, step=0.25,
                               tooltip="Output size = input size x scale, each dimension rounded half up. The model was "
                                       "trained for 4x; 2x and 1.5x are the reference's other settings."),
                io.Combo.Input("color_fix", options=["none", "adain", "wavelet"], default="none",
                               tooltip="Per-frame colour match to the input: AdaIN matches mean and variance, wavelet keeps "
                                       "the input's low frequencies."),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, tooltip="Seed of the one-step noise."),
                io.Combo.Input("decoder", options=["flashdecoder", "vae"], default="flashdecoder",
                               tooltip="flashdecoder: the LQ-conditioned FlashDecoder, streaming and fast. vae: the stock "
                                       "Wan 2.2 VAE connected below, decoded at the end."),
                io.Vae.Input("vae", optional=True, tooltip="The Wan 2.2 VAE, for decoder = vae."),
                io.Combo.Input("kernels", options=["fast", "exact"], default="fast",
                               tooltip="fast: comfy-kitchen's fused norm, rope and residual kernels and retuned attention "
                                       "tiles; 47 dB against exact, no visible difference. The decoder's window kernel needs "
                                       "compute capability 8.0 (RTX 30 series or newer). exact: the reference's op sequence, "
                                       "bit for bit its output."),
                io.Boolean.Input("kv_offload", default=False,
                                 tooltip="Keep the attention history (the past latent frames every block attends) in "
                                         "RAM between steps instead of VRAM: about 0.8 MB per output token, 5 GB at 4x "
                                         "of 480p. For cards whose VRAM the output size does not fit; costs the "
                                         "transfer time each step. ComfyUI streams the weights by itself when they do "
                                         "not fit."),
                io.Combo.Input("encoder", options=["auto"] + list(ENCODERS), default="auto",
                               tooltip="How the output video is encoded as it is made. hevc nvenc / h264 nvenc: NVIDIA's "
                                       "video encoder, a unit beside the CUDA cores, at constant quality 19 / 17 (H.264 up to "
                                       "4096 pixels a side). h264: x264 on the CPU, CRF 14, plays everywhere. auto tries them "
                                       "in that order; a choice that can't run here falls back down the same order."),
                io.Float.Input("fps", default=30.0, min=1.0, max=240.0, step=0.001, optional=True,
                               tooltip="Frame rate of the output when images are connected; a video keeps its own."),
            ],
            outputs=[io.Video.Output(tooltip="The upscaled video, encoded to a file as it was made. Save Video stores it "
                                             "without re-encoding; Get Video Components gives the frames as images.")],
        )

    @classmethod
    def execute(cls, model, scale, color_fix, seed, decoder, kernels="fast", kv_offload=False, encoder="h264", fps=30.0,
                video=None, images=None, vae=None) -> io.NodeOutput:
        if (video is None) == (images is None):
            raise ValueError("ReCaVSR Upscale Video needs a video or images connected, one of the two")
        if decoder == "vae" and vae is None:
            raise ValueError("decoder = vae needs the Wan 2.2 VAE connected")
        if images is not None:
            source = ArrayFrames(images, fps)
        elif isinstance(video, InputImpl.VideoFromFile):
            source = FileFrames(video)
        else:  # a VIDEO made from components: its frames are already in memory
            parts = video.get_components()
            source = ArrayFrames(parts.images, parts.frame_rate)
            if parts.audio is not None:
                source.audio = (parts.audio["waveform"][0].float().numpy(), parts.audio["sample_rate"])
        total = source.total
        in_h, in_w = source.get(0, 1).shape[1:3]
        size = scaled_size(in_h, in_w, scale)
        height, width = padded_size(*size)
        blocks = list(block_schedule(total))
        log.info("ReCaVSR: %d frames %dx%d -> %dx%d (x%g; model size %dx%d), %d blocks, %s transformer, %s kernels, "
                 "decoder %s, colour fix %s, encoder %s%s", total, in_w, in_h, size[1], size[0], scale, width, height,
                 len(blocks), model.precision, kernels, decoder, color_fix, encoder, ", attention history in RAM" if kv_offload else "")
        # fit the activations to the card: whole blocks when they fit, else the input projection a few frames at a time
        # (one bf16 rounding step from the whole-block result), else the attention history in RAM too
        card = comfy.model_management.get_total_memory(model.transformer.load_device) * 0.85
        lq_chunk = activation_bytes(height, width, kv_offload) > card
        if lq_chunk and not kv_offload and activation_bytes(height, width, False, True) > card:
            kv_offload = True
            log.info("ReCaVSR: %dx%d does not fit this card's VRAM otherwise: attention history in RAM", size[1], size[0])
        need = activation_bytes(height, width, kv_offload, lq_chunk)
        if need > card:  # the activations alone; the weights can stream, these cannot
            fits = scale * (card / need) ** 0.5
            raise ValueError(f"ReCaVSR at {size[1]}x{size[0]} needs about {need / 2**30:.0f} GiB of VRAM for activations even "
                             f"streamed; this card has {card / 0.85 / 2**30:.0f} GiB. Lower scale to about {fits:.2f} or less, or "
                             "upscale a smaller input.")
        if lq_chunk:
            log.info("ReCaVSR: input projection %d frames at a time to fit %dx%d", LQ_CHUNK_FRAMES, size[1], size[0])
        chosen = pick_encoder(encoder, size[1], size[0], source.fps)
        if chosen != encoder:
            log.info("ReCaVSR: encoder %s -> %s for %dx%d", encoder, chosen, size[1], size[0])
        encoder = chosen
        model.transformer.model.lq_proj.chunk = LQ_CHUNK_FRAMES if lq_chunk else 0
        patchers = model.patchers() if decoder == "flashdecoder" else [model.transformer]
        started = time.perf_counter()
        comfy.model_management.load_models_gpu(patchers, memory_required=need)
        device = model.transformer.load_device
        folder = os.path.join(folder_paths.get_temp_directory(), "recavsr")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{uuid.uuid4().hex}.mp4")
        latents = []
        progress = comfy.utils.ProgressBar(len(blocks))
        with comfy.model_management.cuda_device_context(device):
            writer = Writer(path, size[1], size[0], source.fps, encoder, device, source.audio)
            stream = VideoStream(model.transformer.model, model.prompt, height, width, seed, device, kernels,
                                 torch.device("cpu") if kv_offload else None)
            decode = DecoderStream(model.decoder.model, model.latent_format, device, torch.bfloat16, kernels) if decoder == "flashdecoder" else None
            sync = torch.cuda.synchronize if device.type == "cuda" else lambda device: None
            sync(device)
            log.info("ReCaVSR: models on %s and conditioning ready in %.1f s", device, time.perf_counter() - started)
            transformer_time = decoder_time = 0.0
            bar = tqdm(total=total, desc="ReCaVSR", unit="frame", dynamic_ncols=True)
            for number, (start, valid, block) in enumerate(blocks, 1):
                lq = prepare_block(source.get(start, start + valid), 0, valid, block, size, device)
                t0 = time.perf_counter()
                latent = stream.step(lq)
                sync(device)
                t1 = time.perf_counter()
                if decode is None:
                    latents.append(latent.cpu())
                else:
                    offset, remaining = start, valid
                    for rgb in decode.step(latent, lq):
                        take = min(remaining, rgb.shape[2])
                        if take:
                            writer.put(finish(rgb[0, :, :take, :size[0], :size[1]].transpose(0, 1), source.get(offset, offset + take), color_fix))
                            offset += take
                            remaining -= take
                    sync(device)
                t2 = time.perf_counter()
                transformer_time += t1 - t0
                decoder_time += t2 - t1
                bar.update(valid)
                bar.set_postfix_str(f"block {number}/{len(blocks)}: transformer {(t1 - t0) * 1000:.0f} ms"
                                    + (f", decode {(t2 - t1) * 1000:.0f} ms" if decode is not None else ""))
                progress.update(1)
            bar.close()
            if decode is None:
                t0 = time.perf_counter()
                log.info("ReCaVSR: decoding %d latent frames with the VAE", sum(l.shape[2] for l in latents))
                rgb = vae.decode(model.latent_format.process_out(torch.cat(latents, 2)))
                rgb = rgb.reshape(-1, *rgb.shape[-3:])[:total, :size[0], :size[1]].permute(0, 3, 1, 2)
                for start in range(0, total, 4):  # four frames at a time on the device
                    writer.put(finish(rgb[start:start + 4].to(device) * 2 - 1, source.get(start, start + 4), color_fix))
                decoder_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            writer.close()
            if isinstance(source, FileFrames):
                source.close()
        elapsed = time.perf_counter() - started
        log.info("ReCaVSR: %d frames in %.1f s (%.2f fps): transformer %.1f s, decode %.1f s, encode wait %.1f s; %s",
                 total, elapsed, total / elapsed, transformer_time, decoder_time, time.perf_counter() - t0, path)
        return io.NodeOutput(InputImpl.VideoFromFile(path))


def finish(rgb, frames, color_fix):
    """Decoded frames [T, 3, H, W] in [-1, 1] on the device, colour fixed against the input frames, as uint8 [T, H, W, 3]."""
    if color_fix == "wavelet":
        x = wavelet_color_fix(rgb, frames)
    else:
        if color_fix == "adain":
            rgb = adain_color_fix(rgb, upscale_frames(frames, rgb.shape[-2:]).to(rgb.device))
        x = (rgb.float().clamp(-1, 1) + 1) * 0.5
    return x.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def activation_bytes(height, width, kv_offload=False, lq_chunk=False):
    per_token = ACTIVATION_BYTES_PER_TOKEN - (KV_BYTES_PER_TOKEN if kv_offload else 0) - (LQ_CHUNK_SAVING_PER_TOKEN if lq_chunk else 0)
    return (height // 32) * (width // 32) * per_token
