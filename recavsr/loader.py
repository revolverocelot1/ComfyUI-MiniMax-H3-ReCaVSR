import logging
import os
import time

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils

from .dit import CONFIG, ReCaVSRTransformer
from .flashdecoder import DecoderState, FlashDecoder

log = logging.getLogger("ReCaVSR")


class ReCaVSR:
    """The loaded model set: the transformer and the FlashDecoder, each in its own patcher, and the prompt."""

    def __init__(self, transformer, decoder, prompt, precision):
        self.transformer = transformer
        self.decoder = decoder
        self.prompt = prompt
        self.precision = precision
        self.latent_format = comfy.latent_formats.Wan22()

    def patchers(self):
        return [self.transformer, self.decoder]


def _patcher(model, state_dict):
    # attribute access on purpose: main.py rebinds CoreModelPatcher to the dynamic patcher after imports
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=comfy.model_management.get_torch_device(),
                                                   offload_device=comfy.model_management.unet_offload_device())
    model.load_state_dict(state_dict, strict=True, assign=patcher.is_dynamic())
    return patcher


class _OpsConfig:
    def __init__(self, quant_config):
        self.quant_config = quant_config


def load(transformer_path, decoder_path, prompt_path):
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    dtype = torch.bfloat16
    # the reference runs its norms in float32 over bf16 weights, so the ops cast per call; a quantized file
    # (tools/convert_int8.py) gets ComfyUI's mixed-precision ops, which cast the same way
    operations = comfy.ops.manual_cast

    started = time.perf_counter()
    state_dict = comfy.utils.load_torch_file(transformer_path)
    quant_config = comfy.utils.detect_layer_quantization(state_dict, "")
    precision = "bf16"
    if quant_config is not None:
        operations = comfy.ops.pick_operations(dtype, dtype, load_device=load_device, model_config=_OpsConfig(quant_config))
        precision = "int8 ConvRot"
    log.info("ReCaVSR: loading %s (%s, %.1f GB)", os.path.basename(transformer_path), precision,
             sum(t.numel() * t.element_size() for t in state_dict.values()) / 2**30)
    transformer = ReCaVSRTransformer(**CONFIG, device=offload_device, dtype=dtype, operations=operations)
    comfy.model_management.archive_model_dtypes(transformer)
    transformer_patcher = _patcher(transformer, state_dict)
    del state_dict

    # the release keeps the decoder's float32 weights and runs its linears in bf16 under autocast
    state_dict = comfy.utils.load_torch_file(decoder_path)
    decoder = FlashDecoder(device=offload_device, dtype=torch.float32, operations=operations)
    decoder.new_state = lambda: DecoderState(decoder)
    comfy.model_management.archive_model_dtypes(decoder)
    decoder_patcher = _patcher(decoder, state_dict)
    del state_dict

    prompt = comfy.utils.load_torch_file(prompt_path)["positive"]
    log.info("ReCaVSR: transformer, %s and %s loaded into RAM in %.1f s; they go to %s at the first run",
             os.path.basename(decoder_path), os.path.basename(prompt_path), time.perf_counter() - started, load_device)
    return ReCaVSR(transformer_patcher, decoder_patcher, prompt, precision)
