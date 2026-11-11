"""Write the int8 ConvRot variant of ReCaVSR's transformer: the blocks' self-attention, cross-attention query and
output, and FFN linears as int8 with one scale per output row and a 256-wide block-Hadamard rotation, the format of
ComfyUI's int8 ConvRot model files (comfy-kitchen's TensorWiseINT8Layout, loaded by ComfyUI's mixed-precision ops).
Everything else (patch embedding, LQ projection, conditioning, cross-attention K/V, norms, head) stays bf16.

    python tools/convert_int8.py models/recavsr/transformer.safetensors [out.safetensors] [--device cuda:0]
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INT8 = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_out.0", "ffn.net.0.proj", "ffn.net.2")
GROUP = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out or args.src.replace(".safetensors", "_int8_convrot.safetensors")
    if os.path.exists(out):
        sys.exit(f"{out} exists")
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    marker = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()), dtype=torch.uint8)
    tensors, errors, start = {}, [], time.time()
    with safe_open(args.src, "pt") as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            module = key[:-len(".weight")] if key.endswith(".weight") else None
            if module is not None and module.startswith("blocks.") and module.split(".", 2)[2] in INT8:
                weight = tensor.to(args.device)
                qdata, params = TensorWiseINT8Layout.quantize(weight, per_channel=True, convrot=True, convrot_groupsize=GROUP)
                tensors[key] = qdata.cpu()
                tensors[module + ".weight_scale"] = params.scale.cpu()
                tensors[module + ".comfy_quant"] = marker.clone()
                if i % 40 == 0:
                    from comfy_kitchen.tensor.base import QuantizedTensor
                    back = QuantizedTensor(qdata, "TensorWiseINT8Layout", params).dequantize().float()
                    errors.append(((back - weight.float()).norm() / weight.float().norm()).item())
                    print(f"{i}/{len(keys)} {module}: relative weight error {errors[-1]:.4f}, {time.time() - start:.0f} s", flush=True)
            else:
                tensors[key] = tensor
    save_file(tensors, out + ".part", metadata={"format": "pt"})
    os.rename(out + ".part", out)
    print(f"wrote {out}: {os.path.getsize(out) / 2**30:.2f} GiB, mean relative weight error {sum(errors) / len(errors):.4f}")


if __name__ == "__main__":
    main()
