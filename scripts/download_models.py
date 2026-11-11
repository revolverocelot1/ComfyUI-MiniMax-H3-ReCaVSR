#!/usr/bin/env python3
"""
Download ReCaVSR (Wan 2.2 TI2V-5B Streaming Video Super-Resolution) weights for ComfyUI.
"""
import argparse
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS = [
    ("PedroMarinhoDev/ReCaVSR-ComfyUI", "prompt.safetensors", "recavsr/prompt.safetensors"),
    ("PedroMarinhoDev/ReCaVSR-ComfyUI", "flashdecoder.safetensors", "recavsr/flashdecoder.safetensors"),
    ("PedroMarinhoDev/ReCaVSR-ComfyUI", "transformer_int8_convrot.safetensors", "recavsr/transformer_int8_convrot.safetensors"),
]

def main():
    parser = argparse.ArgumentParser(description="Download ReCaVSR models for ComfyUI")
    parser.add_argument("--models-dir", type=str, default="ComfyUI/models", help="Path to ComfyUI/models directory")
    parser.add_argument("--bf16", action="store_true", help="Also download the 10.6 GB bf16 transformer.safetensors")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    items = list(MODELS)
    if args.bf16:
        items.append(("kopper/ReCaVSR", "transformer.safetensors", "recavsr/transformer.safetensors"))

    for repo_id, filename, rel_dest in items:
        dest = models_dir / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"[OK] Cached: {dest}")
            continue
        print(f"[DL] Downloading {repo_id}/{filename} -> {dest} ...")
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(cached, dest)
        print(f"[OK] Saved: {dest}")

if __name__ == "__main__":
    main()
