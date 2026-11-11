from .recavsr_nodes import ReCaVSRLoader, ReCaVSRUpscale, ReCaVSRExtension, comfy_entrypoint
from .h3_gate_nodes import (
    NODE_CLASS_MAPPINGS as H3_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as H3_NODE_DISPLAY_NAME_MAPPINGS,
    H3UpscaleApprovalGate,
)

NODE_CLASS_MAPPINGS = {
    **H3_NODE_CLASS_MAPPINGS,
    "ReCaVSRLoader": ReCaVSRLoader,
    "ReCaVSRUpscale": ReCaVSRUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **H3_NODE_DISPLAY_NAME_MAPPINGS,
    "ReCaVSRLoader": "ReCaVSR Model Loader",
    "ReCaVSRUpscale": "ReCaVSR Upscale Video",
}
