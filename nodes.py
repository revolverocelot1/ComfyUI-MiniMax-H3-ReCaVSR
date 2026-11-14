from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .recavsr_nodes import ReCaVSRLoader, ReCaVSRUpscale, activation_bytes
from .h3_gate_nodes import (
    NODE_CLASS_MAPPINGS as H3_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as H3_NODE_DISPLAY_NAME_MAPPINGS,
    H3UpscaleApprovalGate,
)


class ReCaVSRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ReCaVSRLoader, ReCaVSRUpscale]


async def comfy_entrypoint() -> ReCaVSRExtension:
    return ReCaVSRExtension()


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
