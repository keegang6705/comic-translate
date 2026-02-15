import numpy as np
import onnxruntime as ort
import logging
from ..utils.device import (
    get_providers,
    create_onnx_session_with_fallback,
    should_use_gpu_for_inpainting,
)

from ..utils.inpainting import (
    norm_img,
    load_jit_model,
)
from ..utils.download import ModelDownloader, ModelID
from .base import InpaintModel
from .schema import Config

logger = logging.getLogger(__name__)


class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8

    def init_model(self, device, **kwargs):
        self.backend = kwargs.get("backend")
        if self.backend == "onnx":
            ModelDownloader.get(ModelID.LAMA_ONNX)
            onnx_path = ModelDownloader.primary_path(ModelID.LAMA_ONNX)
            self.session_gpu = create_onnx_session_with_fallback(onnx_path, device, logger)
            self.session_cpu = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            self.device_gpu = device
        else:
            ModelDownloader.get(ModelID.LAMA_JIT)
            local_path = ModelDownloader.primary_path(ModelID.LAMA_JIT) 
            self.model = load_jit_model(local_path, device)

    @staticmethod
    def is_downloaded() -> bool:
        return ModelDownloader.is_downloaded(ModelID.LAMA_JIT)

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image_n = norm_img(image)
        mask_n = norm_img(mask)
        mask_n = (mask_n > 0).astype('float32')
        backend = getattr(self, 'backend', 'torch')
        if backend == 'onnx':
            image_tensor = image_n[np.newaxis, ...]
            mask_tensor = mask_n[np.newaxis, ...]
            ort_inputs = {self.session_gpu.get_inputs()[0].name: image_tensor,
                          self.session_gpu.get_inputs()[1].name: mask_tensor}
            
            # Pre-check memory before inference
            use_gpu = should_use_gpu_for_inpainting(
                image.shape[0], image.shape[1], image.shape[2] if len(image.shape) > 2 else 1,
                safety_margin_gb=0.5, logger=logger
            )
            session = self.session_gpu if use_gpu else self.session_cpu
            
            try:
                inpainted = session.run(None, ort_inputs)[0]
            except RuntimeError as e:
                # Fallback to CPU if GPU inference fails
                error_msg = str(e).lower()
                if 'failed to allocate' in error_msg or 'out of memory' in error_msg:
                    logger.warning(f"GPU inference memory failed: {e}")
                    logger.info("Retrying with CPU/DRAM...")
                    inpainted = self.session_cpu.run(None, ort_inputs)[0]
                else:
                    raise
            
            cur_res = inpainted[0].transpose(1, 2, 0)
            cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
            return cur_res
        else:
            import torch  # noqa
            image_t = torch.from_numpy(image_n).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask_n).unsqueeze(0).to(self.device)
            inpainted_image = self.model(image_t, mask_t)
            cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
            return cur_res
