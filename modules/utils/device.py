from __future__ import annotations

import os
from typing import Any, Mapping, Optional
import onnxruntime as ort
from .paths import get_user_data_dir


def torch_available() -> bool:
    """Check if torch is available without raising import errors."""
    try:
        import torch
        return True
    except ImportError:
        return False


def resolve_device(use_gpu: bool, backend: str = "onnx") -> str:
    """Return the best available device string for the specified backend.

    Args:
        use_gpu: Whether to use GPU acceleration
        backend: Backend to use ('onnx' or 'torch')

    Returns:
        Device string compatible with the specified backend
    """
    if not use_gpu:
        return "cpu"

    if backend.lower() == "torch":
        return _resolve_torch_device(fallback_to_onnx=True)
    else:
        return _resolve_onnx_device()


def _resolve_torch_device(fallback_to_onnx: bool = False) -> str:
    """Resolve the best available PyTorch device."""
    try:
        import torch
    except ImportError:
        # Torch not available, fallback to ONNX resolution if requested
        if fallback_to_onnx:
            return _resolve_onnx_device()
        return "cpu"

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Check for XPU (Intel GPU)
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass

    # Fallback to CPU
    return "cpu"


def _resolve_onnx_device() -> str:
    """Resolve the best available ONNX device."""
    providers = ort.get_available_providers() 

    if not providers:
        return "cpu"

    if "CUDAExecutionProvider" in providers:
        return "cuda"
    
    if "TensorrtExecutionProvider" in providers:
        return "tensorrt"

    if "CoreMLExecutionProvider" in providers:
        return "coreml"
    
    if "ROCMExecutionProvider" in providers:
        return "rocm"

    if "OpenVINOExecutionProvider" in providers:
        return "openvino"

    # Fallback to CPU
    return "cpu"

def tensors_to_device(data: Any, device: str) -> Any:
    """Move tensors in nested containers to device; returns the same structure.
    Supports dict, list/tuple, and tensors. Other objects are returned as-is.
    """
    try:
        import torch
    except Exception:
        # Torch is not available; return data unchanged
        return data

    # Map unknown device strings (onnx-driven) to torch-compatible device
    torch_device = device
    if isinstance(device, str):
        low = device.lower()
        if low in ("cpu", "cuda", "mps", "xpu"):
            torch_device = low
        else:
            # Unknown or ONNX-specific device -> fallback to cpu for torch tensors
            torch_device = "cpu"

    if isinstance(data, torch.Tensor):
        return data.to(torch_device)
    if isinstance(data, Mapping):
        return {k: tensors_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        seq = [tensors_to_device(v, device) for v in data]
        return type(data)(seq) if isinstance(data, tuple) else seq
    return data

def get_providers(device: Optional[str] = None) -> list[Any]:
    """Return a providers list for ONNXRuntime (optionally with provider options).

    Rules:
    - If device is the string 'cpu' (case-insensitive) -> return ['CPUExecutionProvider']
    - Otherwise return available providers with options for certain GPU providers
    - If no providers are available, fall back to ['CPUExecutionProvider']
    """
    try:
        available = ort.get_available_providers()
    except Exception:
        available = []

    if device and isinstance(device, str) and device.lower() == 'cpu':
        return ['CPUExecutionProvider']

    if not available:
        return ['CPUExecutionProvider']

    
    # Use user data directory for cache
    base_models_dir = os.path.join(get_user_data_dir(), "models")
    
    # OpenVINO cache
    ov_cache_dir = os.path.join(base_models_dir, 'onnx-gpu-cache', 'openvino')
    os.makedirs(ov_cache_dir, exist_ok=True)

    # TensorRT cache
    trt_cache_dir = os.path.join(base_models_dir, 'onnx-gpu-cache', 'tensorrt')
    os.makedirs(trt_cache_dir, exist_ok=True)

    # CoreML cache
    coreml_cache_dir = os.path.join(base_models_dir, 'onnx-gpu-cache', 'coreml')
    os.makedirs(coreml_cache_dir, exist_ok=True)

    provider_options = {
        'OpenVINOExecutionProvider': {
            'device_type': 'GPU',
            'precision': 'FP32',
            'cache_dir': ov_cache_dir,
        },
        'TensorrtExecutionProvider': {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': trt_cache_dir,
        },
        'CoreMLExecutionProvider': {
            'ModelCacheDirectory': coreml_cache_dir,
        }
    }

    configured: list[Any] = []
    for p in available:
        if p in provider_options:
            configured.append((p, provider_options[p]))
        else:
            configured.append(p)

    return configured


def is_gpu_available() -> bool:
    """Check if a valid GPU provider is available.
    
    Returns False if only AzureExecutionProvider and/or CPUExecutionProvider are present.
    Returns True if any other provider (CUDA, CoreML, etc.) is found.
    """
    try:
        providers = ort.get_available_providers()
    except Exception:
        return False

    ignored_providers = {'AzureExecutionProvider', 'CPUExecutionProvider'}
    available = set(providers)
    
    # If the only available providers are in the ignored list, return False
    # logic: if available is a subset of ignored_providers, then we have nothing else.
    if available.issubset(ignored_providers):
        return False
        
    return True


def get_available_gpu_memory_bytes() -> int:
    """Get available GPU memory in bytes for CUDA devices.
    
    Returns:
        Available GPU memory in bytes, or 0 if GPU not available or error occurs.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.mem_get_info()[0])  # Returns (free_memory, total_memory)
    except Exception:
        return 0


def estimate_inpaint_memory_bytes(height: int, width: int, channels: int = 3) -> int:
    """Estimate memory required for inpainting operation.
    
    Inpainting typically needs:
    - Input image: H*W*C*4 (float32)
    - Mask: H*W*4 (float32)
    - Intermediate buffers: ~4x input size (for model processing)
    - Output image: H*W*C*4 (float32)
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        channels: Number of channels (usually 3 for RGB)
        
    Returns:
        Estimated memory requirement in bytes
    """
    # Calculate sizes
    pixel_count = height * width
    image_bytes = pixel_count * channels * 4  # float32 = 4 bytes
    mask_bytes = pixel_count * 4  # float32 = 4 bytes
    
    # Estimate intermediate buffer size (typically 3-5x input size for model processing)
    # This is a conservative estimate
    intermediate_multiplier = 5.0
    intermediate_bytes = image_bytes * intermediate_multiplier
    
    # Total estimate with overhead (~20% extra for misc allocations)
    total = int((image_bytes + mask_bytes + intermediate_bytes) * 1.2)
    return total


def should_use_gpu_for_inpainting(height: int, width: int, channels: int = 3, 
                                   safety_margin_gb: float = 0.5, logger=None) -> bool:
    """Determine whether GPU should be used based on memory requirements.
    
    Args:
        height: Image height
        width: Image width  
        channels: Number of channels (default 3)
        safety_margin_gb: Safety margin in GB to reserve (default 0.5 GB)
        logger: Optional logger for debug messages
        
    Returns:
        True if enough GPU memory available, False to use CPU instead
    """
    if not is_gpu_available():
        return False
        
    required_bytes = estimate_inpaint_memory_bytes(height, width, channels)
    available_bytes = get_available_gpu_memory_bytes()
    safety_margin_bytes = int(safety_margin_gb * 1024 * 1024 * 1024)
    
    safe_available = available_bytes - safety_margin_bytes
    
    if logger:
        logger.debug(f"Inpainting memory check: required={required_bytes / 1024 / 1024:.1f}MB, "
                    f"available={available_bytes / 1024 / 1024:.1f}MB, "
                    f"safe={safe_available / 1024 / 1024:.1f}MB")
    
    use_gpu = required_bytes < safe_available
    
    if logger:
        if use_gpu:
            logger.debug("Using GPU for inpainting")
        else:
            logger.info(f"Insufficient GPU memory ({safe_available / 1024 / 1024:.1f}MB available, "
                       f"{required_bytes / 1024 / 1024:.1f}MB required). Using CPU/DRAM instead.")
    
    return use_gpu


def create_onnx_session_with_fallback(model_path: str, device: Optional[str] = None, logger=None):
    """Create an ONNX InferenceSession with automatic GPU-to-CPU fallback.
    
    When GPU providers are used, this function catches memory allocation errors and retries
    with CPU provider (DRAM) as a fallback.
    
    Args:
        model_path: Path to the ONNX model file
        device: Device to use ('cpu', 'cuda', etc.)
        logger: Logger instance for logging fallback events
        
    Returns:
        ort.InferenceSession configured with providers
    """
    providers = get_providers(device)
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    except RuntimeError as e:
        # Check if this is a memory allocation error
        error_msg = str(e).lower()
        if 'failed to allocate' in error_msg or 'out of memory' in error_msg:
            if logger:
                logger.warning(f"GPU memory allocation failed: {e}")
                logger.info("Falling back to CPU (DRAM) provider...")
            else:
                import logging as _logging
                _logging.getLogger(__name__).warning(f"GPU memory allocation failed, falling back to CPU: {e}")
            
            # Retry with CPU only
            try:
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                if logger:
                    logger.info("Successfully created session with CPU provider")
                return session
            except Exception as cpu_error:
                raise RuntimeError(f"Failed to create ONNX session on both GPU and CPU: {cpu_error}") from cpu_error
        else:
            raise
