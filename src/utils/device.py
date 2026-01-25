"""Utility functions for device management and deterministic behavior."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. If 'auto', automatically select best available device.
        
    Returns:
        torch.device: The selected device.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        os.environ["PYTHONHASHSEED"] = str(seed)


def get_mixed_precision_dtype(device: torch.device) -> torch.dtype:
    """Get appropriate mixed precision dtype for the device.
    
    Args:
        device: The device to get dtype for.
        
    Returns:
        torch.dtype: The appropriate dtype for mixed precision.
    """
    if device.type == "cuda":
        # Use bf16 for newer GPUs, fp16 for older ones
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif device.type == "mps":
        # MPS supports fp16
        return torch.float16
    else:
        # CPU doesn't benefit from mixed precision
        return torch.float32


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get the model size in megabytes.
    
    Args:
        model: PyTorch model.
        
    Returns:
        float: Model size in MB.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
