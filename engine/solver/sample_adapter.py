"""Utilities to keep detection engine compatible with tensor and multimodal dict samples."""

from __future__ import annotations     

from typing import Any

import torch    


def move_samples_to_device(samples: Any, device: torch.device, non_blocking: bool = True):
    """Move tensor-like sample containers to device while preserving structure."""
    if torch.is_tensor(samples): 
        return samples.to(device, non_blocking=non_blocking)    
    
    if isinstance(samples, dict):
        return {k: move_samples_to_device(v, device, non_blocking=non_blocking) for k, v in samples.items()}     
 
    if isinstance(samples, list):
        return [move_samples_to_device(v, device, non_blocking=non_blocking) for v in samples]    
    
    if isinstance(samples, tuple):  
        return tuple(move_samples_to_device(v, device, non_blocking=non_blocking) for v in samples)   

    return samples

  
def select_model_input(samples: Any, key: str = "rgb") -> torch.Tensor:  
    """Select model forward input from samples. A-plan defaults to RGB branch for multimodal dict."""
    if torch.is_tensor(samples):
        return samples  
 
    if isinstance(samples, dict):  
        if key not in samples:   
            raise KeyError(f"Missing model input key '{key}'. Available sample keys: {list(samples.keys())}")    
        model_input = samples[key]   
        if not torch.is_tensor(model_input):   
            raise TypeError(f"Model input samples['{key}'] must be a Tensor, got {type(model_input)}")
        return model_input

    raise TypeError(f"Unsupported sample type for model input selection: {type(samples)}")
 

def _unwrap_model(model: Any):  
    return model.module if hasattr(model, "module") else model


def is_multimodal_model(model: Any) -> bool:   
    """Detect models that expect multimodal dict input (e.g. DEIM_MG with Get*ModalityTensor)."""    
    if model is None: 
        return False 

    core_model = _unwrap_model(model)
    backbone = getattr(core_model, "backbone", None)
    if backbone is None or not hasattr(backbone, "children"):
        return False

    for module in list(backbone.children())[:4]:    
        if module.__class__.__name__ in {"GetRGBModalityTensor", "GetNPYModalityTensor"}:   
            return True
    return False 

 
def select_model_input_for_model(samples: Any, model: Any = None, key: str = "rgb"):
    """Select model input by sample type and model expectation."""    
    if torch.is_tensor(samples):  
        return samples
   
    if isinstance(samples, dict): 
        if is_multimodal_model(model):    
            return samples    
        return select_model_input(samples, key=key)
 
    return select_model_input(samples, key=key)
   

def select_plot_samples(samples: Any, key: str = "rgb"):
    """Select image-like sample for visualizer; defaults to RGB for multimodal dict."""
    if torch.is_tensor(samples):
        return samples

    if isinstance(samples, dict):
        if key not in samples:    
            raise KeyError(f"Missing plot key '{key}'. Available sample keys: {list(samples.keys())}")
        return samples[key]     

    raise TypeError(f"Unsupported sample type for plot selection: {type(samples)}")


def select_plot_samples_for_logging(samples: Any, keys=("rgb", "npy")):
    """Return ordered (modality, tensor) plot inputs while keeping tensor-only compatibility."""
    if torch.is_tensor(samples):
        return [("rgb", samples)]

    if isinstance(samples, dict):
        selected = []
        for key in keys:    
            if key not in samples:   
                continue 
            value = samples[key]
            if not torch.is_tensor(value):  
                raise TypeError(f"Plot samples['{key}'] must be a Tensor, got {type(value)}")
            selected.append((key, value))     
 
        if selected:  
            return selected

        for key, value in samples.items():     
            if torch.is_tensor(value):
                return [(str(key), value)]    

        raise KeyError(f"No tensor-like plot sample found. Available keys: {list(samples.keys())}")

    raise TypeError(f"Unsupported sample type for plot selection: {type(samples)}")
