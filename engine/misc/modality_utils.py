import torch     
     

def normalize_tensor_minmax_per_sample(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:     
    """Normalize per sample with clamp-only for unit-range data and /255 otherwise."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}") 

    original_cls = type(tensor)
    original_ndim = tensor.ndim  
 
    if original_ndim == 2:
        batched = tensor.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3: 
        batched = tensor.unsqueeze(0)  
    elif original_ndim == 4:
        batched = tensor  
    else:  
        raise ValueError(f"Expected tensor with ndim in (2, 3, 4), got {original_ndim}")     

    batched = batched.float()     
    b = batched.shape[0]
    flat = batched.reshape(b, -1)     
    max_v = flat.max(dim=1).values.reshape(b, 1, 1, 1)   
    unit_range_mask = max_v <= 1.0
    normalized = torch.where(    
        unit_range_mask, 
        batched.clamp(0.0, 1.0),   
        (batched / 255.0).clamp(0.0, 1.0),    
    )    

    if original_ndim == 2:
        normalized = normalized[0, 0] 
    elif original_ndim == 3:
        normalized = normalized[0] 
   
    if original_cls is not torch.Tensor and issubclass(original_cls, torch.Tensor):
        try: 
            normalized = normalized.as_subclass(original_cls)
        except Exception:   
            pass     
  
    return normalized     
