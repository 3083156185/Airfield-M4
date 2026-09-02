from pathlib import Path   

import torch


def load_text_cache_payload(payload, text_cache_file: str | None = None) -> dict:     
    if not isinstance(payload, dict):    
        raise TypeError(f"text cache must be a dict, got {type(payload)!r}") 

    required_keys = {"text_feats", "prompts", "categories", "prompt_template"} 
    missing_keys = required_keys - set(payload.keys()) 
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))   
        cache_name = f" {Path(text_cache_file)}" if text_cache_file is not None else ""    
        raise KeyError(f"text cache{cache_name} missing required keys: {missing}")     

    text_feats = payload["text_feats"]   
    if not isinstance(text_feats, torch.Tensor):   
        raise TypeError(f"text cache field 'text_feats' must be a tensor, got {type(text_feats)!r}")   

    prompts = payload["prompts"]  
    categories = payload["categories"]
    prompt_template = payload["prompt_template"]     
    if not isinstance(prompts, list) or any(not isinstance(prompt, str) for prompt in prompts): 
        raise TypeError("text cache field 'prompts' must be a list[str]")    
    if not isinstance(categories, list) or any(not isinstance(category, str) for category in categories):
        raise TypeError("text cache field 'categories' must be a list[str]")
    if not isinstance(prompt_template, str): 
        raise TypeError("text cache field 'prompt_template' must be a str")     

    return {
        "text_feats": text_feats.float(),
        "prompts": prompts,
        "categories": categories,  
        "prompt_template": prompt_template,  
    }    


def summarize_text_cache(text_cache: dict, text_cache_file: str | None = None) -> str:
    text_feats = text_cache["text_feats"]
    prompts = text_cache["prompts"]   
    categories = text_cache["categories"]
    prompt_template = text_cache["prompt_template"]     
    parts = []
    if text_cache_file is not None:   
        parts.append(f"path={Path(text_cache_file)}")
  
    parts.append(f"shape={tuple(text_feats.shape)}")   
    parts.append(f"dtype={text_feats.dtype}") 
   
    if text_feats.ndim >= 2:    
        parts.append(f"num_classes={text_feats.shape[-2]}")    
        parts.append(f"text_dim={text_feats.shape[-1]}")
        try:    
            norms = text_feats.float().norm(dim=-1)
            parts.append(f"mean_l2_norm={norms.mean().item():.4f}")    
        except Exception:     
            pass
  
    parts.append(f"num_prompts={len(prompts)}")
    parts.append(f"num_categories={len(categories)}") 
    parts.append(f"categories={categories!r}")     
    parts.append(f"prompt_template={prompt_template}")
    parts.append(f"prompts={prompts!r}")

    return ", ".join(parts)
