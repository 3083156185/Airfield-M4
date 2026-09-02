from pathlib import Path    
from typing import Optional   
     
import torch    
import torch.nn as nn    
    
from ..core import register  
from ..logger_module import get_logger
from ..misc.ov_text_cache import load_text_cache_payload, summarize_text_cache
from .text_adapter import TextAdapter   

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
logger = get_logger(__name__)
  
@register()
class OVDEIM(nn.Module):   
    __inject__ = ["backbone", "encoder", "decoder"]
    __share__ = ["text_cache_file"]
    
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,   
        decoder: nn.Module,
        img_dim: int,   
        text_dim: int,     
        text_adapter_layers: int = 1,    
        text_cache_file: Optional[str] = None,
    ):     
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder  
        self.text_adapter = TextAdapter(text_dim, img_dim, text_adapter_layers) 
        self.text_cache_file = text_cache_file
        self.register_buffer("text_feats", torch.empty(0), persistent=False)
        self.text_prompts: list[str] = []
        self.text_categories: list[str] = []     
        self.prompt_template: Optional[str] = None
        self._forward_text_log_emitted = False
        if text_cache_file:
            self._load_text_cache(text_cache_file)

    def _load_text_cache(self, text_cache_file: str) -> None:  
        text_cache = load_text_cache_payload(torch.load(Path(text_cache_file), map_location="cpu"), text_cache_file)   
        self.text_feats = text_cache["text_feats"]
        self.text_prompts = text_cache["prompts"]  
        self.text_categories = text_cache["categories"]     
        self.prompt_template = text_cache["prompt_template"]
        logger.info(
            ORANGE + "OV text cache loaded for model: "
            + summarize_text_cache(text_cache, text_cache_file) + RESET  
        )  

    def _select_text_feats(self, targets=None) -> torch.Tensor:
        if targets: 
            per_sample_text_feats = [target.get("text_feats") for target in targets]
            if all(text_feats is not None for text_feats in per_sample_text_feats): 
                return torch.stack(per_sample_text_feats, dim=0)
        if self.text_feats.numel() == 0:    
            raise RuntimeError("OVDEIM requires text features from targets or text_cache_file.")  
        return self.text_feats  
   
    def forward(self, x, targets=None):   
        feats = self.backbone(x)
        feats = self.encoder(feats)
        text_feats = self.text_adapter(self._select_text_feats(targets).to(x.device))
        if text_feats.dim() == 2:     
            text_feats = text_feats.unsqueeze(0).repeat(x.shape[0], 1, 1)
        if not self._forward_text_log_emitted: 
            logger.info(     
                ORANGE + "OV model text features: "
                f"device={text_feats.device}, shape={tuple(text_feats.shape)}, dtype={text_feats.dtype}" + RESET
            )
            self._forward_text_log_emitted = True   
        return self.decoder(feats, targets=targets, text_feats=text_feats) 
     
    def deploy(self):    
        self.eval()  
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self
