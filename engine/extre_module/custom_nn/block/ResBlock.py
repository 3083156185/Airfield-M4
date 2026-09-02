import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
   
import warnings  
warnings.filterwarnings('ignore') 
from calflops import calculate_flops
  
import torch  
import torch.nn as nn 
import torch.nn.functional as F

from functools import partial   
 
from engine.extre_module.custom_nn.attention.ema import EMA 
from engine.extre_module.custom_nn.attention.lsk import LSKBlock
from engine.extre_module.custom_nn.conv_module.CKConv import CKConv
from engine.extre_module.custom_nn.conv_module.dynamic_snake_conv import DySnakeConv
from engine.extre_module.ultralytics_nn.conv import Conv

class ResidualBlock(nn.Module):   
     
    def __init__(
        self, c1: int, c2: int, module1=partial(Conv, k=3, g=1), module2=partial(Conv, k=3, g=1), attention=None, shortcut: bool = True, e: float = 1.0    
    ):     
        """Initialize a standard bottleneck module.  
   
        Args:
            c1 (int): Input channels. 
            c2 (int): Output channels.   
            module_dict (dict):    
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
        """    
        super().__init__()    
        c_ = int(c2 * e)  # hidden channels 
        self.cv1 = module1(c1, c_)  
        self.cv2 = module2(c_, c2)     
        if attention is None:
            self.attention = nn.Identity()
        else:
            self.attention = attention(c2)
        self.add = shortcut and c1 == c2   

        self.shortcut = shortcut  
        self.e = e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))
   
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    
    device_id = 'cuda:0'     
    batch_size, in_channel, out_channel, height, width = 1, 128, 256, 160, 160

    torch_device = torch.device(device_id)     
    inputs_tensor = torch.randn((batch_size, in_channel, height, width)).to(torch_device)    

    module1_instance = partial(Conv, k=3, g=1)    
    module2_instance = partial(Conv, k=3, g=1)  
    # module1_instance = partial(CKConv)
    # module2_instance = partial(Conv, k=3)     
    attention_instance = partial(EMA, factor=16)  
    # attention_instance = partial(EMA)
   
    resblock = ResidualBlock(in_channel, out_channel, module1_instance, module2_instance, attention_instance, shortcut=True, e=0.5).to(torch_device) 
    resblock.eval()

    outputs = resblock(inputs_tensor)
    print(GREEN + f'inputs.size:{inputs_tensor.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=resblock, 
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True, 
                                     output_precision=4,  
                                     print_detailed=True)    
    print(RESET)
    
    def _module_cfg_to_str(module_cfg):
        if module_cfg is None:
            return "None"   
        if isinstance(module_cfg, partial):
            module = module_cfg.func
            kwargs = module_cfg.keywords or {}
        else:
            module = module_cfg  
            kwargs = {}
        module_name = getattr(module, "__name__", str(module))
        kwargs_text = ", ".join(f"{k}: {repr(v)}" for k, v in kwargs.items()) 
        return f"{{'module': {module_name}, 'param': {{{kwargs_text}}}}}"

    yaml_copy_template = (
        "{'module': ResidualBlock, 'param': {"
        f"'module1': {_module_cfg_to_str(module1_instance)}, " 
        f"'module2': {_module_cfg_to_str(module2_instance)}, "   
        f"'attention': {_module_cfg_to_str(attention_instance)}, "  
        f"'shortcut': {resblock.shortcut}, 'e': {resblock.e}"    
        "}}"   
    )
    print(BLUE + "YAML copy template for current ResidualBlock combination:" + RESET)
    print(yaml_copy_template)
