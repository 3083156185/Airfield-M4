'''     
本文件由BiliBili：魔傀面具整理  
论文链接：https://arxiv.org/pdf/2412.16986
'''  
    
import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
 
import torch
import torch.nn as nn     
import torch.nn.functional as F  
import numpy as np  
from functools import partial
from engine.extre_module.custom_nn.block.MANet import MANet
from engine.extre_module.ultralytics_nn.conv import Conv, autopad
from engine.extre_module.ultralytics_nn.block import C3_Block, C2f_Block, C3k2_Block, MetaFormer_Block, MetaFormer_Mona, MetaFormer_SEFN, MetaFormer_Mona_SEFN, NCHW2NLC2NCHW    
from engine.extre_module.custom_nn.module.APBottleneck import APBottleneck 
from engine.extre_module.custom_nn.module.elgca import ELGCA_EncoderBlock
from engine.extre_module.custom_nn.module.fasterblock import Faster_Block_CGLU
from engine.extre_module.custom_nn.mlp.ConvolutionalGLU import ConvolutionalGLU
from engine.extre_module.custom_nn.transformer.AdaptiveSparseSA import AdaptiveSparseSA   
from engine.extre_module.custom_nn.transformer.PolaLinearAttention import PolaLinearAttention 
from engine.extre_module.custom_nn.mlp.EFFN import EFFN

# 本文件的视频教程是VideoBaiduYun.txt 中的 engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程.
# 本文件的视频教程是VideoBaiduYun.txt 中的 engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程.
# 本文件的视频教程是VideoBaiduYun.txt 中的 engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程.  
# 本文件的视频教程是VideoBaiduYun.txt 中的 engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程.
# 本文件的视频教程是VideoBaiduYun.txt 中的 engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程.  
# 本文件的视频教程是VideoBaiduYun.txt 中的 engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程.

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
  
    module_ = partial(APBottleneck, e=1.0)
    mlp_ = ConvolutionalGLU

    print(YELLOW + '-'*40 + ' C3 ' + '-'*40 + RESET)  
    module = C3_Block(in_channel, out_channel, module_, n=2).to(device)
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(YELLOW + '-'*40 + ' C2f ' + '-'*40 + RESET)     
    module = C2f_Block(in_channel, out_channel, module_, n=2).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
   
    print(YELLOW + '-'*40 + ' C3k2 ' + '-'*40 + RESET)  
    module = C3k2_Block(in_channel, out_channel, module_, n=2).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(YELLOW + '-'*40 + ' MetaFormer ' + '-'*40 + RESET)  
    module = MetaFormer_Block(in_channel, out_channel, token_mixer=module_, mlp=mlp_).to(device)
    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(YELLOW + '-'*40 + ' MetaFormer_Mona ' + '-'*40 + RESET)
    module = MetaFormer_Mona(in_channel, out_channel, token_mixer=module_, mlp=mlp_).to(device)    
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
 
    print(YELLOW + '-'*40 + ' MetaFormer_SEFN ' + '-'*40 + RESET)     
    module = MetaFormer_SEFN(in_channel, out_channel, token_mixer=module_).to(device)
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
 
    print(YELLOW + '-'*40 + ' MetaFormer_Mona_SEFN ' + '-'*40 + RESET)     
    module = MetaFormer_Mona_SEFN(in_channel, out_channel, token_mixer=module_).to(device) 
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(YELLOW + '-'*40 + ' MANet ' + '-'*40 + RESET)
    module = MANet(in_channel, out_channel, module_, n=2).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    # --------------------------- Self-Attention-Series input-size:N x C x H x W ---------------------------     
    print(ORANGE + '\n'*2 + '-'*40 + ' Self-Attention-Series input-size:N x C x H x W ' + '-'*40 + ORANGE)  
   
    module_ = partial(AdaptiveSparseSA, num_heads=8, sparseAtt=True)
 
    print(YELLOW + '-'*40 + ' MetaFormer ' + '-'*40 + RESET)
    module = MetaFormer_Block(in_channel, out_channel, token_mixer=module_, mlp=mlp_, selfatt=True).to(device)
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(YELLOW + '-'*40 + ' MetaFormer_Mona ' + '-'*40 + RESET)
    module = MetaFormer_Mona(in_channel, out_channel, token_mixer=module_, mlp=mlp_, selfatt=True).to(device) 
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(YELLOW + '-'*40 + ' MetaFormer_SEFN ' + '-'*40 + RESET)
    module = MetaFormer_SEFN(in_channel, out_channel, token_mixer=module_, selfatt=True).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
  
    print(YELLOW + '-'*40 + ' MetaFormer_Mona_SEFN ' + '-'*40 + RESET)
    module = MetaFormer_Mona_SEFN(in_channel, out_channel, token_mixer=module_, selfatt=True).to(device)
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    # --------------------------- Self-Attention-Series input-size:N x L(H*W) x C ---------------------------   
    print(ORANGE + '\n'*2 + '-'*40 + ' Self-Attention-Series input-size:N x L(H*W) x C ' + '-'*40 + ORANGE)  

    pola_module_ = partial(PolaLinearAttention, hw=(height, width), num_heads=8)    
    module_ = partial(NCHW2NLC2NCHW, module=pola_module_)

    print(YELLOW + '-'*40 + ' MetaFormer ' + '-'*40 + RESET)
    module = MetaFormer_Block(in_channel, out_channel, token_mixer=module_, mlp=mlp_, selfatt=True).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  
    
    print(YELLOW + '-'*40 + ' MetaFormer_Mona ' + '-'*40 + RESET)
    module = MetaFormer_Mona(in_channel, out_channel, token_mixer=module_, mlp=mlp_, selfatt=True).to(device)
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     
    
    print(YELLOW + '-'*40 + ' MetaFormer_SEFN ' + '-'*40 + RESET)
    module = MetaFormer_SEFN(in_channel, out_channel, token_mixer=module_, selfatt=True).to(device)     
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
  
    print(YELLOW + '-'*40 + ' MetaFormer_Mona_SEFN ' + '-'*40 + RESET)   
    module = MetaFormer_Mona_SEFN(in_channel, out_channel, token_mixer=module_, selfatt=True).to(device) 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)