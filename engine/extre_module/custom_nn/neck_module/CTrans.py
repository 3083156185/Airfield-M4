'''  
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/AAAI2022-CTrans.png     
engine/extre_module/module_images/AAAI2022-CTrans.md
论文链接：https://ojs.aaai.org/index.php/AAAI/article/view/20144
'''   

import warnings   
warnings.filterwarnings('ignore')    
from calflops import calculate_flops

import copy
import math  
import numpy as np 
import torch     
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import Conv2d, Dropout, LayerNorm, Softmax
from torch.nn.modules.utils import _pair

__all__ = ("ChannelTransformer",) 
 
class Channel_Embeddings(nn.Module):
    """Patch embedding used by ChannelTransformer for each input scale."""    
 
    def __init__(self, patchsize, img_size, in_channels): 
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)     
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.grid_size = int(math.sqrt(n_patches))
        self.patch_embeddings = nn.Sequential(     
            nn.MaxPool2d(kernel_size=5, stride=5),    
            Conv2d(     
                in_channels=in_channels,     
                out_channels=in_channels,
                kernel_size=patchsize // 5,   
                stride=patchsize // 5,
            ),     
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):   
        if x is None:
            return None
        pool_kernel_h, pool_kernel_w = _pair(self.patch_embeddings[0].kernel_size)
        kernel_h, kernel_w = self.patch_embeddings[1].kernel_size 
        if x.shape[-2] < pool_kernel_h or x.shape[-1] < pool_kernel_w:
            pooled = F.adaptive_avg_pool2d(x, (self.grid_size * kernel_h, self.grid_size * kernel_w)) 
        else:
            pooled = self.patch_embeddings[0](x)
        if pooled.shape[-2] < kernel_h or pooled.shape[-1] < kernel_w:  
            x = F.adaptive_avg_pool2d(pooled, (self.grid_size, self.grid_size))     
        else:     
            x = self.patch_embeddings[1](pooled)
        x = x.flatten(2).transpose(-1, -2)
        if x.shape[1] != self.position_embeddings.shape[1]: 
            pos = self.position_embeddings.transpose(1, 2)    
            pos = F.interpolate(pos, size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.position_embeddings
        return self.dropout(x + pos)
   
  
class Reconstruct(nn.Module):   
    """Project token sequences back to feature maps at the original pyramid scale."""  

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__() 
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
     
    def forward(self, x, output_size):  
        if x is None:
            return None
   
        batch, n_patch, hidden = x.size()
        h = w = int(np.sqrt(n_patch))  
        x = x.permute(0, 2, 1).contiguous().view(batch, hidden, h, w)
        x = F.interpolate(x, size=output_size, mode="nearest")   
        x = self.conv(x)  
        x = self.norm(x)
        return self.activation(x)
   

class Attention_org(nn.Module):
    """Cross-scale channel attention core used by ChannelTransformer."""    
     
    def __init__(self, vis, channel_num): 
        super().__init__()
        self.vis = vis
        self.kv_size = sum(channel_num)
        self.num_attention_heads = 4
  
        self.query1 = nn.ModuleList()  
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()  
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList() 

        for _ in range(2):
            self.query1.append(copy.deepcopy(nn.Linear(channel_num[0], channel_num[0], bias=False)))
            self.query2.append(copy.deepcopy(nn.Linear(channel_num[1], channel_num[1], bias=False)))
            self.query3.append(copy.deepcopy(nn.Linear(channel_num[2], channel_num[2], bias=False)))
            self.query4.append(    
                copy.deepcopy(nn.Linear(channel_num[3], channel_num[3], bias=False))  
                if len(channel_num) == 4   
                else nn.Identity()
            )     
            self.key.append(copy.deepcopy(nn.Linear(self.kv_size, self.kv_size, bias=False)))
            self.value.append(copy.deepcopy(nn.Linear(self.kv_size, self.kv_size, bias=False)))    
   
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)    
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False) if len(channel_num) == 4 else nn.Identity()
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)
   
    def _stack_queries(self, emb, query_layers):
        if emb is None:
            return None     
        multi_head_q = torch.stack([query(emb) for query in query_layers], dim=1)     
        return multi_head_q.transpose(-1, -2)  
 
    def forward(self, emb1, emb2, emb3, emb4, emb_all):    
        multi_head_q1 = self._stack_queries(emb1, self.query1)    
        multi_head_q2 = self._stack_queries(emb2, self.query2)
        multi_head_q3 = self._stack_queries(emb3, self.query3)
        multi_head_q4 = self._stack_queries(emb4, self.query4)   
     
        multi_head_k = torch.stack([key(emb_all) for key in self.key], dim=1)
        multi_head_v = torch.stack([value(emb_all) for value in self.value], dim=1)

        def attend(multi_head_q):  
            if multi_head_q is None: 
                return None
            attention_scores = torch.matmul(multi_head_q, multi_head_k) / math.sqrt(self.kv_size)
            attention_probs = self.softmax(self.psi(attention_scores))   
            attention_probs = self.attn_dropout(attention_probs)
            context = torch.matmul(attention_probs, multi_head_v.transpose(-1, -2))
            context = context.permute(0, 3, 2, 1).contiguous().mean(dim=3)
            return context, attention_probs  

        ctx1 = attend(multi_head_q1)     
        ctx2 = attend(multi_head_q2)   
        ctx3 = attend(multi_head_q3)    
        ctx4 = attend(multi_head_q4)
 
        weights = None
        if self.vis:
            weights = [     
                None if ctx1 is None else ctx1[1].mean(1),
                None if ctx2 is None else ctx2[1].mean(1), 
                None if ctx3 is None else ctx3[1].mean(1),   
                None if ctx4 is None else ctx4[1].mean(1),
            ]

        out1 = None if ctx1 is None else self.proj_dropout(self.out1(ctx1[0])) 
        out2 = None if ctx2 is None else self.proj_dropout(self.out2(ctx2[0]))
        out3 = None if ctx3 is None else self.proj_dropout(self.out3(ctx3[0]))
        out4 = None if ctx4 is None else self.proj_dropout(self.out4(ctx4[0]))  
        return out1, out2, out3, out4, weights
   

class Mlp(nn.Module): 
    """Feed-forward block inside each ChannelTransformer layer."""
     
    def __init__(self, in_channel, mlp_channel):
        super().__init__()   
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()    
        self.dropout = Dropout(0.0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x): 
        x = self.fc1(x)   
        x = self.act_fn(x)
        x = self.dropout(x) 
        x = self.fc2(x)  
        return self.dropout(x)
   
    
class Block_ViT(nn.Module):   
    """Single cross-scale transformer block."""

    def __init__(self, vis, channel_num):
        super().__init__()   
        expand_ratio = 4  
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)    
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)  
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6) if len(channel_num) == 4 else nn.Identity()     
        self.attn_norm = LayerNorm(sum(channel_num), eps=1e-6)    
        self.channel_attn = Attention_org(vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)     
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)    
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6) if len(channel_num) == 4 else nn.Identity()  
        self.ffn1 = Mlp(channel_num[0], channel_num[0] * expand_ratio)
        self.ffn2 = Mlp(channel_num[1], channel_num[1] * expand_ratio)     
        self.ffn3 = Mlp(channel_num[2], channel_num[2] * expand_ratio)
        self.ffn4 = Mlp(channel_num[3], channel_num[3] * expand_ratio) if len(channel_num) == 4 else nn.Identity()
     
    def forward(self, emb1, emb2, emb3, emb4):
        embcat = [emb for emb in (emb1, emb2, emb3, emb4) if emb is not None]
        emb_all = self.attn_norm(torch.cat(embcat, dim=2)) 
 
        cx1 = None if emb1 is None else self.attn_norm1(emb1)   
        cx2 = None if emb2 is None else self.attn_norm2(emb2)
        cx3 = None if emb3 is None else self.attn_norm3(emb3)     
        cx4 = None if emb4 is None else self.attn_norm4(emb4)    
    
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = None if emb1 is None else emb1 + cx1    
        cx2 = None if emb2 is None else emb2 + cx2   
        cx3 = None if emb3 is None else emb3 + cx3
        cx4 = None if emb4 is None else emb4 + cx4  

        x1 = None if cx1 is None else self.ffn1(self.ffn_norm1(cx1)) + cx1
        x2 = None if cx2 is None else self.ffn2(self.ffn_norm2(cx2)) + cx2     
        x3 = None if cx3 is None else self.ffn3(self.ffn_norm3(cx3)) + cx3
        x4 = None if cx4 is None else self.ffn4(self.ffn_norm4(cx4)) + cx4 
        return x1, x2, x3, x4, weights
     

class Encoder(nn.Module):
    """Stacked ChannelTransformer encoder."""

    def __init__(self, vis, channel_num):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([copy.deepcopy(Block_ViT(vis, channel_num)) for _ in range(1)])
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)  
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)   
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6) if len(channel_num) == 4 else nn.Identity()
   
    def forward(self, emb1, emb2, emb3, emb4):    
        attn_weights = [] 
        for layer_block in self.layer: 
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)    
            if self.vis:     
                attn_weights.append(weights)  
   
        emb1 = None if emb1 is None else self.encoder_norm1(emb1)
        emb2 = None if emb2 is None else self.encoder_norm2(emb2)
        emb3 = None if emb3 is None else self.encoder_norm3(emb3)
        emb4 = None if emb4 is None else self.encoder_norm4(emb4)    
        return emb1, emb2, emb3, emb4, attn_weights   


class ChannelTransformer(nn.Module):
    """Cross-scale channel transformer used ahead of the detection neck."""

    def __init__(self, channel_num=(64, 128, 256, 512), img_size=640, vis=False, patchSize=(40, 20, 10, 5)):
        super().__init__()
        channel_num = list(channel_num)
        patchSize = list(patchSize)

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]     
        self.patchSize_3 = patchSize[2]  
        self.patchSize_4 = patchSize[3]    
        self.embeddings_1 = Channel_Embeddings(self.patchSize_1, img_size=img_size // 8, in_channels=channel_num[0]) 
        self.embeddings_2 = Channel_Embeddings(self.patchSize_2, img_size=img_size // 16, in_channels=channel_num[1])   
        self.embeddings_3 = Channel_Embeddings(self.patchSize_3, img_size=img_size // 32, in_channels=channel_num[2])     
        self.embeddings_4 = (  
            Channel_Embeddings(self.patchSize_4, img_size=img_size // 64, in_channels=channel_num[3])     
            if len(channel_num) == 4     
            else nn.Identity()  
        )
        self.encoder = Encoder(vis, channel_num)   
        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))   
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2)) 
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))  
        self.reconstruct_4 = (
            Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))     
            if len(channel_num) == 4
            else nn.Identity()
        ) 
 
    def forward(self, en): 
        if len(en) == 3:
            en1, en2, en3 = en 
            en4 = None
        else: 
            en1, en2, en3, en4 = en   
     
        emb1 = None if en1 is None else self.embeddings_1(en1)   
        emb2 = None if en2 is None else self.embeddings_2(en2)     
        emb3 = None if en3 is None else self.embeddings_3(en3)
        emb4 = None if en4 is None else self.embeddings_4(en4)   

        encoded1, encoded2, encoded3, encoded4, _ = self.encoder(emb1, emb2, emb3, emb4)  
        x1 = None if en1 is None else self.reconstruct_1(encoded1, en1.shape[2:]) + en1    
        x2 = None if en2 is None else self.reconstruct_2(encoded2, en2.shape[2:]) + en2    
        x3 = None if en3 is None else self.reconstruct_3(encoded3, en3.shape[2:]) + en3  
        x4 = None if en4 is None else self.reconstruct_4(encoded4, en4.shape[2:]) + en4    
        return [x1, x2, x3, x4]   


if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size = 1

    print(RED + '-' * 20 + " ChannelTransformer 4-Scale " + '-' * 20 + RESET)     
    channels_4 = [64, 128, 256, 512]
    feats_4 = [   
        torch.randn((batch_size, channels_4[0], 80, 80)).to(device),
        torch.randn((batch_size, channels_4[1], 40, 40)).to(device),  
        torch.randn((batch_size, channels_4[2], 20, 20)).to(device), 
        torch.randn((batch_size, channels_4[3], 10, 10)).to(device), 
    ]
    module = ChannelTransformer(channel_num=channels_4, img_size=640, vis=False, patchSize=[40, 20, 10, 5]).to(device)

    outputs = module(feats_4)
    print(   
        GREEN     
        + f'inputs_P2.size:{feats_4[0].size()} inputs_P3.size:{feats_4[1].size()} '
          f'inputs_P4.size:{feats_4[2].size()} inputs_P5.size:{feats_4[3].size()} '
          f'outputs[0].size:{outputs[0].size()} outputs[1].size:{outputs[1].size()} '
          f'outputs[2].size:{outputs[2].size()} outputs[3].size:{outputs[3].size()}'
        + RESET
    )   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[feats_4], 
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True) 
    print(RESET)
  
    print(RED + '-' * 20 + " ChannelTransformer 3-Scale " + '-' * 20 + RESET)     
    channels_3 = [64, 128, 256]     
    feats_3 = [
        torch.randn((batch_size, channels_3[0], 80, 80)).to(device), 
        torch.randn((batch_size, channels_3[1], 40, 40)).to(device),
        torch.randn((batch_size, channels_3[2], 20, 20)).to(device),
    ]   
    module = ChannelTransformer(channel_num=channels_3, img_size=640, vis=False, patchSize=[40, 20, 10, 5]).to(device)

    outputs = module(feats_3)
    print(
        GREEN
        + f'inputs_P3.size:{feats_3[0].size()} inputs_P4.size:{feats_3[1].size()} '
          f'inputs_P5.size:{feats_3[2].size()} outputs[0].size:{outputs[0].size()} '    
          f'outputs[1].size:{outputs[1].size()} outputs[2].size:{outputs[2].size()} '
          f'outputs[3]:{outputs[3]}'
        + RESET  
    )   
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     args=[feats_3],
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)
