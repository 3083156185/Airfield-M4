'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/PST.md
论文链接：https://arxiv.org/abs/2505.12772
'''     
   
import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings   
warnings.filterwarnings('ignore')  
from calflops import calculate_flops 
  
import torch    
import torch.nn as nn
import torch.nn.functional as F  

from engine.extre_module.ultralytics_nn.conv import Conv

class PSAttn(nn.Module):
    """    
    Pyramid Sparse Attention module for efficient multi-scale feature fusion in object detection.

    This module implements a cross-attention mechanism where queries are derived from lower-level features
    and keys/values from higher-level features. It provides a coarse attention output during training and,   
    optionally, a fine attention output during inference when `topk > 0`, enhancing performance by focusing 
    on key regions across scales.    

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        q (Conv): Convolution layer for computing queries from the input feature. 
        kv (Conv): Convolution layer for computing keys and values from the upper feature.     
        proj (Conv): Projection convolution layer for the output.  
        pe (Conv): Positional encoding convolution layer.  
        gate_conv1d (nn.Conv1d): 1D convolution for computing the gating mechanism.    

    Methods:
        forward: Applies pyramid sparse attention to the input tensors.

    Examples:  
        >>> attn = PSAttn(dim=256, num_heads=8, topk=4, tau=1.0)
        >>> x = torch.randn(1, 256, 32, 32)    
        >>> upper_feat = torch.randn(1, 256, 16, 16)
        >>> output = attn(x, upper_feat)     
        >>> print(output.shape)  
        torch.Size([1, 256, 32, 32])   
    """   

    def __init__(self, dim, num_heads, topk=4, tau=1.0):
        """  
        Initialize the Pyramid Sparse Attention module.   

        Args: 
            dim (int): Number of hidden channels.
            num_heads (int): Number of attention heads.
            topk (int): Number of top tokens to select for fine attention (set to 0 to disable).  
            tau (float): Temperature for Gumbel-Softmax (not used in the provided implementation).
        """     
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads
        self.topk = topk
        self.tau = tau
   
        # Convolution layers for queries, keys/values, projection, and positional encoding
        self.q = Conv(dim, all_head_dim, 1, act=False)  # Query convolution    
        self.kv = Conv(dim, all_head_dim * 2, 1, act=False)  # Key/Value convolution    
        self.proj = Conv(all_head_dim, dim, 1, act=False)  # Output projection
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)  # Positional encoding  
        self.gate_conv1d = nn.Conv1d(2 * head_dim, head_dim, kernel_size=1)  # Gating mechanism   

    @staticmethod
    def gumbel_softmax(logits):  
        """     
        Apply Gumbel-Softmax to approximate differentiable top-k selection.    
     
        Args:
            logits (torch.Tensor): Input logits for token scoring.

        Returns:   
            torch.Tensor: Soft weights for token selection.    
        """
        gumbels = -torch.empty_like(logits).exponential_().log()  # Generate Gumbel noise
        logits = logits + gumbels   
        return F.softmax(logits, dim=-1)  # Apply softmax to get soft weights
  
    def forward(self, x, upper_feat):
        """
        Process the input tensors through pyramid sparse attention.

        This method computes coarse attention using queries from `x` and keys/values from `upper_feat`. During  
        inference, if `topk > 0`, it additionally computes fine attention by selecting key regions from `x`   
        based on coarse attention scores, then fuses the outputs using a gating mechanism.
   
        Args:     
            x (torch.Tensor): Lower-level feature map; shape [B, C, H, W].   
            upper_feat (torch.Tensor): Higher-level feature map; shape [B, C, H/2, W/2].
    
        Returns:
            torch.Tensor: Fused feature map after attention; shape [B, C, H, W].  
        """
        B, C, H, W = x.shape   
        N = H * W
        _, _, H_up, W_up = upper_feat.shape     
  
        # Compute queries from lower-level feature
        q = self.q(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # [B, num_heads, N, head_dim]
        # Compute keys and values from higher-level feature
        kv = self.kv(upper_feat).view(B, self.num_heads, 2 * self.head_dim, H_up * W_up).permute(0, 1, 3, 2)
        k, v = kv.split(self.head_dim, dim=3)  # [B, num_heads, H_up*W_up, head_dim] each   

        # Compute coarse attention
        sim = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, N, H_up*W_up]
        attn = sim.softmax(dim=-1)  # Attention weights     
        coarse_out = (attn @ v)  # [B, num_heads, N, head_dim]
   
        # Fine attention (computed only during inference if topk > 0)   
        if 0 < self.topk <= H_up * W_up:     
            # Compute fine keys and values from lower-level feature
            f_kv = self.kv(x).view(B, self.num_heads, 2 * self.head_dim, N).permute(0, 1, 3, 2)    
            f_k, f_v = f_kv.split(self.head_dim, dim=3)  # [B, num_heads, N, head_dim] each     

            # Aggregate similarity scores over query dimension for token selection   
            global_sim = sim.mean(dim=2)  # [B, num_heads, H_up*W_up]
            soft_weights = PSAttn.gumbel_softmax(global_sim)  # [B, num_heads, H_up*W_up]
            topk_weights, topk_indices = torch.topk(soft_weights, k=self.topk, dim=-1)  # [B, num_heads, topk]   
  
            # Map selected indices from upper_feat to x (assuming 2x downsampling)
            scale = 2  
            h_idx = (topk_indices // W_up) * scale  # Row indices in x
            w_idx = (topk_indices % W_up) * scale   # Column indices in x     
            topk_x_indices = []
            for dh in range(scale):    
                for dw in range(scale): 
                    idx = (h_idx + dh) * W + (w_idx + dw)   
                    topk_x_indices.append(idx)
            topk_x_indices = torch.cat(topk_x_indices, dim=-1)  # [B, num_heads, 4*topk]
 
            # Gather fine keys and values using mapped indices
            topk_k = torch.gather(f_k, dim=2, index=topk_x_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            topk_v = torch.gather(f_v, dim=2, index=topk_x_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            # [B, num_heads, 4*topk, head_dim] each   
  
            # Compute fine attention   
            fine_attn = (q @ topk_k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, N, 4*topk]    
            fine_attn = fine_attn.softmax(dim=-1) 
            refined_out = fine_attn @ topk_v  # [B, num_heads, N, head_dim] 
     
            # Fuse coarse and refined outputs using gating
            fusion_input = torch.cat([coarse_out, refined_out], dim=-1)  # [B, num_heads, N, 2*head_dim]
            fusion_input = fusion_input.view(B * self.num_heads, N, -1).transpose(1, 2)  # [B*num_heads, 2*head_dim, N]     
            gate = self.gate_conv1d(fusion_input)  # [B*num_heads, head_dim, N] 
            gate = torch.sigmoid(gate).transpose(1, 2).view(B, self.num_heads, N, self.head_dim)    
            x = gate * refined_out + (1 - gate) * coarse_out  # Gated fusion     
        else:     
            x = coarse_out  # Use coarse output only if fine attention is disabled     
    
        # Reshape and apply positional encoding
        x = x.transpose(2, 3).reshape(B, self.all_head_dim, H, W)  # [B, all_head_dim, H, W]  
        v_reshaped = v.transpose(2, 3).reshape(B, self.all_head_dim, H_up, W_up)  # [B, all_head_dim, H_up, W_up] 
        v_pe = self.pe(v_reshaped)  # [B, dim, H_up, W_up]    
        v_pe = F.interpolate(v_pe, size=(H, W), mode='bilinear', align_corners=False)  # [B, dim, H, W]  
        x = x + v_pe  # Add positional encoding

        # Project back to original dimension
        return self.proj(x)  # [B, C, H, W]
   
class PSAttnBlock(nn.Module):
    """   
    Pyramid Sparse Attention block module for efficient feature fusion.

    This module implements a Pyramid Sparse Attention (PSAttn) mechanism combined with a 
    multi-layer perceptron (MLP) to enhance feature representation while maintaining
    computational efficiency. It is designed for feature fusion across different scales 
    in computer vision architectures.
   
    Attributes:
        attn (PSAttn): Pyramid Sparse Attention module for cross-scale feature fusion.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies attention and feed-forward processing to the input tensor. 
     
    Examples:
        >>> block = PSAttnBlock(dim=256, num_heads=8, mlp_ratio=2)  
        >>> x = torch.randn(1, 256, 32, 32)
        >>> upper_feat = torch.randn(1, 256, 16, 16)
        >>> output = block(x, upper_feat)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=2, topk = 0):
        """
        Initialize the Pyramid Sparse Attention block module.
     
        Args:
            dim (int): Number of input channels.     
            num_heads (int): Number of attention heads in the PSAttn module.    
            mlp_ratio (float): Expansion ratio for the MLP hidden dimension.    
            topk (int): Number of selected token in fine attention, set 0 for training stage.     
        """ 
        super().__init__()  
        self.attn = PSAttn(dim, num_heads=num_heads, topk=topk)  # Pyramid Sparse Attention module 
        mlp_hidden_dim = int(dim * mlp_ratio)  # Calculate hidden dimension for MLP    
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1),  # Expansion convolution
            Conv(mlp_hidden_dim, dim, 1, act=False)  # Projection back to input dimension   
        )    
  
        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        This method ensures that convolutional layers are initialized with weights drawn     
        from a truncated normal distribution, aiding in training stability and convergence.
 
        Args:
            m (nn.Module): Module to initialize.  
        """ 
        if isinstance(m, nn.Conv2d):   
            nn.init.trunc_normal_(m.weight, std=0.02)  # Truncated normal initialization 
            if m.bias is not None:  
                nn.init.constant_(m.bias, 0)  # Zero initialization for biases    

    def forward(self, x, upper_feat):
        """     
        Forward pass through the PSAttnBlock.

        Applies the Pyramid Sparse Attention mechanism followed by the MLP to the input tensor,
        using residual connections to preserve information flow.     
  
        Args:
            x (torch.Tensor): Input feature map; shape [B, C, H, W].
            upper_feat (torch.Tensor): Higher-level feature map; shape [B, C, H/2, W/2].   

        Returns:
            torch.Tensor: Output feature map after attention and feed-forward processing.   
        """
        x = x + self.attn(x, upper_feat)  # Apply attention with residual connection
        return x + self.mlp(x)  # Apply MLP with residual connection 

class PST(nn.Module):    
    """
    Pyramid Sparse Transformer (PST) module for enhanced feature fusion with attention mechanisms.

    This module integrates Pyramid Sparse Attention (PSA) blocks to fuse features from different scales,    
    leveraging cross-attention and dynamic token selection for efficient computation. It is designed to   
    enhance feature representations in tasks such as object detection and image classification.    
   
    Attributes:  
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.  
        cvup (Conv): Initial 1x1 convolution layer that reduces input channels from upper-level feature to hidden channels.  
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.     
        attnlayer_{i} (PSAttnBlock): Stacked Pyramid Sparse Attention blocks for feature fusion.    
    
    Examples:    
        >>> m = PST(512, 512, 256, n=1, mlp_ratio=2.0, e=0.5, k=0)    
        >>> x = (torch.randn(1, 512, 32, 32), torch.randn(1, 512, 16, 16))
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """    
  
    def __init__(self, c1, c_up, c2, n=1, mlp_ratio=2.0, e=0.5, k=0):
        """  
        Initialize the Pyramid Sparse Transformer module.
 
        Args:
            c1 (int): Number of input channels.   
            c_up (int): Number of input channels from upper-level feature. 
            c2 (int): Number of output channels.     
            n (int): Number of PSAttnBlock modules to stack.    
            mlp_ratio (float): Expansion ratio for MLP hidden dimension in PSAttnBlock.   
            e (float): Channel expansion ratio for hidden channels. 
            k (int): Number of top-k tokens in fine attention, set to 0 in training phase.
        """ 
        super().__init__()
        c_ = int(c2 * e)  # Calculate hidden channels     
        assert c_ % 32 == 0, "Hidden channels must be a multiple of 32."
    
        # Initial convolutions to reduce input and upper feature channels    
        self.cv1 = Conv(c1, c_, 1, 1)  # Convolution for input feature    
        self.cvup = Conv(c_up, c_, 1, 1)  # Convolution for upper-level feature
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # Final convolution to output channels    

        self.num_layers = n   
        for i in range(n):
            # Stack PSAttnBlock modules for feature fusion
            layer = PSAttnBlock(c_, c_ // 32, mlp_ratio, topk=k)
            self.add_module(f"attnlayer_{i}", layer)     

    def forward(self, x):  
        """  
        Forward pass through the PST module. 
     
        Processes the input feature and upper-level feature through initial convolutions,
        applies stacked PSAttnBlock modules for feature fusion, and concatenates the outputs
        before a final convolution to produce the output tensor.   
   
        Args:  
            x (tuple): Tuple containing two tensors:
                - x[0] (torch.Tensor): Input feature map; shape [B, c1, H, W].  
                - x[1] (torch.Tensor): Upper-level feature map; shape [B, c_up, H/2, W/2].
   
        Returns:
            torch.Tensor: Output feature map after processing; shape [B, c2, H, W].  
        """    
        # Extract input and upper-level features from tuple    
        upper_feat = x[1]
        x = self.cv1(x[0])

        # Apply initial convolution to upper-level feature
        upper_feat = self.cvup(upper_feat)
  
        # Initialize list to collect outputs from attention blocks     
        y = [x]  
        for i in range(self.num_layers):
            # Retrieve and apply the i-th attention block  
            layer = getattr(self, f"attnlayer_{i}")
            attened = layer(y[-1], upper_feat)
            y.append(attened)   

        # Concatenate all outputs and apply final convolution 
        y = self.cv2(torch.cat(y, 1))   
        return y 
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, channel_1, height_1, width_1 = 1, 32, 40, 40
    batch_size, channel_2, height_2, width_2 = 1, 64, 20, 20
    ouc_channel = 64
    inputs_1 = torch.randn((batch_size, channel_1, height_1, width_1)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height_2, width_2)).to(device)    

    # 可以参考进阶视频中的 engine/extre_module/custom_nn/featurefusion/mpca.py 的教程
    module = PST(channel_1, channel_2, ouc_channel, n=1).to(device)     

    outputs = module([inputs_1, inputs_2])  
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET) 

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module, 
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET) 
