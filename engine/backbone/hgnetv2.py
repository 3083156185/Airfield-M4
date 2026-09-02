"""  
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.     
"""  
    
import torch     
import torch.nn as nn
import torch.nn.functional as F
import os, re   
from .common import FrozenBatchNorm2d
from ..core import register     
from ..extre_module.custom_nn.attention.ema import EMA 
from ..extre_module.custom_nn.attention.simam import SimAM 
from ..misc.dist_utils import Multiprocess_sync, is_dist_available_and_initialized
from ..logger_module import get_logger
 
# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_  
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_
  
logger = get_logger(__name__)  
RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"

__all__ = ['HGNetv2'] 


def filter_loadable_state_dict(target_state_dict, source_state_dict):  
    return {
        key: value
        for key, value in source_state_dict.items()  
        if key in target_state_dict and value.shape == target_state_dict[key].shape
    }


class LearnableAffineBlock(nn.Module):
    """  
    可学习的仿射变换模块 (Learnable Affine Block)
    
    该模块对输入 `x` 进行仿射变换：  
        y = scale * x + bias     
    其中 `scale` 和 `bias` 是可训练参数。     

    适用于需要简单线性变换的场景，例如： 
    - 归一化调整 
    - 特征平移缩放    
    - 作为更复杂模型的一部分
    """
    def __init__(
            self,
            scale_value=1.0,  # 初始化缩放因子，默认为 1.0（保持输入不变）
            bias_value=0.0    # 初始化偏移量，默认为 0.0（无偏移）
    ):
        super().__init__()   
        # 定义可学习参数：缩放因子和偏移量
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True) 
 
    def forward(self, x):
        """
        前向传播：执行仿射变换

        参数:
        x (Tensor) - 输入张量

        返回:
        Tensor - 变换后的输出张量    
        """ 
        return self.scale * x + self.bias  
  
    
class ConvBNAct(nn.Module):   
    def __init__(    
            self,
            in_chs,    
            out_chs,     
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,   
            use_lab=False 
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab   
        if padding == 'same':   
            self.conv = nn.Sequential(    
                # nn.ZeroPad2d([0, 1, 0, 1]) 手动填充 右侧 1 个像素 和 底部 1 个像素，而左侧和顶部不填充。
	            # 这种方式适用于 kernel_size=2 的情况，使得卷积输出的尺寸与输入相同（在 stride=1 时）。     
                nn.ZeroPad2d([0, 1, 0, 1]), 
                nn.Conv2d(
                    in_chs,
                    out_chs,    
                    kernel_size,   
                    stride,
                    groups=groups, 
                    bias=False     
                )
            ) 
        else:
            self.conv = nn.Conv2d( 
                in_chs,    
                out_chs,   
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2, # 表示 PyTorch 默认的 SAME 填充，即对 左右、上下 进行均匀填充。  
                groups=groups,
                bias=False     
            )     
        self.bn = nn.BatchNorm2d(out_chs) 
        if self.use_act:
            self.act = nn.ReLU()  
        else:
            self.act = nn.Identity()  
        if self.use_act and self.use_lab:     
            self.lab = LearnableAffineBlock()
        else:   
            self.lab = nn.Identity()

    def forward(self, x):     
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x  

   
class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
    ):  
        super().__init__()    
        self.conv1 = ConvBNAct(
            in_chs, 
            out_chs, 
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )  
        self.conv2 = ConvBNAct(    
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,  
        )
 
    def forward(self, x):   
        x = self.conv1(x)
        x = self.conv2(x)  
        return x
 

class StemBlock(nn.Module):    
    """ 
    HGNetv2 的 Stem Block，作为网络的输入处理模块。   
    
    主要作用：
    1. 进行初步特征提取（stem1）。 
    2. 通过两条不同路径（stem2a 和 stem2b）处理特征，并进行特征融合。    
    3. 采用池化（pool）和卷积（stem3、stem4）进一步提取特征。
    4. 适用于深度神经网络的前几层，以减少计算量并提高模型稳定性。     
 
    Args:   
        in_chs (int): 输入通道数。
        mid_chs (int): 中间层通道数。    
        out_chs (int): 输出通道数。
        use_lab (bool): 是否使用 Lab 颜色空间的特殊处理（默认为 False)。  
    """
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False): 
        super().__init__()  
        # 初始卷积层，进行基本特征提取，步长为2，实现下采样 
        self.stem1 = ConvBNAct( 
            in_chs,     
            mid_chs,    
            kernel_size=3,  
            stride=2,     
            use_lab=use_lab,   
        )
    
        # 分支 1：stem2a 和 stem2b 组成一个小的子路径
        # stem2a：将 mid_chs 降维到 mid_chs//2 
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,   
            kernel_size=2,
            stride=1,
            use_lab=use_lab,  
        )   
        # stem2b：再将 mid_chs//2 提升回 mid_chs  
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs, 
            kernel_size=2,   
            stride=1,
            use_lab=use_lab,    
        )
        
        # 分支 2：最大池化层，对 stem1 输出进行下采样    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)     
    
        # 合并两个分支后的卷积层 
        # stem3：对拼接后的特征进行卷积，步长为 2 进行下采样
        self.stem3 = ConvBNAct( 
            mid_chs * 2,  # 两个分支的特征图通道数加起来
            mid_chs,
            kernel_size=3,     
            stride=2, 
            use_lab=use_lab,
        )
    
        # stem4：最后一个 1x1 卷积层，调整通道数为 out_chs  
        self.stem4 = ConvBNAct( 
            mid_chs,    
            out_chs,  
            kernel_size=1,
            stride=1,    
            use_lab=use_lab,
        )
 
    def forward(self, x):     
        """
        前向传播过程：    
        1. 通过 stem1 进行特征提取。   
        2. 通过 stem2a 和 stem2b 进行一个额外的特征变换路径（右分支）。   
        3. 通过最大池化（左分支）。     
        4. 拼接两个分支的特征（cat）。
        5. 通过 stem3 和 stem4 进行进一步特征提取。   
 
        Args:
            x (torch.Tensor): 输入特征图 (batch_size, in_chs, H, W) 
  
        Returns:    
            torch.Tensor: 处理后的特征图 (batch_size, out_chs, H // 4, W // 4)
        """  
        # Step 1: 初步特征提取   
        x = self.stem1(x)
        
        # Step 2: 右分支，使用额外的 2x2 卷积进行特征处理
        x = F.pad(x, (0, 1, 0, 1))  # 右侧和底部填充 1 个像素
        x2 = self.stem2a(x)     
        x2 = F.pad(x2, (0, 1, 0, 1))  # 右侧和底部再次填充
        x2 = self.stem2b(x2)  # 经过第二个小卷积
        
        # Step 3: 左分支，使用最大池化进行特征下采样  
        x1 = self.pool(x)    

        # Step 4: 组合左右分支的特征     
        x = torch.cat([x1, x2], dim=1)  # 在通道维度（dim=1）拼接
        
        # Step 5: 进一步处理
        x = self.stem3(x)  # 进行 3x3 卷积    
        x = self.stem4(x)  # 进行 1x1 卷积    
   
        return x    
  
    
class EseModule(nn.Module): 
    def __init__(self, chs):     
        super().__init__() 
        self.conv = nn.Conv2d(
            chs,     
            chs,  
            kernel_size=1,   
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)  
        return torch.mul(identity, x)
  

class HG_Block(nn.Module):
    def __init__(     
            self,   
            in_chs,        # 输入通道数
            mid_chs,       # 中间通道数（隐含层）   
            out_chs,       # 输出通道数
            layer_num,     # 卷积层的数量     
            kernel_size=3, # 卷积核大小 
            residual=False, # 是否使用残差连接
            light_block=False, # 是否使用轻量级卷积块（深度可分离卷积）     
            use_lab=False, # 是否使用可学习的仿射变换模块     
            agg='ese',     # 选择特征聚合方式（'se' 或 'ese' 或 'ema' 或 'simam'）
            drop_path=0.,  # DropPath 比例（用于正则化）  
    ):    
        super().__init__() 
        self.residual = residual     

        # 构建卷积层序列    
        self.layers = nn.ModuleList()   
        for i in range(layer_num):    
            if light_block: 
                # 轻量级卷积块（深度可分离卷积）    
                self.layers.append( 
                    LightConvBNAct(   
                        in_chs if i == 0 else mid_chs, 
                        mid_chs,
                        kernel_size=kernel_size,    
                        use_lab=use_lab,  
                    )     
                ) 
            else:
                # 标准卷积块
                self.layers.append(    
                    ConvBNAct( 
                        in_chs if i == 0 else mid_chs, 
                        mid_chs,  
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,  
                    )  
                )   
  
        # 计算聚合层的输入通道数
        total_chs = in_chs + layer_num * mid_chs    
     
        # 特征聚合方式     
        if agg == 'se':  # SE注意力机制
            aggregation_squeeze_conv = ConvBNAct(
                total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab    
            )
            aggregation_excitation_conv = ConvBNAct(     
                out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab 
            )     
            self.aggregation = nn.Sequential(  
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            ) 
        elif agg == 'ese':  # 默认ESE注意力机制
            aggregation_conv = ConvBNAct(    
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab
            )
            att = EseModule(out_chs)  # ESE注意力模块   
            self.aggregation = nn.Sequential(   
                aggregation_conv, 
                att,     
            )
        elif agg == 'ema': # ema注意力 
            aggregation_conv = ConvBNAct(    
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab
            )
            att = EMA(out_chs)  
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            ) 
        elif agg == 'simam': # simam注意力
            aggregation_conv = ConvBNAct(  
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab
            )
            att = SimAM()
            self.aggregation = nn.Sequential(   
                aggregation_conv,
                att,
            )   
        else:
            raise Exception(f"param agg{agg} Illegal.")
 
        # DropPath 处理（防止过拟合）
        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()  

    def forward(self, x):
        identity = x  # 保留输入用于残差连接
        output = [x]  # 存储各层输出     
        for layer in self.layers:    
            x = layer(x)
            output.append(x)  # 逐层添加到列表 
   
        # x1 = layers_1(x)    
        # x2 = layers_2(x1)   
        # x3 = layers_3(x2)
        # output = [x, x1, x2, x3]
     
        # 进行通道拼接  
        x = torch.cat(output, dim=1)    

        # 进行聚合处理
        x = self.aggregation(x)

        # 若使用残差连接，则进行残差加和     
        if self.residual:
            x = self.drop_path(x) + identity
        
        return x


class HG_Stage(nn.Module):
    """     
    HG_Stage 模块 (Hourglass Stage)     
    
    该模块由多个 HG_Block 组成，并可选进行下采样（downsample）。
    
    主要功能：
    1. 可选地对输入特征进行下采样，以减少空间尺寸，提高感受野。     
    2. 通过多个 HG_Block 进行特征提取，每个块逐层累积特征。   
    3. 适用于分层特征提取任务，如目标检测、分割和回归任务。 

    参数:   
    - in_chs (int): 输入通道数
    - mid_chs (int): 中间通道数（隐层维度）
    - out_chs (int): 输出通道数
    - block_num (int): 该 stage 内包含的 HG_Block 数量
    - layer_num (int): 每个 HG_Block 内的层数
    - downsample (bool): 是否在进入 Stage 之前进行下采样（默认 True）    
    - light_block (bool): 是否使用轻量级卷积块（影响 HG_Block 内部结构）
    - kernel_size (int): 卷积核大小（默认为 3）
    - use_lab (bool): 是否使用可学习的仿射变换模块    
    - agg (str): 特征聚合方式，可选 'se' 或 'ese' 或 'ema' 或 'simam'
    - drop_path (float or list): DropPath 比例（用于防止过拟合）

    """  

    def __init__(   
            self,
            in_chs,        # 输入通道数     
            mid_chs,       # 中间通道数（隐藏层）
            out_chs,       # 输出通道数    
            block_num,     # 该 stage 内包含的 HG_Block 数量
            layer_num,     # 每个 HG_Block 内的层数
            downsample=True, # 是否进行下采样   
            light_block=False, # 是否使用轻量级卷积块  
            kernel_size=3, # 卷积核大小
            use_lab=False, # 是否使用可学习的仿射变换模块
            agg='se',      # 特征聚合方式 ('se' 或 'ese' 或 'ema' 或 'simam') 
            drop_path=0.,  # DropPath 比例     
    ):     
        super().__init__()
        self.downsample = downsample     

        # 1. 下采样层（可选）  
        if downsample:     
            self.downsample = ConvBNAct(
                in_chs,   # 输入通道数   
                in_chs,   # 维持通道数不变
                kernel_size=3,
                stride=2,  # 采用 stride=2 进行下采样
                groups=in_chs,  # 分组卷积（深度卷积） 
                use_act=False,  # 关闭激活函数   
                use_lab=use_lab, # 是否使用可学习的仿射变换模块 
            )
        else:  
            self.downsample = nn.Identity()  # 如果不下采样，则直接返回输入   
  
        # 2. 构建多个 HG_Block
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(  
                HG_Block(    
                    in_chs if i == 0 else out_chs,  # 第一个 HG_Block 需要匹配输入通道
                    mid_chs,   # 中间层通道数
                    out_chs,   # 输出通道数
                    layer_num, # 该 HG_Block 内的层数
                    residual=False if i == 0 else True, # 只有第一个块不使用残差连接
                    kernel_size=kernel_size, # 卷积核大小
                    light_block=light_block, # 是否使用轻量级卷积块（深度可分离卷积）
                    use_lab=use_lab, # 是否使用可学习的仿射变换模块  
                    agg=agg, # 特征聚合方式 ('se' 或 'ese' 或 'ema' 或 'simam')  
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path, # DropPath 比例（用于正则化）
                )     
            )  
        
        # 3. 将所有 HG_Block 组成一个顺序执行的模块     
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):     
        """
        前向传播过程：   
        1. 先进行可选的下采样     
        2. 通过多个 HG_Block 进行特征提取
        """
        x = self.downsample(x)  # 进行下采样（如果启用）    
        x = self.blocks(x)  # 通过多个 HG_Block 进行特征提取 
        return x 


  
@register()   
class HGNetv2(nn.Module):
    """
    HGNetV2    
    Args: 
        stem_channels: list. Number of channels for the stem block.     
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.     
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages. 
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """   
 
    arch_configs = {
        'Atto': {      # only 3 stages
            'stem_channels': [3, 16, 16],     
            'stage_config': {     
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num  
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 256, 1, True, True, 3, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },    
        'Femto': {      # only 3 stages    
            'stem_channels': [3, 16, 16], 
            'stage_config': {    
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],   
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 1, True, True, 5, 3],
            }, 
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'Pico': {      # only 3 stages     
            'stem_channels': [3, 16, 16],
            'stage_config': {     
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3], 
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],    
            },    
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3], 
            },   
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth' 
        },
        'B1': {  
            'stem_channels': [3, 24, 32],   
            'stage_config': {  
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],  
                "stage3": [256, 96, 512, 2, True, True, 5, 3],     
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],    
            },     
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'    
        },     
        'B2': { 
            'stem_channels': [3, 24, 32],
            'stage_config': {     
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4], 
                "stage2": [96, 64, 384, 1, True, False, 3, 4],    
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],     
            },  
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'     
        },
        'B3': {
            'stem_channels': [3, 24, 32],    
            'stage_config': {     
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num    
                "stage1": [32, 32, 128, 1, False, False, 3, 5],    
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],  
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],    
            },   
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {    
            'stem_channels': [3, 32, 48],  
            'stage_config': { 
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num     
                "stage1": [48, 48, 128, 1, False, False, 3, 6],   
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],    
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],     
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'  
        },
        'B5': {     
            'stem_channels': [3, 32, 64], 
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num  
                "stage1": [64, 64, 128, 1, False, False, 3, 6], 
                "stage2": [128, 128, 512, 2, True, False, 3, 6],   
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],   
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],  
            }, 
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth' 
        },     
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],   
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],    
            }, 
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        }, 
    }    
    
    def __init__(self,
                 name,   
                 use_lab=False,
                 return_idx=[1, 2, 3],     
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True, 
                 agg='se',
                 local_model_dir='weight/hgnetv2/'):
        super().__init__() 
        self.use_lab = use_lab     
        self.return_idx = return_idx
     
        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']     
        download_url = self.arch_configs[name]['url']   
    
        self._out_strides = [4, 8, 16, 32]  
        self._out_channels = [stage_config[k][2] for k in stage_config]    
   
        # stem 
        self.stem = StemBlock(   
                in_chs=stem_channels[0],     
                mid_chs=stem_channels[1], 
                out_chs=stem_channels[2],
                use_lab=use_lab)   

        # stages  
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append( 
                HG_Stage(    
                    in_channels,    
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,  
                    light_block,  
                    kernel_size,
                    use_lab,
                    agg))     
    
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])    
     
        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:   
            try:
                if (name in ['Atto', 'Femto', 'Pico']):   
                    model_path = local_model_dir + 'PPHGNetV2_' + 'B0' + '_stage1.pth' 
                else:
                    model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    logger.info(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    # If the file doesn't exist locally, download from the URL    
                    if is_dist_available_and_initialized() and torch.distributed.get_rank() == 0: 
                        logger.warning(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        logger.warning(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)    
                        Multiprocess_sync()    
                    else:
                        Multiprocess_sync()
                        state = torch.load(local_model_dir) 
     
                    logger.info(f"Loaded stage1 {name} HGNetV2 from URL.")    

                # need_keep_key_prefix_list = ['stem.*', 'stages.0.*'] 
                # need_pop_key_list = []
                # for key in state.keys():  
                #     need_pop = True
                #     for keep in need_keep_key_prefix_list:     
                #         if re.match(keep, key):
                #             need_pop = False
                #             break
                #     if need_pop:     
                #         need_pop_key_list.append(key) 
                # for key in need_pop_key_list:
                #     state.pop(key)
                
                if name == 'Atto':  
                    self.load_partial_state_dict(self, state)    
                else:     
                    filtered_state = filter_loadable_state_dict(self.state_dict(), state) 
                    logger.info(RED + f'Loading Pretrained State Dict Key Names:{filtered_state.keys()}' + RESET)    
                    self.load_state_dict(filtered_state, strict=False)
     
            except (Exception, KeyboardInterrupt) as e:   
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or (not is_dist_available_and_initialized()):
                    print(f"{str(e)}")     
                    logger.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                    logger.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()

    @staticmethod
    def load_partial_state_dict(model, state_dict):   
        model_dict = model.state_dict()    
        filtered_dict = filter_loadable_state_dict(model_dict, state_dict)    
        logger.info(RED + f'Loading Pretrained State Dict Key Names:{filtered_dict.keys()}' + RESET)
        # 更新模型参数
        model_dict.update(filtered_dict)    
        model.load_state_dict(model_dict, strict=False)   

    def _freeze_norm(self, m: nn.Module):     
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features) 
        else:
            for name, child in m.named_children():  
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)  
        return m
    
    def _freeze_parameters(self, m: nn.Module):    
        for p in m.parameters():
            p.requires_grad = False 
    
    def forward(self, x):
        x = self.stem(x)   
        outs = []
        for idx, stage in enumerate(self.stages):  
            x = stage(x) 
            if idx in self.return_idx:  
                outs.append(x)    
        return outs 
