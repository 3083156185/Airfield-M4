"""     
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)    
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""   

import torch
import torch.nn as nn  
import torch.optim as optim
from torch.utils.data import DataLoader  

import re
import copy     

from ._config import BaseConfig
from .workspace import create 
from .yaml_utils import load_config, merge_config, merge_dict
from ..logger_module import get_logger

logger = get_logger(__name__) 
     
from collections import defaultdict
def debug_optimizer_params(param_groups):     
    """
    检查优化器参数组中的重复参数     
    
    Args:
        param_groups: 传递给优化器的参数组列表 
    """   
    print("=== 优化器参数重复检查 ===")    
    
    # 用于记录每个参数出现在哪些组中
    param_to_groups = defaultdict(list)
  
    # 遍历所有参数组  
    for group_idx, group in enumerate(param_groups):  
        print(f"\n参数组 {group_idx}:")   
        
        # 获取参数列表    
        if isinstance(group, dict) and 'params' in group:
            params = group['params']
            print(f"  - 参数数量: {len(params)}")  
            print(f"  - 其他配置: {dict((k, v) for k, v in group.items() if k != 'params')}")
        else: 
            params = group
            print(f"  - 参数数量: {len(params)}")   
        
        # 检查每个参数    
        for param_idx, param in enumerate(params):   
            param_id = id(param)
            param_to_groups[param_id].append((group_idx, param_idx))   
 
            # 打印参数信息     
            if hasattr(param, 'shape'): 
                print(f"    参数 {param_idx}: shape={param.shape}, id={param_id}")     
            else:
                print(f"    参数 {param_idx}: type={type(param)}, id={param_id}")
     
    # 检查重复的参数    
    print("\n=== 重复参数检查结果 ===")
    duplicates_found = False     
    
    for param_id, group_locations in param_to_groups.items():     
        if len(group_locations) > 1:
            duplicates_found = True
            print(f"\n❌ 发现重复参数 (id: {param_id}):")
            print(f"   出现在以下位置: {group_locations}")    
            
            # 尝试获取参数的更多信息     
            for group_idx, param_idx in group_locations:
                if isinstance(param_groups[group_idx], dict):    
                    param = param_groups[group_idx]['params'][param_idx]   
                else:
                    param = param_groups[group_idx][param_idx]
   
                if hasattr(param, 'shape'):
                    print(f"   - 组{group_idx}[{param_idx}]: shape={param.shape}")     
                if hasattr(param, '_get_name'):
                    print(f"   - 组{group_idx}[{param_idx}]: name={param._get_name()}")    
   
    if not duplicates_found:
        print("✅ 未发现重复参数") 
   
    return duplicates_found   

def check_model_params_with_names(model):
    """
    检查模型参数的分组情况，并返回参数ID到名称的映射
    """
    print("=== 模型参数分析 ===")
    
    all_params = list(model.parameters())  
    named_params = dict(model.named_parameters())
    
    print(f"模型总参数数量: {len(all_params)}")
    print(f"命名参数数量: {len(named_params)}")
    
    # 检查参数名称和对应的参数对象
    param_id_to_name = {}
    for name, param in named_params.items():   
        param_id_to_name[id(param)] = name
    
    print("\n参数名称映射:")
    for param_id, name in param_id_to_name.items():
        print(f"  {name}: id={param_id}")
  
    return param_id_to_name     
     
# 在你的代码中，在创建优化器之前添加这段调试代码：
def debug_before_optimizer_creation(model, param_groups_or_params):
    """     
    在创建优化器前进行调试 
    """
    print("=" * 50)   
    print("开始优化器参数调试")     
    print("=" * 50) 
    
    # 分析模型参数  
    param_id_to_name = check_model_params_with_names(model)    
    
    # 检查传递给优化器的参数
    if isinstance(param_groups_or_params, (list, tuple)):
        print(f"\n传递给优化器的参数组数量: {len(param_groups_or_params)}")
        debug_optimizer_params(param_groups_or_params)    
    else:
        print("\n传递给优化器的是单个参数列表")    
        debug_optimizer_params([param_groups_or_params])
    
    print("=" * 50)     
    print("调试完成")   
    print("=" * 50)
     
class YAMLConfig(BaseConfig):    
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__() 
  
        cfg = load_config(cfg_path)  
        cfg = merge_dict(cfg, kwargs)  
        if cfg.get('use_amp_fallback_fp32', False):  
            cfg['use_amp'] = True    

        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:    
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]
    
    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property 
    def model(self, ) -> torch.nn.Module:  
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)   
        return super().model   
     
    @property
    def postprocessor(self, ) -> torch.nn.Module:     
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:    
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    @property    
    def criterion(self, ) -> torch.nn.Module:   
        if self._criterion is None and 'criterion' in self.yaml_cfg:   
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)    
        return super().criterion

    @property
    def optimizer(self, ) -> optim.Optimizer: 
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:    
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            # debug_before_optimizer_creation(self.model, params)  
            self._optimizer = create('optimizer', self.global_cfg, params=params) 
        return super().optimizer     

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
            logger.info(f'Initial lr: {self._lr_scheduler.get_last_lr()}')
        return super().lr_scheduler

    @property 
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler: 
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :
            self._lr_warmup_scheduler = create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler
    
    @property  
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return super().train_dataloader

    @property 
    def val_dataloader(self, ) -> DataLoader:    
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:    
            self._val_dataloader = self.build_dataloader('val_dataloader')     
        return super().val_dataloader  

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):  
            self._ema = create('ema', self.global_cfg, model=self.model)
        return super().ema   
  
    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False): 
            self._scaler = create('scaler', self.global_cfg)
        return super().scaler

    @property   
    def evaluator(self, ):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from ..data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset) 
                self._evaluator = create('evaluator', self.global_cfg, coco_gt=base_ds)  
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return super().evaluator
    
    @staticmethod  
    def get_optim_params(cfg: dict, model: nn.Module):
        """    
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """  
        assert 'type' in cfg, '' 
        cfg = copy.deepcopy(cfg)  
   
        if 'params' not in cfg:  
            return model.parameters()   

        assert isinstance(cfg['params'], list), '' 

        param_groups = []
        visited = []
        for pg in cfg['params']:    
            pattern = pg['params']  
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}   
            pg['params'] = params.values()
            param_groups.append(pg)     
            visited.extend(list(params.keys()))
            # print(pattern, params.keys())     
  
        names = [k for k, v in model.named_parameters() if v.requires_grad]
 
        if len(visited) < len(names):  
            unseen = set(names) - set(visited)  
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})     
            visited.extend(list(params.keys()))     
            # print(params.keys())   
        
        # from collections import Counter
        # counter = Counter(visited)
        # duplicates = [item for item, count in counter.items() if count > 1]
        # print(duplicates)

        # assert len(visited) == len(names), ''    

        return param_groups  
    
    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided. 
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'     
  
        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else: 
            from ..misc import dist_utils    
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs     
  
    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])    
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        logger.info(f'building {name} with batch_size={bs}...')     
        loader = create(name, global_cfg, batch_size=bs)     
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)
        return loader 
