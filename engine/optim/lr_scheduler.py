"""   
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.  
"""    

import math   
from functools import partial
import matplotlib.pyplot as plt
from ..extre_module.utils import plt_settings, TryExcept
from ..logger_module import get_logger
  
RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, LAVENDER, GOLD, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[96m", "\033[95m", "\033[38;5;147m", "\033[38;5;220m", "\033[0m"
logger = get_logger(__name__)   

   
def _paint(text, color):
    return f"{color}{text}{RESET}" 

   
def _format_lr(value):  
    return f"{value:.6e}"

     
def _get_param_group_name(group, index):     
    for key in ("name", "group_name", "tag", "role", "module", "label"):
        value = group.get(key)
        if value:
            return str(value) 
    return f"unnamed"     
 

def _describe_param_group(group, index, base_lr, min_lr): 
    parts = [
        _paint(f"group {index}", CYAN) + f" ({_paint(_get_param_group_name(group, index), MAGENTA)})",    
        _paint("initial_lr", GREEN) + f"={_format_lr(base_lr)}",
        _paint("min_lr", YELLOW) + f"={_format_lr(min_lr)}",
    ]    
    if "weight_decay" in group:
        parts.append(_paint("weight_decay", ORANGE) + f"={_format_lr(group['weight_decay'])}")
    if "momentum" in group: 
        parts.append(_paint("momentum", BLUE) + f"={_format_lr(group['momentum'])}")    
    if "dampening" in group:
        parts.append(_paint("dampening", BLUE) + f"={_format_lr(group['dampening'])}")   
    if "nesterov" in group:
        parts.append(_paint("nesterov", BLUE) + f"={group['nesterov']}")
    if "betas" in group:
        betas = group["betas"]
        parts.append(_paint("betas", LAVENDER) + f"=({', '.join(_format_lr(beta) for beta in betas)})")    
    parts.append(_paint("params", GOLD) + f"={sum(p.numel() for p in group['params'])}")
    return " | ".join(parts)
     

def _collect_schedule_checkpoints(total_iter, warmup_iter, flat_iter, no_aug_iter): 
    checkpoints = [    
        (0, "start"),
        (warmup_iter, "warmup_end"),
        (flat_iter, "flat_end"),
        (max(total_iter - no_aug_iter, 0), "no_aug_start"),
        (max(total_iter - 1, 0), "final"), 
    ]   
    merged = {}    
    for step, label in checkpoints:  
        step = max(0, min(step, max(total_iter - 1, 0)))
        merged.setdefault(step, []).append(label) 
    return [(step, labels) for step, labels in merged.items()]
     

def _format_phase_ranges(total_iter, warmup_iter, flat_iter, no_aug_iter):  
    no_aug_start = max(total_iter - no_aug_iter, 0)
    return [    
        _paint("warmup", GREEN) + f"   : iter <= {warmup_iter}",
        _paint("flat", CYAN) + f"     : {warmup_iter} < iter <= {flat_iter}",  
        _paint("cosine", MAGENTA) + f"   : {flat_iter} < iter < {no_aug_start}",
        _paint("no_aug", YELLOW) + f"   : iter >= {no_aug_start}",    
    ]  
     
def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):    
    """    
    Computes the learning rate using a warm-up, flat, and cosine decay schedule.
    计算基于 warm-up、flat 以及 cosine 衰减的学习率。
    Args: 
        total_iter (int): Total number of iterations. 总迭代次数。
        warmup_iter (int): Number of iterations for warm-up phase. 预热阶段的迭代次数。
        flat_iter (int): Number of iterations for flat phase. 平坦阶段的迭代次数（warm-up 之后，cosine 衰减之前）。   
        no_aug_iter (int): Number of iterations for no-augmentation phase. 无增强阶段的迭代次数（最后的学习率固定为 min_lr）。
        current_iter (int): Current iteration. 当前迭代次数。   
        init_lr (float): Initial learning rate. 初始学习率。   
        min_lr (float): Minimum learning rate. 最小学习率。

    Returns:    
        float: Calculated learning rate.
    """
    # **1. 预热阶段（warm-up）**：使用平方增长策略，使学习率逐渐增加到 init_lr
    if warmup_iter > 0 and current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2    
    # **2. 平坦阶段（flat）**：保持学习率恒定为 init_lr  
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr     
    # **3. 无增强阶段（no-augmentation）**：保持学习率恒定为 min_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr    
    # **4. 余弦衰减阶段（cosine decay）**：
    else:
        # 计算余弦衰减因子    
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iter) /
                                           (total_iter - flat_iter - no_aug_iter)))   
        # 计算余弦衰减因子
        return min_lr + (init_lr - min_lr) * cosine_decay
  
     
class FlatCosineLRScheduler:
    """    
    Learning rate scheduler with warm-up, optional flat phase, and cosine decay following RTMDet.   
    具有 warm-up、flat 和 cosine 衰减的学习率调度器，类似于 RTMDet。

    Args: 
        optimizer (torch.optim.Optimizer): Optimizer instance. PyTorch 优化器实例。     
        lr_gamma (float): Scaling factor for the minimum learning rate. 最小学习率相对于初始学习率的缩放因子。
        iter_per_epoch (int): Number of iterations per epoch. 每个 epoch 的迭代次数（batch 数量）。 
        total_epochs (int): Total number of training epochs. 训练的总 epoch 数。  
        warmup_epochs (int): Number of warm-up epochs. 预热阶段的迭代次数。
        flat_epochs (int): Number of flat epochs (for flat-cosine scheduler). 平坦阶段的 epoch 数（平稳学习率）。
        no_aug_epochs (int): Number of no-augmentation epochs. 无增强阶段的 epoch 数（学习率锁定为 min_lr）。     
        scheduler_type (str): 学习率调度类型（默认为 "cosine"）。
    """ 
     
    '''   
    学习率变化过程
     
    假设：     
        •	init_lr = 0.01   
        •	min_lr = 0.0001  
        •	total_iter = 10000  
        •	warmup_iter = 1000
        •	flat_iter = 3000  
        •	no_aug_iter = 500   
     
    则：
        1.	[0 - 1000] 预热阶段：学习率从 0 增长到 0.01（二次方增长）。   
        2.	[1000 - 3000] 平坦阶段：学习率保持 0.01。     
        3.	[3000 - 9500] 余弦衰减阶段：学习率从 0.01 逐渐降低到 0.0001。
        4.	[9500 - 10000] 无增强阶段：学习率保持 0.0001。
    '''     

    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs, 
                 warmup_iter, flat_epochs, no_aug_epochs, scheduler_type="cosine", lr_scyedule_save_path=None):
        # **1. 计算基础学习率（initial_lr）和最小学习率（min_lr）**
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]  # 获取优化器中每组参数的初始学习率
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]  # 计算最小学习率（init_lr * lr_gamma）
        self.external_lr_scale = 1.0     

        # **2. 计算不同阶段的迭代次数**   
        total_iter = int(iter_per_epoch * total_epochs)  # 总训练迭代次数 = 迭代数/epoch * 总 epoch 数
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)  # 无增强阶段的迭代次数    
        flat_iter = int(iter_per_epoch * flat_epochs)  # 平坦阶段的迭代次数
        if flat_iter > total_iter:  
            flat_iter = total_iter - warmup_iter

        # **3. 打印关键超参数信息**
        logger.info(_paint("========== LR Scheduler ==========", GOLD))     
        logger.info(_paint("[Scheduler Overview]", GOLD)) 
        logger.info(  
            f"{_paint('type', CYAN)}: flatcosine | " 
            f"{_paint('groups', CYAN)}: {len(optimizer.param_groups)} | "
            f"{_paint('iter_per_epoch', CYAN)}: {iter_per_epoch} | " 
            f"{_paint('total_epochs', CYAN)}: {total_epochs}"    
        )     
        logger.info(
            f"{_paint('total_iter', GREEN)}: {total_iter} | "
            f"{_paint('warmup_iter', GREEN)}: {warmup_iter} | "   
            f"{_paint('flat_iter', GREEN)}: {flat_iter} | "
            f"{_paint('no_aug_iter', GREEN)}: {no_aug_iter}"
        ) 
        logger.info(_paint("[Phase Ranges]", GOLD))    
        for line in _format_phase_ranges(total_iter, warmup_iter, flat_iter, no_aug_iter):
            logger.info(f"  {line}")   
        logger.info(_paint("[Param Group Details]", GOLD))
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"  {_describe_param_group(group, i, self.base_lrs[i], self.min_lrs[i])}") 
        logger.info(_paint("[Phase Preview]", GOLD))
        for step, labels in _collect_schedule_checkpoints(total_iter, warmup_iter, flat_iter, no_aug_iter):
            values = [   
                flat_cosine_schedule(
                    total_iter,    
                    warmup_iter,
                    flat_iter,   
                    no_aug_iter,    
                    step,
                    self.base_lrs[i], 
                    self.min_lrs[i],  
                )
                for i, _ in enumerate(optimizer.param_groups) 
            ]
            label_text = ", ".join(_paint(label, LAVENDER) for label in labels)
            lr_text = ", ".join(_format_lr(v) for v in values) 
            logger.info(f"  {_paint(f'iter {step}', BLUE)}: ({label_text}) [{lr_text}]")   
        logger.info(_paint("==================================", GOLD)) 
  
        # **4. 绑定 `flat_cosine_schedule` 计算函数**  
        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter, flat_iter, no_aug_iter)    
     
        for i, _ in enumerate(optimizer.param_groups):
            plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, self.base_lrs[i], self.min_lrs[i], lr_scyedule_save_path / f"lr_schedule_{i}.png")
 
    def set_external_lr_scale(self, scale):    
        scale = float(scale)    
        if not math.isfinite(scale) or scale < 0:     
            raise ValueError("external lr scale must be finite and >= 0")
        self.external_lr_scale = scale   

    def get_external_lr_scale(self):
        return self.external_lr_scale     
   
    def step(self, current_iter, optimizer):
        """   
        Updates the learning rate of the optimizer at the current iteration.

        Args:    
            current_iter (int): Current iteration.    
            optimizer (torch.optim.Optimizer): Optimizer instance. 
        """
        # 遍历优化器中的参数组，更新学习率
        for i, group in enumerate(optimizer.param_groups):
            scheduled_lr = self.lr_func(current_iter, self.base_lrs[i], self.min_lrs[i])
            group["lr"] = scheduled_lr * self.external_lr_scale # 计算并设置新学习率
        return optimizer # 返回更新后的优化器 

  
class MetricLrReducer:
    def __init__(
        self,
        threshold=0.001,
        patience=1,
        factor=0.5,
        min_scale=0.0,
        cooldown=0,
        start_epoch=0,    
        enabled=True, 
    ):     
        threshold = float(threshold)
        factor = float(factor)   
        min_scale = float(min_scale)   
        patience = int(patience)     
        cooldown = int(cooldown)  
        start_epoch = int(start_epoch)
        enabled = bool(enabled)
  
        if patience < 1:     
            raise ValueError("patience must be >= 1")
        if not math.isfinite(threshold) or threshold < 0:     
            raise ValueError("threshold must be finite and >= 0")
        if not 0 < factor < 1:  
            raise ValueError("factor must be in (0, 1)")
        if min_scale < 0 or min_scale > 1:    
            raise ValueError("min_scale must be in [0, 1]")
        if cooldown < 0: 
            raise ValueError("cooldown must be >= 0")
        if start_epoch < 0:     
            raise ValueError("start_epoch must be >= 0") 
 
        self.threshold = threshold
        self.patience = patience 
        self.factor = factor 
        self.min_scale = min_scale 
        self.cooldown = cooldown
        self.start_epoch = start_epoch    
        self.enabled = enabled 
  
        self.best_metric = None   
        self.bad_epochs = 0
        self.cooldown_counter = 0  
        self.lr_scale = 1.0

    def step(self, metric, epoch):
        metric = float(metric) 
        if not math.isfinite(metric):    
            raise ValueError("metric must be finite") 
        epoch = int(epoch)
        triggered = False
        old_lr_scale = self.lr_scale
        improvement = 0.0   
        cooldown_before = self.cooldown_counter   
        in_cooldown = cooldown_before > 0
   
        result = {   
            "triggered": triggered,    
            "metric": metric,
            "best_metric": self.best_metric,  
            "improvement": improvement,
            "bad_epochs": self.bad_epochs,     
            "lr_scale": self.lr_scale,     
            "cooldown_counter": self.cooldown_counter,
        }
        if (not self.enabled) or epoch < self.start_epoch:  
            return result 
  
        if in_cooldown:     
            self.cooldown_counter = cooldown_before - 1     

        if self.best_metric is None:
            self.best_metric = metric    
            self.bad_epochs = 0   
        else:
            improvement = metric - self.best_metric
            if improvement >= self.threshold:
                self.best_metric = metric
                self.bad_epochs = 0
            elif not in_cooldown:   
                self.bad_epochs += 1     
                if self.bad_epochs >= self.patience:
                    self.bad_epochs = 0     
                    self.cooldown_counter = self.cooldown  
                    self.lr_scale = max(self.lr_scale * self.factor, self.min_scale)    
                    triggered = self.lr_scale < old_lr_scale  
     
        result = {   
            "triggered": triggered,
            "metric": metric,   
            "best_metric": self.best_metric,
            "improvement": improvement,
            "bad_epochs": self.bad_epochs,
            "lr_scale": self.lr_scale,
            "cooldown_counter": self.cooldown_counter,  
        }
        if triggered:    
            result["old_lr_scale"] = old_lr_scale   
        return result 
    
    def state_dict(self):
        return {
            "best_metric": self.best_metric,     
            "bad_epochs": self.bad_epochs, 
            "cooldown_counter": self.cooldown_counter,
            "lr_scale": self.lr_scale,  
        }    
 
    def load_state_dict(self, state_dict):   
        self.best_metric = state_dict.get("best_metric", self.best_metric)     
        self.bad_epochs = state_dict.get("bad_epochs", self.bad_epochs)
        self.cooldown_counter = state_dict.get("cooldown_counter", self.cooldown_counter)
        self.lr_scale = state_dict.get("lr_scale", self.lr_scale)     

@TryExcept('WARNING ⚠️ plot_lr_schedule failed.')     
@plt_settings()  
def plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, init_lr, min_lr, save_path):
    is_four_stage = True
    iters = list(range(total_iter))
    if flat_iter == (total_iter - warmup_iter):
        is_four_stage = False
    lrs = [flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, i, init_lr, min_lr) for i in iters]
    
    plt.figure(figsize=(8, 5))
    plt.plot(iters, lrs, label='Learning Rate') 
    plt.axvline(x=warmup_iter, color='r', linestyle='--', label='Warmup End')
    if is_four_stage: 
        plt.axvline(x=flat_iter, color='g', linestyle='--', label='Flat End')
        plt.axvline(x=total_iter - no_aug_iter, color='b', linestyle='--', label='No Aug Start')
    else:    
        plt.axvline(x=flat_iter + warmup_iter, color='g', linestyle='--', label='Flat End')
    plt.xlabel('Iterations')     
    plt.ylabel('Learning Rate')
    plt.title('Flat Cosine Learning Rate Schedule') 
    plt.legend() 
    plt.grid()
    plt.tight_layout() 
    plt.savefig(save_path, dpi=300)
    plt.close('all')
    
if __name__ == '__main__':
    # 参数设置
    total_iter = 10000    
    warmup_iter = 1000
    flat_iter = 3000
    no_aug_iter = 500  
    init_lr = 0.01
    min_lr = 0.0001    
  
    # 绘制学习率曲线 
    plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, init_lr, min_lr)     
