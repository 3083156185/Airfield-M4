import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_decay_strategies():
    # 模拟参数
    epoches = 5
    len_data_loader = 100
    total_iters = epoches * len_data_loader
    
    # 存储不同策略的衰减值
    strategies = {
        'constant': [],
        'cosine': [],
        'linear': [],
        'cosine_epoch': [],
        'linear_epoch': []
    }
    
    cur_iters = 0
    for epoch in range(epoches):
        for i in range(len_data_loader):
            global_iter = cur_iters + i
            
            # Constant
            strategies['constant'].append(1.0)
            
            # Cosine (单epoch)
            eta_min, base_ratio, T_max = 0.01, 1.0, 10
            cosine_val = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
            strategies['cosine'].append(cosine_val)
            
            # Linear (单epoch)
            linear_val = ((1 - math.cos(i * math.pi / len_data_loader)) / 2) * (0.01 - 1) + 1
            strategies['linear'].append(linear_val)
            
            # Cosine_epoch (全局)
            cosine_epoch_val = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * global_iter / T_max)) / 2
            strategies['cosine_epoch'].append(cosine_epoch_val)
            
            # Linear_epoch (全局)
            linear_epoch_val = ((1 - math.cos(global_iter * math.pi / total_iters)) / 2) * (0.01 - 1) + 1
            strategies['linear_epoch'].append(linear_epoch_val)
        
        cur_iters += len_data_loader
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    x = range(total_iters)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, values) in enumerate(strategies.items()):
        plt.subplot(2, 3, i+1)
        plt.plot(x, values, color=colors[i], linewidth=2)
        plt.title(f'{name.capitalize()} Decay Strategy')
        plt.xlabel('Iteration')
        plt.ylabel('Distill Decay Factor')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        # 添加epoch分割线
        for epoch in range(1, epoches):
            plt.axvline(x=epoch * len_data_loader, color='gray', linestyle='--', alpha=0.5)
    
    # 综合对比图
    plt.subplot(2, 3, 6)
    for i, (name, values) in enumerate(strategies.items()):
        plt.plot(x, values, color=colors[i], linewidth=2, label=name, alpha=0.8)
    
    plt.title('All Strategies Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Distill Decay Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # 添加epoch分割线
    for epoch in range(1, epoches):
        plt.axvline(x=epoch * len_data_loader, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distill_decay.png')

# 调用可视化函数
visualize_decay_strategies()