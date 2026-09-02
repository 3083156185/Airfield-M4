
import torch

if __name__ == '__main__':
    weight = torch.load('weight/hgnetv2/PPHGNetV2_B0_stage1.pth')
    old = weight.keys()

    new_weight = {}
    for keys in weight.keys():
        old_keys = keys
        if keys.startswith('stem.'):
            keys = keys.replace('stem.', '0.')
        elif keys.startswith('stages.0'):
            keys = keys.replace('stages.0', '1')
        elif keys.startswith('stages.1'):
            keys = keys.replace('stages.1', '2')
        elif keys.startswith('stages.2'):
            keys = keys.replace('stages.2', '3')
        elif keys.startswith('stages.3'):
            keys = keys.replace('stages.3', '4')
        
        new_weight[keys] = weight[old_keys]

    torch.save(new_weight, 'weight/hgnetv2/PPHGNetV2_B0_stage1_MG.pth')