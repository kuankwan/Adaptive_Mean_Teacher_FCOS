import torch
import torch.nn as nn
import torch_pruning as tp

class MyMagnitudeImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        # 1. 首先定义一个列表用于存储分组内每一层的重要性
        group_imp = []
        # 2. 迭代分组内的各个层，对Conv层计算重要性
        for dep, idxs in group: # idxs是一个包含所有可剪枝索引的列表，用于处理DenseNet中的局部耦合的情况
            layer = dep.target.module # 获取 nn.Module
            prune_fn = dep.handler    # 获取 剪枝函数
            # 3. 这里我们简化问题，仅计算卷积输出通道的重要性
            if isinstance(layer, nn.Conv2d) and prune_fn == tp.prune_conv_out_channels:
                w = layer.weight.data[idxs].flatten(1) # 用索引列表获取耦合通道对应的参数，并展开成2维
                local_norm = w.abs().sum(1) # 计算每个通道参数子矩阵的 L1 Norm
                group_imp.append(local_norm) # 将其保存在列表中

        if len(group_imp)==0: return None # 跳过不包含卷积层的分组
        # 4. 按通道计算平均重要性
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        return group_imp