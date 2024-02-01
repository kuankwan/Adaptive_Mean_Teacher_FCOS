import math

import torch
from torch.nn import Parameter, Module
import torch.nn.functional as F


class MaskedConv(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 将weight转换为可学习的变量
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # 初始化mask的值为1，并转换为可学习的变量
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            # 对bias进行初始化
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            # 将bias设置为空
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 参数初始化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向传播（实际上也是使用标准的Liner层）
    def forward(self, input):
        # 其中的weight、mask都定义成可变的可学习变量
        return F.conv2d(input, self.weight * self.mask, self.bias)


class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.in_features = in_channels
        self.out_features = out_channels
        # 初始化mask的值为1，并转换为可学习的变量
        self.mask = Parameter(torch.ones([self.out_features, self.in_features]), requires_grad=False)
        self.reset_parameters()

    # 参数初始化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向传播（实际上也是使用标准的Liner层）
    def forward(self, input):
        # 其中的weight、mask都定义成可变的可学习变量
        self.weight.data *= self.mask  # mask weights
        return super().forward(input)
