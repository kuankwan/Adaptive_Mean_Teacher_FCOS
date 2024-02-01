import torch.nn as nn
import torch
from math import sqrt
class SelfAttention(nn.Module):
    def __init__(self, in_channels, key_channels,value_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        # 定义线性变换函数
        self.keys = nn.Conv2d(in_channels, key_channels, 1,bias=False)
        self.queries = nn.Conv2d(in_channels, key_channels, 1,bias=False)
        self.values = nn.Conv2d(in_channels, value_channels, 1,bias=False)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        self._norm_fact = 1 / sqrt(self.key_channels)

    def forward(self, x):
        # x: batch, n, dim_q
        # 根据文本获得相应的维度

        n, _, h, w = x.size()

        q = self.keys(x).reshape((n, self.key_channels, h * w))  # batch, n, dim_k
        k = self.queries(x).reshape((n, self.key_channels, h * w))  # batch, n, dim_k
        v = self.values(x).reshape((n, self.value_channels, h * w))  # batch, n, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(k.transpose(1, 2),q) * self._norm_fact  # batch, n, n
        Amap,_ = torch.max(dist,1)
        Amap = Amap.reshape(n,1,h,w)
        A_min, A_max = torch.min(Amap), torch.max(Amap)
        Amap_ = (Amap - A_min) / (A_max - A_min)
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(v,dist)
        att = att.reshape(n, self.value_channels, h, w)
        att = self.reprojection(att)
        att = att + x
        return Amap_,att
