import math

import torch
import torch.nn as nn

from ..basic.conv import Conv
from models.basic.conv import get_activation,get_norm
from models.neck.extra_module import (Conv2dStaticSamePadding, MaxPool2dStaticSamePadding,
                         MemoryEfficientSwish, Swish)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        #ratio一般会指定成2，保证输出特征层的通道数等于exp
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        #利用1x1卷积对输入进来的特征图进行通道的浓缩，获得特征通缩
        #跨通道的特征提取
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),  #1x1卷积的输入通道数为GhostModule的输出通道数oup/2
            nn.BatchNorm2d(init_channels),                       #1x1卷积后进行标准化
            nn.ReLU(inplace=True) if relu else nn.Sequential(),  #ReLU激活函数
        )

        #在获得特征浓缩后，使用逐层卷积，获得额外的特征图
        #跨特征点的特征提取    一般会设定大于1的卷积核大小
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),  #groups参数的功能就是将普通卷积转换成逐层卷据
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        #将1x1卷积后的结果和逐层卷积后的结果进行堆叠
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None,conv_kernels=[1, 3, 5, 7], norm='', activation='relu', onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv1 = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=conv_kernels[0], stride=1,
                                                      groups=in_channels, bias=False)
        self.depthwise_conv3 = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=conv_kernels[1], stride=1,padding=conv_kernels[1]//2,
                                                      groups=in_channels, bias=False)
        self.depthwise_conv5 = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=conv_kernels[2], stride=1,padding=conv_kernels[2]//2,
                                                       groups=in_channels, bias=False)
        self.depthwise_conv7 = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=conv_kernels[3], stride=1,
                                                       padding=conv_kernels[3] // 2,
                                                       groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels*4, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation
        if self.activation:
            # self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
            self.swish = nn.ReLU()

    def forward(self, x):
        x1 = self.depthwise_conv1(x)
        x2 = self.depthwise_conv3(x)
        x3 = self.depthwise_conv5(x)
        x4 = self.depthwise_conv7(x)
        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = self.pointwise_conv(x)

        if self.norm != '':
            x = self.bn(x)

        if self.activation != '':
            x = self.swish(x)

        return x


class RTEHead(nn.Module):
    def __init__(self,
                 head_dim=64,
                 num_cls_head=4,
                 num_reg_head=4,
                 act_type='relu',
                 norm_type=''):
        super().__init__()

        print('==============================')
        print('Head: Decoupled RTE Head')

        # self.cls_feats = nn.Sequential(*[SeparableConvBlock(head_dim,
        #                                       head_dim,norm=norm_type,activation=act_type) for _ in range(num_cls_head)])
        # self.reg_feats = nn.Sequential(*[SeparableConvBlock(head_dim,
        #                                       head_dim,norm=norm_type,activation=act_type) for _ in range(num_reg_head)])
        self.cls_feats = nn.Sequential(*[Conv(head_dim,
                                              head_dim,
                                              k=3, p=1, s=1,
                                              act_type=act_type,
                                              norm_type=norm_type) for _ in range(num_cls_head)])
        self.reg_feats = nn.Sequential(*[Conv(head_dim,
                                              head_dim,
                                              k=3, p=1, s=1,
                                              act_type=act_type,
                                              norm_type=norm_type) for _ in range(num_reg_head)])

        self._init_weight()


    def _init_weight(self):
        # init weight of detection head
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats
