import torch
import torch.nn as nn
from models.SelfAttention import SelfAttention
from ..basic.conv import Conv
from ..maskconv import MaskedConv2d,MaskedConv


class MaskConv(nn.Module):
    def __init__(self,
                 c1,  # in channels
                 c2,  # out channels
                 k=1,  # kernel size
                 norm_type='',  # normalization
                 padding_mode='ZERO',  # padding mode: "ZERO" or "SAME"
                 depthwise=False):
        super(MaskConv, self).__init__()
        convs = []
        convs.append(MaskedConv(c1,c2))
        convs.append(torch.nn.ReLU())
        add_bias = False if norm_type else True
        if add_bias:
            convs.append(torch.nn.BatchNorm2d(c2))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)

class WeakHead(nn.Module):
    def __init__(self,
                 head_dim=256,
                 num_cls_head=4,
                 num_reg_head=4,
                 act_type='relu',
                 norm_type=''):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')

        self.cls_conv1 = Conv(head_dim, head_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        self.cls_conv2 = Conv(head_dim, head_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        self.cls_conv3 = Conv(head_dim, head_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        self.cls_conv4 = Conv(head_dim, head_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        self.reg_conv1 = Conv(head_dim, head_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        self.reg_conv2 = Conv(head_dim, head_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        self.reg_conv3 = Conv(head_dim, head_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        self.reg_conv4 = Conv(head_dim, head_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)

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
        cls_feats=[]
        reg_feats=[]
        cls_feat1 = self.cls_conv1(x)
        cls_feats.append(cls_feat1)
        cls_feat2 = self.cls_conv2(cls_feat1)
        cls_feats.append(cls_feat2)
        cls_feat3 = self.cls_conv3(cls_feat2)
        cls_feats.append(cls_feat3)
        cls_feat4 = self.cls_conv4(cls_feat3)
        cls_feats.append(cls_feat4)
        reg_feat1 = self.reg_conv1(x)
        reg_feats.append(reg_feat1)
        reg_feat2 = self.reg_conv2(reg_feat1)
        reg_feats.append(reg_feat2)
        reg_feat3 = self.reg_conv3(reg_feat2)
        reg_feats.append(reg_feat3)
        reg_feat4 = self.reg_conv4(reg_feat3)
        reg_feats.append(reg_feat4)
        # cls_feats = self.cls_feats(x)
        # reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats
