from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import numpy as np
# from .SE_weight_module import SEWeightModule
import torch.nn as nn
from torch.autograd import Function
from models.da.Drop import Drop

class ImageLabelResizeLayer(nn.Module):
    """
    Resize label to be the same size with the samples
    """
    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()


    def forward(self,x,need_backprop):

        feats = x.detach().cpu().numpy()
        lbs = need_backprop.detach().cpu().numpy()
        gt_blob = np.zeros((lbs.shape[0], feats.shape[2], feats.shape[3], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            lb=np.array([lbs[i]])
            lbs_resize = cv2.resize(lb, (feats.shape[3] ,feats.shape[2]),  interpolation=cv2.INTER_NEAREST)
            gt_blob[i, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize

        channel_swap = (0, 3, 1, 2)
        gt_blob = gt_blob.transpose(channel_swap)
        y=Variable(torch.from_numpy(gt_blob)).cuda()
        # y=y.squeeze(1).long()
        return y


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# class PSAModule(nn.Module):
#
#     def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
#         super(PSAModule, self).__init__()
#         self.conv_1 = nn.Conv2d(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
#                             stride=stride, groups=conv_groups[0])
#         self.conv_2 = nn.Conv2d(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
#                             stride=stride, groups=conv_groups[1])
#         self.conv_3 = nn.Conv2d(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
#                             stride=stride, groups=conv_groups[2])
#         self.conv_4 = nn.Conv2d(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
#                             stride=stride, groups=conv_groups[3])
#         self.se = SEWeightModule(planes // 4)
#         self.t_se = SEWeightModule(planes // 4)
#         self.split_channel = planes // 4
#         self.softmax = nn.Softmax(dim=1)
#         self._init_weight()
#
#     def _init_weight(self):
#         # init weight of detection head
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, mean=0, std=0.01)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#             if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#
#         batch_size = x[0].shape[0]
#         x1 = self.conv_1(x[0])
#         x2 = self.conv_2(x[1])
#         x3 = self.conv_3(x[2])
#         x4 = self.conv_4(x[3])
#
#         feats = torch.cat((x1, x2, x3, x4), dim=1)
#         feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
#         x1_se = self.se(x1)
#         x2_se = self.se(x2)
#         x3_se = self.se(x3)
#         x4_se = self.se(x4)
#
#         x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
#         attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
#         attention_vectors = self.softmax(attention_vectors)
#         feats_weight = feats * attention_vectors
#         for i in range(4):
#             x_se_weight_fp = feats_weight[:, i, :, :]
#             if i == 0:
#                 out = x_se_weight_fp
#             else:
#                 out = torch.cat((x_se_weight_fp, out), 1)
#
#         return out

class FeatureConcat(nn.Module):
    def __init__(self,layers):
        super(FeatureConcat,self).__init__()
        self.layers = layers
        self.lengths = len(layers)
    def forward(self,x,outputs):
        return torch.cat([outputs[i] for i in self.layers],1) if self.lengths else outputs[self.layers[0]]

class D_layer2(nn.Module):
    def __init__(self,in_channel:int,grad_reverse_lambda=0.1):
        super(D_layer2,self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.classier = nn.Conv2d(self.in_channel, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        # self.relu = nn.SiLU(inplace=True)
        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.LabelResizeLayer = ImageLabelResizeLayer()
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classier.bias, bias_value)

    def forward(self, x,need_backprop):
        x = self.grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.classier(x)
        if need_backprop is not None:
            label=self.LabelResizeLayer(x,need_backprop)
        else:
            label=None
        return x,label



class D_layer3(nn.Module):
    def __init__(self,in_channel:int,grad_reverse_lambda=0.1):
        super(D_layer3,self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.classier = nn.Conv2d(self.in_channel, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        # self.relu = nn.SiLU(inplace=True)
        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.LabelResizeLayer = ImageLabelResizeLayer()
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classier.bias, bias_value)

    def forward(self, x,need_backprop):
        x = self.grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.classier(x)
        if need_backprop is not None:
            label=self.LabelResizeLayer(x,need_backprop)
        else:
            label=None
        return x,label


class D_layer4(nn.Module):
    def __init__(self,in_channel:int,grad_reverse_lambda=0.1):
        super(D_layer4,self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.classier = nn.Conv2d(self.in_channel, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        # self.relu = nn.SiLU(inplace=True)
        self._init_weights()
        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.LabelResizeLayer = ImageLabelResizeLayer()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classier.bias, bias_value)

    def forward(self, x,need_backprop):
        x = self.grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.classier(x)
        if need_backprop is not None:
            label=self.LabelResizeLayer(x,need_backprop)
        else:
            label=None
        return x,label


class _InstanceDA(nn.Module):
    def __init__(self,grad_reverse_lambda=1.0,):
        super(_InstanceDA,self).__init__()
        self.conv1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.classifer = nn.Conv2d(256,1,kernel_size=3,padding=1)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classifer.bias, bias_value)


    def forward(self,x):
        ins_feat = self.grad_reverse(x)
        mask = self.relu(self.conv1(ins_feat))
        mask = self.relu(self.conv2(mask))
        mask = self.classifer(mask)
        # mask = F.avg_pool1d(mask,(mask[2],mask[3]))
        return mask

class Global_D(nn.Module):
    def __init__(self,grad_reverse_lambda=1.0,):
        super(Global_D,self).__init__()
        self.conv1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classifer.bias, bias_value)


    def forward(self,x):
        x = self.grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.avgpool(x)

        return x


class MAE_layer(nn.Module):
    def __init__(self,in_channel:int,grad_reverse_lambda=0.1):
        super(MAE_layer,self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.classier = nn.Conv2d(self.in_channel, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        # self.relu = nn.SiLU(inplace=True)
        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.LabelResizeLayer = ImageLabelResizeLayer()
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classier.bias, bias_value)

    def forward(self, x,need_backprop):
        x = self.grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.classier(x)
        if need_backprop is not None:
            label=self.LabelResizeLayer(x,need_backprop)
        else:
            label=None
        return x,label

if __name__=='__main__':
    model = Global_D()
    input = torch.ones(2, 2048, 20, 20)
    p = model(input)
    print(p.shape)