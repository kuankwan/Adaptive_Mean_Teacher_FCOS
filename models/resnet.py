import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from torch.hub import load_state_dict_from_url
from torchvision.models._utils import IntermediateLayerGetter
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-f46c3f97.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-b641f3a9.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-089c0848.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-b260af35.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class resnet(torch.nn.Module):

    all_layers={18:[2, 2, 2, 2],
                34:[3, 4, 6, 3],
                50:[3,4,6,3],
                101:[3,4,23,3],
                152:[3,8,36,3]}

    def __init__(self,renet_depth=50,groups: int = 1, width_per_group: int = 64,replace_stride_with_dilation: Optional[List[bool]] = None,norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(resnet, self).__init__()
        # assert cfg.resnet_depth in self.all_layers.keys(),"不支持除resnet50,resnet101,resnet152之外的resnet类型"
        if norm_layer == 'BN':
            _norm_layer = nn.BatchNorm2d
        elif norm_layer == 'FrozeBN':
            _norm_layer = FrozenBatchNorm2d
        else:
            _norm_layer = nn.BatchNorm2d
        self.depth = renet_depth
        if renet_depth not in self.all_layers.keys():
            raise ValueError('Renet_depth is not in [18,34,50,101,152]')
        self._norm_layer = _norm_layer
        self.groups = groups
        self.dilation = 1
        self.inplanes = 64
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = _norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        block = Bottleneck if renet_depth>=50 else BasicBlock
        layers = self.all_layers[renet_depth]
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        self._freeze_stages()


    def _freeze_stages(self):
        """
        固定resnet中的参数，其中stem阶段的参数一定固定，layer1-layer4可选
        :return:
        """
        for m in [self.conv1,self.bn1]:
            for param in m.parameters():
                param.requires_grad = False

        #这里的cfg.freeze_stages表示需要固定参数的stage的数量，例如cfg.freeze_stages=2时，表示layer1和layer2的参数会被固定
        for i in range(1, 2):
            m=getattr(self,"layer"+str(i))
            for param in m.parameters():
                param.requires_grad=False

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def train(self, mode=True):
        super(resnet, self).train(mode)
        for m in self.modules():
            if(isinstance(m,torch.nn.BatchNorm2d)):
                m.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5


def _resnet(model_name='resnet50', pretrained=False,norm_layer = None):
    if model_name in ['resnet18']:
        backbone = resnet(renet_depth=18,norm_layer= norm_layer)

        feats = [128, 256, 512]  # [C3, C4, C5]

    elif model_name in ['resnet34']:
        backbone = resnet(renet_depth=34,norm_layer= norm_layer)

        feats = [128, 256, 512]  # [C3, C4, C5]

    elif model_name in ['resnet50']:
        backbone = resnet(renet_depth=50,norm_layer= norm_layer)

        feats = [512, 1024, 2048]  # [C3, C4, C5]

    elif model_name in ['resnet101']:
        backbone = resnet(renet_depth=101,norm_layer= norm_layer)

        feats = [512, 1024, 2048]  # [C3, C4, C5]

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[model_name],
                                              progress=True)
        backbone.load_state_dict(state_dict,strict=False)
        print('---------loaded pretrained backbone---------')


    return backbone, feats



