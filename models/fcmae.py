import torch
from models.backbone.convnextv2 import DropPath,Block
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度可分离卷积包括深度卷积和逐点卷积两个步骤
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      stride=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,
                      stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class FCMAE_Decoder(nn.Module):
    def __init__(self,
                 decoder_embed_dim=256,
                 decoder_output_dim=256,
                 drop_path=0.,):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        # self.dwconv = nn.Conv2d(decoder_embed_dim, decoder_embed_dim, kernel_size=7, padding=3, groups=decoder_embed_dim)
        # self.act = nn.GELU()
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.dsc1 = DepthwiseSeparableConv(decoder_embed_dim,4*decoder_output_dim)
        # self.dsc2 = DepthwiseSeparableConv(decoder_output_dim * 4, 4 * decoder_output_dim)
        # self.dsc3 = DepthwiseSeparableConv(decoder_output_dim * 4, decoder_output_dim)
        self.block = Block(dim=decoder_embed_dim)



    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        x = self.block(x)


        return x
