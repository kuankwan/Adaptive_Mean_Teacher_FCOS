import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

class self_attn2(nn.Module):
    #Self attention Layer"""

    def __init__(self,in_channels,device):
        super(self_attn2, self).__init__()
        self.in_channels = in_channels
        #print("\nchannels:",self.in_channels)
        #with torch.no_grad():
        #self.dom = domain

        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self._init_weights()

        #print(self_attn2.snconv1x1_attn.weight_orig)

    def _init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
            #print(m,"\n")


    def forward(self, x):
        #self.__init__(in_channels,st)
        #super(self_attn2, self).__init__()
        #with torch.no_grad():

        self.apply(init_weights)

        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        #print(theta.dtype)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        #print("\nx:",x.shape)
        phi = self.snconv1x1_phi(x)
        #print("phiconv:",phi.shape)
        phi = self.maxpool(phi)
        #print("phimax:",phi.shape)
        #phi = phi.view(-1, ch//8, h*w//4)
        phi = phi.view(-1, ch//8, phi.shape[2]*phi.shape[3])
        #print("phiview:",phi.shape)
        # Attn map
        attn = torch.matmul(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        #g = g.view(-1, ch//2, h*w//4)
        g = g.view(-1, ch//2, g.shape[2]*g.shape[3])
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        #attn_g = attn_g.cuda()
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)

        #print("\n________________WRITING_____________\n")
        #f.write(str(domain_p1_en.cpu().detach().numpy()))
        #torch.save(attn_g.cpu().detach().numpy(), "/home2/vkhindkar/ILLUME-PyTorch-main/attn_g1.pt")
        #print("\n____Done____\n")


        #Out
        out = x + self.sigma*attn_g
        #out1 = Variable(out, requires_grad = True).cuda(cuda0)
        #out1 = out.cuda(cuda0)

        # del self.snconv1x1_theta,
        # del self.snconv1x1_phi,
        # del self.snconv1x1_g,
        # del self.snconv1x1_attn,
        # del self.maxpool,   self.softmax,
        # del self.sigma
        #
        # del x, theta, phi, attn, g
        #return -torch.mul(domain, attn_g)

        return out