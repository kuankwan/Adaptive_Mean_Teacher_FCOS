import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from utils.misc import sigmoid_focal_loss,condition_focal_loss
import random


class QFLLoss_s(nn.Module):
    def __init__(self,qfl_loss_weight,qfl_loss_beta):
        super(QFLLoss_s, self).__init__()
        self.weight=qfl_loss_weight
        self.beta=qfl_loss_beta

    def forward(self,preds,targets,avg_factor=None):
        loss=F.binary_cross_entropy_with_logits(preds,targets,reduction='none')
        preds_sigmoid=preds.sigmoid()
        scale_factor=(preds_sigmoid-targets).abs().pow(self.beta)
        loss=loss*scale_factor
        loss=loss.mean()
        if(not avg_factor is None):
            return loss/avg_factor
        return loss

class computer_da_loss(object):
    def __init__(self,device,qfl_loss_weight=1.0,qfl_loss_beta=2.0,lambda1=1.0,lambda2=1.0,lambda3=1.0):
        self.qfl = QFLLoss_s(qfl_loss_weight,qfl_loss_beta)
        self.Lambda1 = lambda1
        self.Lambda2 = lambda2
        self.Lambda3 = lambda3
        self.Device = device


    def computer_loss(self,sources,targets):
        s1,s2,s3 = sources
        t1,t2,t3 = targets

        s1 = s1.permute(0, 2, 3, 1).view(-1, 1)
        t1 = t1.permute(0, 2, 3, 1).view(-1, 1)
        d1_real_label = torch.full(s1.shape,1.0,dtype=torch.float,device=self.Device)
        d1_target_label = torch.full(t1.shape,0.0,dtype=torch.float,device=self.Device)
        layer4_loss = sigmoid_focal_loss(s1, d1_real_label,alpha=-1,reduction='mean') + sigmoid_focal_loss(t1, d1_target_label,alpha=-1,reduction='mean')

        s2 = s2.permute(0, 2, 3, 1).view(-1, 1)
        t2 = t2.permute(0, 2, 3, 1).view(-1, 1)
        d2_real_label = torch.full(s2.shape,1.0,dtype=torch.float,device=self.Device)
        d2_target_label = torch.full(t2.shape,0.0,dtype=torch.float,device=self.Device)
        layer3_loss = sigmoid_focal_loss(s2, d2_real_label,alpha=-1,reduction='mean') + sigmoid_focal_loss(t2, d2_target_label,alpha=-1,reduction='mean')

        s3 = s3.permute(0, 2, 3, 1).view(-1, 1)
        t3 = t3.permute(0, 2, 3, 1).view(-1, 1)
        d3_real_label = torch.full(s3.shape,1.0,dtype=torch.float,device=self.Device)
        d3_target_label = torch.full(t3.shape,0.0,dtype=torch.float,device=self.Device)
        layer2_loss = sigmoid_focal_loss(s3, d3_real_label,alpha=-1,reduction='mean') + sigmoid_focal_loss(t3, d3_target_label,alpha=-1,reduction='mean')


        DA_loss = self.Lambda1 * layer2_loss + self.Lambda2 * layer3_loss + self.Lambda3 * layer4_loss
        da_loss = [layer2_loss,layer3_loss,layer4_loss]

        return DA_loss,da_loss

    def __call__(self,
                 sources,
                 targets,
                 ):
        return self.computer_loss(sources, targets)


class computer_ins_loss(object):
    def __init__(self,device,num_class=1,gamma=2.0):
        self.num_class = num_class
        self.gamma = gamma
        self.beta = 2.0
        self.qfl = QFLLoss_s(1.0, 2.0)
        self.Device = device

    def computer_loss(self,sources,targets):
        source = [spred.permute(0,2,3,1).contiguous().view(-1,self.num_class) for spred in sources]
        target = [tpred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_class) for tpred in targets]
        s_perdict = torch.cat(source,dim=0)
        t_perdict = torch.cat(target,dim=0)
        device = sources[0].device
        real_label = torch.full(s_perdict.shape,0.0,dtype=torch.float,device=self.Device)
        target_label = torch.full(t_perdict.shape,1.0,dtype=torch.float,device=self.Device)



        ins_loss =condition_focal_loss(s_perdict,real_label,reduction='mean',) + condition_focal_loss(t_perdict,target_label,reduction='mean',)



        return ins_loss

    def __call__(self,
                 source_list,
                 target_list,
                 ):
        return self.computer_loss(source_list, target_list)



