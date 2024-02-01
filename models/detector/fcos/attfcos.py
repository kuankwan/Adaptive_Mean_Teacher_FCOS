import enum
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as plt
from ...backbone import build_backbone
from ...basic.conv import Conv
from ...fcmae import FCMAE_Decoder
from ...head.weak_head import WeakHead
from ...neck.fpn import build_fpn
from ...head.decoupled_head import DecoupledHead
from .loss import Criterion
from ...da.efficient_attention import EfficientAttention
from ...da.DA import D_layer4, D_layer3, D_layer2, _InstanceDA, Global_D, MAE_layer
from ...da.da_loss import *
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
import joblib

import random
from utils.visual_attention import visualize_grid_attention_v2, visualize_mask
from models.SelfAttention import SelfAttention
from utils.misc import ContrastiveLoss
from utils.misc import NT_Xent,shared_loss,cos_loss

matplotlib.use('Agg')
# np.set_printoptions(threshold=np.inf)
num_sum = 250  # 根据样本的实际情况自己改数值
src_feats = []
tgt_feats = []
colors = ["#C05757", "#3939EF"]  # ,"g","y","o"#根据情况自己改配色

def T_SNE(pred):
    pred1 = TSNE(n_components=2, init="pca", random_state=0).fit_transform(pred)
    pred1_min,pred1_max = pred1.min(0),pred1.max(0)
    pred1 = (pred1 - pred1_min) / (pred1_max - pred1_min)
    return pred1

def draw(src,tgt,save_name):
    ax = plt.subplot(111)
    my_font1 = {"family": "Times New Roman", "size": 12, "style": "normal","weight":"bold"}
    source = plt.scatter(src[:,0], src[:,1], s=10, c="#C05757", linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    target = plt.scatter(tgt[:,0], tgt[:,1], s=10, c="#3939EF", linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.legend((source, target), ('source', 'target'),prop=my_font1)
    # plt.show()
    save_path = "./tsne_result/" + save_name + ".jpg"
    print("==================finish save "+save_path+"=====================")
    plt.savefig(save_path,dpi=300,bbox_inches='tight')  # 保存图像
    plt.close()


def vis_tsne(class_out, target_out,save_name):
    # class_out是需要t-SNE可视化的特征，可以来自模型任意一层，我这里用的是最后一层
    class_out = class_out[:, :, :, :]
    target_out = target_out[:, :, :, :]
    b,c,w,h = class_out.size()
    b1,c1,w1,h1 = target_out.size()
    src_out = class_out.contiguous().detach().reshape(b,c,w*h)
    tgt_out = target_out.contiguous().detach().reshape(b1,c1,w1*h1)
    source_outs = torch.mean(src_out,dim=2)
    target_outs = torch.mean(tgt_out,dim=2)
    src_feats.append(source_outs)
    tgt_feats.append(target_outs)
    if len(src_feats) == num_sum and len(tgt_feats) == num_sum:
        source_outs = [out.view(b,-1) for out in src_feats]
        target_outs = [out.view(b1, -1) for out in tgt_feats]
        source_out = torch.cat(source_outs,dim=0)
        target_out = torch.cat(target_outs, dim=0)
        print(source_out.size())
        src_pred = T_SNE(np.array(source_out.cpu()))
        tgt_pred = T_SNE(np.array(target_out.cpu()))
        draw(src_pred,tgt_pred,save_name)
    if len(src_feats) >= num_sum and len(tgt_feats) >= num_sum:
        src_feats.clear()
        tgt_feats.clear()

    # plane = class_out.size(1)
    # feats = np.array(
    #     class_out.permute(0, 2, 3, 1).contiguous().detach().view(-1, plane).cpu())  # t-SNE可视化的特征输入必须是np.array格式
    # target_feats = np.array(target_out.permute(0, 2, 3, 1).contiguous().detach().view(-1, plane).cpu())
    # print(feats.shape)
    # joblib.dump(feats, 'feat.pkl')  # 保存特征
    # joblib.dump(target_feats, 'target_feat.pkl')
    # X = joblib.load('feat.pkl')  # 读取预保存的特征信息
    # Y = joblib.load('target_feat.pkl')
    # C = [i for i in range(num_sum)]
    # # print(type(X))  # 必须是np.array
    # X_embedded = TSNE(n_components=2, init="pca", random_state=0).fit_transform(X[:num1, :])
    # Y_embedded = TSNE(n_components=2, init="pca", random_state=0).fit_transform(Y[:num1, :])
    # print(X_embedded.shape)
    # x = X_embedded[:, 0]  # 横坐标
    # y = X_embedded[:, 1]  # 纵坐标
    # ax = plt.subplot(111)
    # source = plt.scatter(x, y, s=10, c=colors[:num1], linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    # # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # # ax.yaxis.set_major_formatter(NullFormatter())
    # x = Y_embedded[:, 0]  # 横坐标
    # y = Y_embedded[:, 1]  # 纵坐标
    # target = plt.scatter(x, y, s=10, c=colors[num1:], linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.legend((source, target), ('source', 'target'))
    # # plt.show()
    # plt.savefig("tsne.jpg")  # 保存图像
    # plt.close()

def visualize_feature_map(processed,m_processed,save_name):

    assert len(processed)==len(m_processed)
    # fig = plt.figure(figsize=(50, 50))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.axis('off')
    plt.margins(0, 0)
    for i in range(len(processed)):  # len(processed) = 17
        # a = fig.add_subplot(5, 5, i + 1)
        img_plot = plt.imshow(processed[i])
        # a.set_title(i, fontsize=30)  # names[i].split('(')[0] 结果为Conv2d
        plt.savefig(save_name + str(i) + '.jpg', bbox_inches='tight',format='png',pad_inches = 0,transparent=True, dpi=300)
    for i in range(len(m_processed)):  # len(processed) = 17
        # a = fig.add_subplot(5, 5, i + 1+len(processed))
        img_plot = plt.imshow(m_processed[i])
        # a.set_title(i, fontsize=30)  # names[i].split('(')[0] 结果为Conv2d
        plt.savefig(save_name + str(i)+'_mask' + '.jpg', bbox_inches='tight',format='png',pad_inches = 0,transparent=True, dpi=300)


    # plt.savefig(save_name+'.jpg', bbox_inches='tight')
    plt.close()




def visualize_attention_map(attention_map, root):
    """
    The attention map is a matrix ranging from 0 to 1, where the greater the value,
    the greater attention is suggests.
    :param attention_map: np.numpy matrix hanging from 0 to 1
    :return np.array matrix with rang [0, 255]
    """

    attention_map_color = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1], 3],
        dtype=np.uint8
    )

    red_color_map = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1]],
        dtype=np.uint8
    ) + 255
    red_color_map = red_color_map * attention_map
    red_color_map = np.array(red_color_map, dtype=np.uint8)

    attention_map_color[:, :, 2] = red_color_map
    cv.imwrite(root, attention_map_color)

    return attention_map_color


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale

def m(x):
    x = 1 - x.sigmoid() * torch.log(x.sigmoid())
    return x

def w(x):
    w_p = 1 + x.sigmoid() * torch.log(x.sigmoid())
    return w_p


def reshape_input(x):
    n,c,w,h = x.size()
    x = x.reshape((n, c * h * w))
    return x

class classifer(nn.Module):
    def __init__(self,input,output):
        super(classifer, self).__init__()
        self.input = input
        self.output = output
        self.conv1 = nn.Conv2d(self.input,self.input,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(self.input, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.liner = nn.Linear(self.input,self.output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.avg_pooling(x)
        # x = torch.flatten(x, 1)
        # x = self.liner(x)
        return x



class FCOS(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes=20,
                 conf_thresh=0.05,
                 nms_thresh=0.6,
                 trainable=False,
                 topk=1000):
        super(FCOS, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk

        # backbone
        self.backbone, bk_dim = build_backbone(model_name=cfg['backbone'],
                                               pretrained=trainable,
                                               norm_type=cfg['norm_type'])

        # fpn neck
        self.fpn = build_fpn(cfg=cfg,
                             in_dims=bk_dim,
                             out_dim=cfg['head_dim'],
                             from_c5=cfg['from_c5'],
                             p6_feat=cfg['p6_feat'],
                             p7_feat=cfg['p7_feat'])

        # head
        self.head = DecoupledHead(head_dim=cfg['head_dim'],
                                  num_cls_head=cfg['num_cls_head'],
                                  num_reg_head=cfg['num_reg_head'],
                                  act_type=cfg['act_type'],
                                  norm_type=cfg['head_norm'])

        # pred
        self.cls_pred = nn.Conv2d(cfg['head_dim'],
                                  self.num_classes,
                                  kernel_size=3,
                                  padding=1)
        self.reg_pred = nn.Conv2d(cfg['head_dim'],
                                  4,
                                  kernel_size=3,
                                  padding=1)
        self.ctn_pred = nn.Conv2d(cfg['head_dim'],
                                  1,
                                  kernel_size=3,
                                  padding=1)

        self.weak_head = WeakHead(head_dim=cfg['head_dim'],
                                  num_cls_head=cfg['num_cls_head'],
                                  num_reg_head=cfg['num_reg_head'],
                                  act_type=cfg['act_type'],
                                  norm_type=cfg['head_norm'])

        self.weak_target_pred = nn.Conv2d(cfg['head_dim'],
                                  self.num_classes,
                                  kernel_size=3,
                                  padding=1)
        self.weak_reg_pred = nn.Conv2d(cfg['head_dim'],
                                  4,
                                  kernel_size=3,
                                  padding=1)
        self.weak_ctn_pred = nn.Conv2d(cfg['head_dim'],
                                  1,
                                  kernel_size=3,
                                  padding=1)
        self.theta = cfg['theta']
        self.delta = cfg['global_delta']
        self.mask_ratio = cfg['mask_ratio']
        self.proj = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=1)
        self.decoder_embed_dim = cfg['decoder_embed_dim']
        self.decoder_output_dim = cfg['decoder_output_dim']
        self.source_mask_token = nn.Parameter(torch.zeros(1, self.decoder_embed_dim, 1, 1))
        self.target_mask_token = nn.Parameter(torch.zeros(1, self.decoder_embed_dim, 1, 1))
        self.fcmaedecoder = FCMAE_Decoder(decoder_embed_dim=self.decoder_embed_dim,
                                          decoder_output_dim=self.decoder_output_dim)
        self.mae_layer = MAE_layer(in_channel=self.decoder_output_dim, grad_reverse_lambda=cfg['grad_reverse_lambda'])

        # scale
        self.scales = nn.ModuleList([Scale() for _ in range(len(self.stride))])

        if trainable:
            # init bias
            self._init_pred_layers()

        # criterion
        if self.trainable:
            self.criterion = Criterion(cfg=cfg,
                                       device=device,
                                       alpha=cfg['alpha'],
                                       gamma=cfg['gamma'],
                                       loss_cls_weight=cfg['loss_cls_weight'],
                                       loss_reg_weight=cfg['loss_reg_weight'],
                                       loss_ctn_weight=cfg['loss_ctn_weight'],
                                       num_classes=num_classes)
        # DA
        self.D_layer2 = D_layer2(in_channel=bk_dim[0],grad_reverse_lambda=cfg['grad_reverse_lambda'])
        self.D_layer3 = D_layer3(in_channel=bk_dim[1],grad_reverse_lambda=cfg['grad_reverse_lambda'])
        self.D_layer4 = D_layer4(in_channel=bk_dim[2], grad_reverse_lambda=cfg['grad_reverse_lambda'])
        self.cr = nn.Sequential(
            Conv(self.decoder_output_dim*2,self.decoder_output_dim*2,k=3, p=1, s=1, act_type=cfg['act_type'], norm_type=cfg['head_norm']),
            Conv(self.decoder_output_dim * 2, self.decoder_output_dim * 2, k=3, p=1, s=1, act_type=cfg['act_type'],
                 norm_type=cfg['head_norm']),
            nn.Conv2d(self.decoder_output_dim * 2,self.num_classes,kernel_size=3,padding=1)
        )
        # self.atten_relu = nn.ReLU()
        # self.global_D = nn.ModuleList()
        # self.s_discriminators = nn.ModuleList()
        # for j in range(self.num_classes):
        #     ins = _InstanceDA(grad_reverse_lambda=cfg['ins_grad_reverse_lambda'])
        #     self.s_discriminators.append(ins)
        #     self.global_D.append(Global_D(grad_reverse_lambda=cfg['global_grad_reverse_lambda']))
        # self.beta = cfg['insloss_beta']
        # self.epsilon = 1e-4
        self.computer_da_loss = computer_da_loss(device=device,lambda1=cfg['lambda1'],lambda2=cfg['lambda2'],lambda3=cfg['lambda3'])
        # self.computer_ins_loss = computer_ins_loss(device=device,num_class=num_classes)







    def _init_pred_layers(self):
        # init cls pred
        nn.init.normal_(self.cls_pred.weight, mean=0, std=0.01)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
        # init reg pred
        nn.init.normal_(self.reg_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)
        # init ctn pred
        nn.init.normal_(self.ctn_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors

    def decode_boxes(self, anchors, pred_deltas):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [M, 4] (l, t, r, b)
        """
        # x1 = x_anchor - l, x2 = x_anchor + r
        # y1 = y_anchor - t, y2 = y_anchor + b
        pred_x1y1 = anchors - pred_deltas[..., :2]
        pred_x2y2 = anchors + pred_deltas[..., 2:]
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

    def gen_random_mask(self, x,mask_ratio):
        N = x.shape[0]
        L = x.shape[2] * x.shape[3]
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.fcmaedecoder(x)
        return x

    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    @torch.no_grad()
    def inference_single_image(self, x):
        img_h, img_w = x.shape[2:]
        # backbone
        feats = self.backbone(x)
        # fpn neck
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
        pyramid_feats = self.fpn(pyramid_feats)
        # shared head
        all_scores = []
        all_labels = []
        all_bboxes = []
        processed = []
        cos_sim_list = []
        for level, feat in enumerate(pyramid_feats):
            cls_feat, reg_feat = self.head(feat)
            feature_map = feat.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
            processed.append(gray_scale.data.cpu().numpy())

            # [1, C, H, W]
            cls_pred = self.cls_pred(cls_feat[-1])
            reg_pred = self.reg_pred(reg_feat[-1])
            ctn_pred = self.ctn_pred(reg_feat[-1])
            # decode box
            N, C, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [1, C, H, W] -> [H, W, C] -> [M, C]
            cls_cos = cls_pred[0].permute(1, 2, 0).contiguous().view(N,-1, self.num_classes)
            cos_sim_list.append(cls_cos)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)
            reg_pred = F.relu(self.scales[level](reg_pred)) * self.stride[level]
            ctn_pred = ctn_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)

            # scores
            scores, labels = torch.max(torch.sqrt(cls_pred.sigmoid() * ctn_pred.sigmoid()), dim=-1)

            # [M, 4]
            anchors = self.generate_anchors(level, fmp_size)
            # topk
            if scores.shape[0] > self.topk:
                scores, indices = torch.topk(scores, self.topk)
                labels = labels[indices]
                reg_pred = reg_pred[indices]
                anchors = anchors[indices]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchors, reg_pred)

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)
        cos_sim = torch.cat(cos_sim_list,dim=1)
        cos_sim,_ = torch.max(cos_sim,dim=1)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # normalize bbox
        bboxes[..., [0, 2]] /= img_w
        bboxes[..., [1, 3]] /= img_h
        bboxes = bboxes.clip(0., 1.)
        # fig = plt.figure(figsize=(30, 50))
        # for i in range(len(processed)):  # len(processed) = 17
        #     a = fig.add_subplot(5, 5, i + 1)
        #     img_plot = plt.imshow(processed[i])
        #     a.axis("off")
        #     a.set_title(i, fontsize=30)  # names[i].split('(')[0] 结果为Conv2d
        #
        # plt.savefig('fpn_feature_maps.jpg', bbox_inches='tight')
        # plt.close()

        return bboxes, scores, labels

    def forward(self, x, mask=None, targets=None, tar_x=None, open_ins=False,weak_train=False, visual=False,save_name="tsne"):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            feats = self.backbone(x)
            source_layer4,_ = self.D_layer4(feats['layer4'],need_backprop=None)
            source_layer3,_ = self.D_layer3(feats['layer3'],need_backprop=None)
            source_layer2,_ = self.D_layer2(feats['layer2'],need_backprop=None)


            pyramid_feats0 = feats['layer2']
            pyramid_feats1 = feats['layer3']
            pyramid_feats2 = feats['layer4']

            if tar_x is not None:
                tar_feats = self.backbone(tar_x)
                target_layer4,_ = self.D_layer4(tar_feats['layer4'],need_backprop=None)
                target_layer3,_ = self.D_layer3(tar_feats['layer3'],need_backprop=None)
                target_layer2,_ = self.D_layer2(tar_feats['layer2'],need_backprop=None)

                pyramid_tar_feats0 = tar_feats['layer2']
                pyramid_tar_feats1 = tar_feats['layer3']
                pyramid_tar_feats2 = tar_feats['layer4']

                pyramid_tar_feats = [pyramid_tar_feats0, pyramid_tar_feats1, pyramid_tar_feats2]
                pyramid_tar_feats = self.fpn(pyramid_tar_feats)
                tar_feats_iter = iter(pyramid_tar_feats)
                # vis_tsne(pyramid_feats2, pyramid_tar_feats2,save_name=save_name+"fpn")
            else:
                target_layer4, target_layer3, target_layer2, tar_feats_iter = None, None, None, None

            # fpn neck
            pyramid_feats = [pyramid_feats0,pyramid_feats1,pyramid_feats2]
            pyramid_feats = self.fpn(pyramid_feats)  # [P3, P4, P5, P6, P7]

            # shared head
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_ctn_preds = []
            stu_cls_preds = []
            stu_reg_preds = []
            stu_ctn_preds = []
            all_masks = []
            all_tar_cls_preds = []
            all_ins_source = []
            all_ins_target = []
            source_ins_weight = []
            target_ins_weight = []
            mae_source = []
            mae_target = []
            s_global_all = []
            t_global_all = []
            mse_loss = []
            cr_list = []
            s_processed = []
            t_processed = []
            sm_processed = []
            tm_processed = []
            if weak_train:
                for level, feat in enumerate(pyramid_feats):
                    target_feat = next(tar_feats_iter)
                    # FCMAE
                    #source
                    source_mask = self.gen_random_mask(feat,mask_ratio=self.mask_ratio[level])
                    ns, cs, hs, ws = feat.shape
                    source_mask = source_mask.reshape(-1, hs, ws).unsqueeze(1).type_as(feat)
                    source_mask_token = self.source_mask_token.repeat(ns, 1, hs, ws)
                    source_input = feat * (1. - source_mask) + source_mask_token * source_mask
                    source_output = self.fcmaedecoder(source_input)
                    #可视化
                    feature_map = feat[0,:,:,:].squeeze(0)
                    gray_scale = torch.sum(feature_map, 0)
                    gray_scale = gray_scale / feature_map.shape[0]
                    s_processed.append(gray_scale.data.cpu().numpy())

                    mask_map = source_output[0,:,:,:].squeeze(0)
                    mask_scale = torch.sum(mask_map, 0)
                    mask_scale = mask_scale / mask_map.shape[0]
                    sm_processed.append(mask_scale.data.cpu().numpy())
                    if visual:
                        visualize_mask(x[0], './visualization/source_mask/', source_mask[0].squeeze(0).data.cpu().numpy(), level=level,save_image=True)

                    # target
                    target_mask = self.gen_random_mask(target_feat,mask_ratio=self.mask_ratio[level])
                    nt, ct, ht, wt = target_feat.shape
                    target_mask = target_mask.reshape(-1, ht, wt).unsqueeze(1).type_as(target_feat)
                    target_mask_token = self.target_mask_token.repeat(nt, 1, ht, wt)
                    target_input = target_feat * (1. - target_mask) + target_mask_token * target_mask
                    target_output = self.fcmaedecoder(target_input)
                    # 可视化
                    t_feature_map = target_feat[0,:,:,:].squeeze(0)
                    t_gray_scale = torch.sum(t_feature_map, 0)
                    t_gray_scale = t_gray_scale / t_feature_map.shape[0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
                    t_processed.append(t_gray_scale.data.cpu().numpy())
                    t_mask_map = target_output[0,:,:,:].squeeze(0)
                    t_mask_scale = torch.sum(t_mask_map, 0)
                    t_mask_scale = t_mask_scale / t_mask_map.shape[0]
                    tm_processed.append(t_mask_scale.data.cpu().numpy())
                    if visual:
                        visualize_mask(tar_x[0], './visualization/target_mask/', target_mask[0].squeeze(0).data.cpu().numpy(), level=level,save_image=True)

                    s_mask,_ = self.mae_layer(source_output,need_backprop=None)
                    t_mask,_ = self.mae_layer(target_output,need_backprop=None)
                    mae_source.append(s_mask)
                    mae_target.append(t_mask)
                    cr_feat = torch.cat([source_output,target_output],dim=1)
                    cr_output = self.cr(cr_feat)
                    cr_list.append(cr_output)
                    # student head
                    source_size = hs * ws
                    target_size = ht * wt
                    # if source_size > target_size:
                    #     target_feat = F.interpolate(target_feat,(hs,ws),mode='bilinear',align_corners=True)
                    # elif source_size < target_size:
                    #     feat = F.interpolate(feat,(ht,wt),mode='bilinear',align_corners=True)

                    cls_feat, reg_feat = self.weak_head(target_feat)
                    weak_target_pred = self.weak_target_pred(cls_feat[-1])
                    reg_pred = self.weak_reg_pred(reg_feat[-1])
                    ctn_pred = self.weak_ctn_pred(reg_feat[-1])
                    stu_cls_preds.append(weak_target_pred)
                    stu_reg_preds.append(reg_pred)
                    stu_ctn_preds.append(ctn_pred)
                if visual:
                    visualize_feature_map(s_processed, sm_processed, save_name='./visualization/source_feature/source_feature_map')
                    visualize_feature_map(t_processed, tm_processed, save_name='./visualization/target_feature/target_feature_map')

            else:
                for level, feat in enumerate(pyramid_feats):
                    cls_feat, reg_feat = self.head(feat)

                    # [B, C, H, W]
                    cls_pred = self.cls_pred(cls_feat[-1])
                    reg_pred = self.reg_pred(reg_feat[-1])
                    ctn_pred = self.ctn_pred(reg_feat[-1])
                    stu_cls_preds.append(cls_pred)
                    stu_reg_preds.append(reg_pred)
                    stu_ctn_preds.append(ctn_pred)

                    # s_weight = torch.sigmoid(cls_pred.sigmoid() * ctn_pred.sigmoid() * self.theta)
                    # s_global_weight=F.softmax(torch.flatten(F.adaptive_avg_pool2d(cls_pred.sigmoid(),(1,1)),1),dim=1)
                    # source_ins_weight.append(s_global_weight)



                    B, _, H, W = cls_pred.size()
                    fmp_size = [H, W]
                    # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                    cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                    reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                    reg_pred = F.relu(self.scales[level](reg_pred)) * self.stride[level]
                    ctn_pred = ctn_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)

                    all_cls_preds.append(cls_pred)
                    all_reg_preds.append(reg_pred)
                    all_ctn_preds.append(ctn_pred)
                    # for i in range(B):
                    #     for j in range(self.num_classes):
                    #         s = cls_pred[i]
                    #         print(torch.max(torch.sigmoid(s[:,j:j+1])))
                    #     print(torch.mean(torch.sigmoid(ctn_pred[i])))

                    if mask is not None:
                        # [B, H, W]
                        mask_i = torch.nn.functional.interpolate(mask[None], size=[H, W]).bool()[0]
                        # [B, H, W] -> [B, M]
                        mask_i = mask_i.flatten(1)

                        all_masks.append(mask_i)

                    if tar_feats_iter is not None and open_ins:
                        target_feat = next(tar_feats_iter)
                        tar_cls_feat, tar_reg_feat = self.head(target_feat)
                        tar_cls_pred = self.cls_pred(tar_cls_feat[-1])
                        tar_ctn_pred = self.ctn_pred(tar_reg_feat[-1])

                        t_weight = torch.sigmoid(tar_cls_pred.sigmoid() * tar_ctn_pred.sigmoid() * self.theta)
                        t_global_weight = F.softmax(torch.flatten(F.adaptive_avg_pool2d(tar_cls_pred.sigmoid(),(1,1)),1),dim=1)
                        target_ins_weight.append(t_global_weight)
#                         tar_cls_preds = tar_cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
#                         all_tar_cls_preds.append(tar_cls_preds)
                        source_list = []
                        target_list = []
                        s_global_list = []
                        t_global_list = []
                        # for i in range(self.num_classes):
                        #     src_input = feat * s_weight[0, i:i + 1, :, :] + feat
                        #     s_ins_pred = self.s_discriminators[i](src_input)
                        #     source_list.append(s_ins_pred)
                        #     tgt_input = tar_feat * t_weight[0, i:i + 1, :, :] + tar_feat
                        #     t_ins_pred = self.s_discriminators[i](tgt_input)
                        #     target_list.append(t_ins_pred)
                        #     sweight_hat = 1 + F.sigmoid(s_ins_pred) * torch.log(F.sigmoid(s_ins_pred))
                        #     s_input= sweight_hat * feat + feat
                        #     # s_input = self.eff_att(s_input)
                        #     s_local_pred = self.global_D[i](s_input)
                        #     s_global_list.append(s_local_pred)
                        #     tweight_hat = 1 + F.sigmoid(t_ins_pred) * torch.log(F.sigmoid(t_ins_pred))
                        #     t_input= tweight_hat * tar_feat + tar_feat
                        #     # t_input = self.eff_att(t_input)
                        #     t_local_pred = self.global_D[i](t_input)
                        #     t_global_list.append(t_local_pred)
                            # if visual:
                            #     s_attention_map = s_weight[0, i:i + 1, :, :].permute(1, 2, 0).squeeze(
                            #         dim=2)
                            #     s_attention_map = Variable(s_attention_map, requires_grad=False)
                            #     s_attention_map = np.asarray(s_attention_map.cpu())
                            #     visualize_grid_attention_v2(x[0], './visualization/source/', s_attention_map, i, level,
                            #                                 save_image=True)
                            #
                            #     t_attention_map = t_weight[0, i:i + 1, :, :].permute(1, 2, 0).squeeze(
                            #         dim=2)
                            #     t_attention_map = Variable(t_attention_map, requires_grad=False)
                            #     t_attention_map = np.asarray(t_attention_map.cpu())
                            #     visualize_grid_attention_v2(tar_x[0], './visualization/target/', t_attention_map, i,
                            #                                 level, save_image=True)

                                # tweight_hat_map = tweight_hat[0].permute(1, 2, 0).squeeze(dim=2)
                                # tweight_hat_map = Variable(tweight_hat_map, requires_grad=False)
                                # tweight_hat_map = np.asarray(tweight_hat_map.cpu())
                                # visualize_grid_attention_v2(tar_x[0], './visualization/global_target/', tweight_hat_map, i,
                                #                             level, save_image=True)
                                #
                                # sweight_hat_map = sweight_hat[0].permute(1, 2, 0).squeeze(dim=2)
                                # sweight_hat_map = Variable(sweight_hat_map, requires_grad=False)
                                # sweight_hat_map = np.asarray(sweight_hat_map.cpu())
                                # visualize_grid_attention_v2(x[0], './visualization/global_source/', sweight_hat_map,
                                #                             i,
                                #                             level, save_image=True)

                        # s_pred = torch.cat(source_list, dim=1)
                        # t_pred = torch.cat(target_list, dim=1)
                        # s_global_p = torch.cat(s_global_list, dim=1)
                        # s_global_all.append(s_global_p)
                        # t_global_p = torch.cat(t_global_list, dim=1)
                        # t_global_all.append(t_global_p)
                        # all_ins_source.append(s_pred)
                        # all_ins_target.append(t_pred)
                    # generate anchor boxes: [M, 4]
                    anchors = self.generate_anchors(level, fmp_size)
                    all_anchors.append(anchors)
            # for i in s_local_list:
            #     print(i.shape)

            # output dict
            outputs = {"pred_cls": all_cls_preds,  # List [B, M, C]
                       "pred_reg": all_reg_preds,  # List [B, M, 4]
                       "pred_ctn": all_ctn_preds,  # List [B, M, 1]
                       'strides': self.stride,
                       "mask": all_masks}  # List [B, M,]
            stu_outputs = {"pred_cls": stu_cls_preds,  # List [B, M, C]
                       "pred_reg": stu_reg_preds,  # List [B, M, 4]
                       "pred_ctn": stu_ctn_preds,  # List [B, M, 1]
                       "mae_source": mae_source,
                       "mae_target": mae_target,
                       "pred_cr":cr_list,
                       "mse_loss":mse_loss,
                       'strides': self.stride,
                       "mask": all_masks}  # List [B, M,]
            if weak_train:
                loss_dict = {}
            else:
            # loss
                loss_dict = self.criterion(outputs=outputs,
                                           targets=targets,
                                           anchors=all_anchors)
                pga_loss = torch.tensor(0, dtype=torch.float)
                local_loss = torch.tensor(0, dtype=torch.float)
                global_loss = torch.tensor(0, dtype=torch.float)
                if tar_x is not None:
                    # if open_ins:
                    #     self.alpha = 1.0
                    # else:
                    #     self.alpha = 0.0
                    pga_loss,da_loss_list = self.computer_da_loss([source_layer4, source_layer3, source_layer2],
                                                    [target_layer4, target_layer3, target_layer2])
                    pga_loss = pga_loss / 3
                    loss_dict.update({'pga_loss': pga_loss})
                #     # snce_loss = (self.infonce1(reshape_input(fg1),reshape_input(tfg1)) + self.infonce2(reshape_input(fg2),reshape_input(tfg2)) + self.infonce3(reshape_input(fg3),reshape_input(tfg3)))/3
                #     # losses = loss_dict['losses'] + snce_loss
                #     # loss_dict.update({'losses': losses, 'snce_loss': snce_loss})
                #
                #     # cos_losses = ((self.cos_loss(fg1, tfg1)) + (self.cos_loss(fg2,tfg2)) + (self.cos_loss(fg3, tfg3)))/3.0
                #     # losses = loss_dict['losses'] + cos_losses
                #     # loss_dict.update({'losses': losses, 'cos_losses': cos_losses})
                #
                #     # share_loss = (self.shared_loss(fg1,tfg1) + self.shared_loss(fg2,tfg2) + self.shared_loss(fg3,tfg3))/3
                #     # losses = loss_dict['losses'] + share_loss
                #     # loss_dict.update({'losses': losses, 'share_loss': share_loss})
                #
                #     if tar_feats_iter is not None and open_ins:
                #         s_global_all = torch.cat(s_global_all,dim=1)
                #         t_global_all = torch.cat(t_global_all, dim=1)
                #         local_loss = self.computer_ins_loss(all_ins_source, all_ins_target)
                #         local_loss = local_loss * self.beta
                #         losses = loss_dict['losses'] + local_loss
                #         loss_dict.update({'losses': losses, 'local_loss': local_loss})
                #         global_loss = F.binary_cross_entropy_with_logits(s_global_all,torch.full(s_global_all.shape,0.0,dtype=torch.float,device=self.device),reduction="mean") + F.binary_cross_entropy_with_logits(t_global_all,torch.full(t_global_all.shape,1.0,dtype=torch.float,device=self.device),reduction='mean')
                #         global_loss = global_loss * self.delta
                #         losses = loss_dict['losses'] + global_loss
                #         loss_dict.update({'losses': losses, 'global_loss': global_loss})
            return loss_dict,stu_outputs
