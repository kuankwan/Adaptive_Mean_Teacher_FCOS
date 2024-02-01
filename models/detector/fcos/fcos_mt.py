import enum
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ...backbone import build_backbone
from ...fcmae import FCMAE_Decoder
from ...neck.fpn import build_fpn
from ...head.decoupled_head import DecoupledHead
from .loss import Criterion
from ...da.DA import D_layer4, D_layer3, D_layer2, _InstanceDA, Global_D, MAE_layer
from ...da.da_loss import *
from ...da.SE_weight_module import ChannelAttention
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
import joblib
import random
from utils.visual_attention import visualize_grid_attention_v2

matplotlib.use('Agg')
# np.set_printoptions(threshold=np.inf)
num1, num_sum = 500, 1000  # 根据样本的实际情况自己改数值


def get_color(labels):
    colors = ["#C05757", "#3939EF"]  # ,"g","y","o"#根据情况自己改配色
    color = []
    for i in range(num1):
        color.append(colors[0])
    for i in range(num1, len(labels)):
        color.append(colors[1])
    return color


def vis_tsne(class_out, target_out,name="img_tsne.jpg"):
    # class_out是需要t-SNE可视化的特征，可以来自模型任意一层，我这里用的是最后一层
    class_out = class_out[:, :, :, :]
    target_out = target_out[:, :, :, :]
    plane = class_out.size(1)
    feats = np.array(
        class_out.permute(0, 2, 3, 1).contiguous().detach().view(-1, plane).cpu())  # t-SNE可视化的特征输入必须是np.array格式
    target_feats = np.array(target_out.permute(0, 2, 3, 1).contiguous().detach().view(-1, plane).cpu())
    print(feats.shape)
    joblib.dump(feats, 'feat.pkl')  # 保存特征
    joblib.dump(target_feats, 'target_feat.pkl')
    X = joblib.load('feat.pkl')  # 读取预保存的特征信息
    Y = joblib.load('target_feat.pkl')
    C = [i for i in range(num_sum)]
    # print(type(X))  # 必须是np.array
    X_embedded = TSNE(n_components=2, init="pca", random_state=0).fit_transform(X[:num1, :])
    Y_embedded = TSNE(n_components=2, init="pca", random_state=0).fit_transform(Y[:num1, :])
    print(X_embedded.shape)
    colors = get_color(C)  # 配置点的颜色
    x = X_embedded[:, 0]  # 横坐标
    y = X_embedded[:, 1]  # 纵坐标
    ax = plt.subplot(111)
    source = plt.scatter(x, y, s=10, c=colors[:num1], linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    x = Y_embedded[:, 0]  # 横坐标
    y = Y_embedded[:, 1]  # 纵坐标
    target = plt.scatter(x, y, s=10, c=colors[num1:], linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.legend((source, target), ('source', 'target'))
    # plt.show()
    plt.savefig(name)  # 保存图像
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


class FCOS_MT(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes=20,
                 conf_thresh=0.05,
                 nms_thresh=0.6,
                 trainable=False,
                 topk=1000):
        super(FCOS_MT, self).__init__()
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

        # scale
        self.scales = nn.ModuleList([Scale() for _ in range(len(self.stride))])


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
            # self.D_layer2 = D_layer2(in_channel=bk_dim[0], grad_reverse_lambda=cfg['grad_reverse_lambda'])
            # self.D_layer3 = D_layer3(in_channel=bk_dim[1], grad_reverse_lambda=cfg['grad_reverse_lambda'])
            # self.D_layer4 = D_layer4(in_channel=bk_dim[2], grad_reverse_lambda=cfg['grad_reverse_lambda'])

            # self.atten_relu = nn.ReLU()
            # self.global_D = nn.ModuleList()
            # self.s_discriminators = nn.ModuleList()
            # for j in range(self.num_classes):
            #     ins = _InstanceDA(grad_reverse_lambda=cfg['ins_grad_reverse_lambda'])
            #     self.s_discriminators.append(ins)
            #     self.global_D.append(Global_D(grad_reverse_lambda=cfg['global_grad_reverse_lambda']))
            # self.alpha = cfg['daloss_alpha']
            # self.beta = cfg['insloss_beta']
            # self.epsilon = 1e-4
            # self.computer_da_loss = computer_da_loss(device=device, lambda1=cfg['lambda1'], lambda2=cfg['lambda2'],
            #                                          lambda3=cfg['lambda3'])
            # self.computer_ins_loss = computer_ins_loss(device=device, num_class=num_classes)


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
            self.mae_layer = MAE_layer(in_channel=self.decoder_output_dim,
                                       grad_reverse_lambda=cfg['grad_reverse_lambda'])



    def _init_layers(self):
        # init cls pred
        for m in self.modules():
            m.requires_grad_(False)


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
        for level, feat in enumerate(pyramid_feats):
            cls_feat, reg_feat = self.head(feat)

            # [1, C, H, W]
            cls_pred = self.cls_pred(cls_feat[-1])
            reg_pred = self.reg_pred(reg_feat[-1])
            ctn_pred = self.ctn_pred(reg_feat[-1])
            # decode box
            _, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [1, C, H, W] -> [H, W, C] -> [M, C]
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

        return bboxes, scores, labels

    def forward(self, x, mask=None, visual=False):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            feats = self.backbone(x)
            # fpn neck
            pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
            pyramid_feats = self.fpn(pyramid_feats)  # [P3, P4, P5, P6, P7]
            # shared head
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_ctn_preds = []
            all_masks = []

            for level, feat in enumerate(pyramid_feats):
                cls_feat, reg_feat = self.head(feat)

                # [B, C, H, W]
                cls_pred = self.cls_pred(cls_feat[-1])
                reg_pred = self.reg_pred(reg_feat[-1])
                ctn_pred = self.ctn_pred(reg_feat[-1])
                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                # cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                # reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                # reg_pred = F.relu(self.scales[level](reg_pred)) * self.stride[level]
                # ctn_pred = ctn_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)

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
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                all_anchors.append(anchors)
            # for i in range(len(all_anchors)):
            #     print(all_anchors[i].shape)
            outputs = {"pred_cls": all_cls_preds,  # List [B, M, C]
                       "pred_reg": all_reg_preds,  # List [B, M, 4]
                       "pred_ctn": all_ctn_preds,  # List [B, M, 1]
                       'strides': self.stride,
                       "mask": all_masks}  # List [B, M,]
            return outputs
