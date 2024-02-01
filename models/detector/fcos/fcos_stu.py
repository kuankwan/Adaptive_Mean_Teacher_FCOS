import enum
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ...backbone import build_backbone
from ...neck.fpn import build_fpn
from ...head.decoupled_head import DecoupledHead
from .loss import Criterion
from ...da.DA import D_layer4, D_layer3, D_layer2, _InstanceDA
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

class QFLLoss(nn.Module):
    def __init__(self):
        super(QFLLoss, self).__init__()
        self.weight=1.0
        self.beta=2.0

    def forward(self,preds,targets,avg_factor=None):
        loss=F.binary_cross_entropy_with_logits(preds,targets,reduction='none')
        preds_sigmoid=preds.sigmoid()
        scale_factor=(preds_sigmoid-targets).abs().pow(self.beta)
        loss=loss*scale_factor
        loss=loss.sum()*self.weight
        if(not avg_factor is None):
            return loss/avg_factor
        return loss

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


class FCOS_STU(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes=20,
                 conf_thresh=0.05,
                 nms_thresh=0.6,
                 trainable=False,
                 topk=1000):
        super(FCOS_STU, self).__init__()
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
        self.D_layer4 = D_layer4()
        self.D_layer3 = D_layer3()
        self.D_layer2 = D_layer2()
        self.s_discriminators = []
        for j in range(self.num_classes):
            ins = _InstanceDA()
            self.add_module('ins_{}'.format(int(j)), nn.Sequential(ins))
            self.s_discriminators.append(ins)
        # self.regD = RegDA(grad_reverse_lambda=cfg['reg_grad_reverse_lambda'])

        self.computer_da_loss = computer_da_loss(device=device,lambda1=cfg['lambda1'],lambda2=cfg['lambda2'],lambda3=cfg['lambda3'])
        self.computer_ins_loss = computer_ins_loss(device=device,num_class=num_classes)
        self.computer_reg_loss = Regularization_loss(device=device,num_class=num_classes)
        self.ins_beta = cfg['insloss_beta']
        self.da_alpha = cfg['daloss_alpha']
        self.qfl = QFLLoss_s(1.0,2.0)


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
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            ctn_pred = self.ctn_pred(reg_feat)
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

    def forward(self, x, mask=None, targets=None, tar_x=None, open_ins=False, visual=False,tar_gt=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            feats = self.backbone(x)

            source_layer4 = self.D_layer4(feats['layer4'])
            source_layer3 = self.D_layer3(feats['layer3'])
            source_layer2 = self.D_layer2(feats['layer2'])

            if tar_x is not None:
                tar_feats = self.backbone(tar_x)
                target_layer4 = self.D_layer4(tar_feats['layer4'])
                target_layer3 = self.D_layer3(tar_feats['layer3'])
                target_layer2 = self.D_layer2(tar_feats['layer2'])
                pyramid_tar_feats = [tar_feats['layer2'], tar_feats['layer3'], tar_feats['layer4']]
                pyramid_tar_feats = self.fpn(pyramid_tar_feats)
                tar_feats_iter = iter(pyramid_tar_feats)
                # if visual:
                #     vis_tsne(feats['layer4'], tar_feats['layer4'])
            else:
                target_layer4, target_layer3, target_layer2, tar_feats_iter = None, None, None, None

            # fpn neck
            pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
            pyramid_feats = self.fpn(pyramid_feats)  # [P3, P4, P5, P6, P7]

            # shared head
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_ctn_preds = []
            all_masks = []
            all_tar_cls_preds = []
            all_ins_source = []
            all_ins_target = []
            source_ins_weight = []
            target_ins_weight = []
            all_reg_source = []
            all_reg_target = []
            all_uns_cls = []
            for level, feat in enumerate(pyramid_feats):
                cls_feat, reg_feat = self.head(feat)

                # [B, C, H, W]
                cls_pred = self.cls_pred(cls_feat)
                reg_pred = self.reg_pred(reg_feat)
                ctn_pred = self.ctn_pred(reg_feat)

                # source_reg_pred = self.regD(reg_feat * ctn_pred.sigmoid())
                if level < 3:
                    ctn_p = ctn_pred.sigmoid()
                    cls_p = cls_pred.sigmoid()

                    # source_ctn_norm = (ctn_p - torch.min(ctn_p)) / (torch.max(ctn_p) - torch.min(ctn_p))
                    s_weight = F.sigmoid(ctn_p * 20 * cls_p)

                    source_ins_weight.append(s_weight)
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
                if level < 3:
                    if tar_feats_iter is not None and open_ins:
                        tar_feat = next(tar_feats_iter)
                        tar_cls_feat, tar_reg_feat = self.head(tar_feat)
                        tar_cls_pred = self.cls_pred(tar_cls_feat)
                        # tar_reg_pred = self.reg_pred(tar_reg_feat[-1])
                        tar_ctn_pred = self.ctn_pred(tar_reg_feat)
                        tar_ctn_p = tar_ctn_pred.sigmoid()
                        tar_cls_p = tar_cls_pred.sigmoid()
                        t_weight = F.sigmoid(tar_ctn_p * tar_cls_p * 20)
                        target_ins_weight.append(t_weight)
                        tar_cls_preds = tar_cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                        uns_cls_preds = tar_cls_pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
                        all_tar_cls_preds.append(tar_cls_preds)
                        all_uns_cls.append(uns_cls_preds)
                        source_list = []
                        target_list = []
                        # if visual:
                        #     vis_tsne(cls_feat, tar_cls_feat, name="ins_tsne.jpg")
                        for i in range(self.num_classes):
                            s_ins_pred = self.s_discriminators[i](feat)
                            source_list.append(s_ins_pred)
                            t_ins_pred = self.s_discriminators[i](tar_feat)
                            target_list.append(t_ins_pred)
                            # if visual:
                            #     s_attention_map = s_weight[0, i:i + 1, :, :].permute(1, 2, 0).squeeze(
                            #         dim=2)
                            #     s_attention_map = Variable(s_attention_map, requires_grad=False)
                            #     s_attention_map = np.asarray(s_attention_map.cpu())
                            #     visualize_grid_attention_v2(x[0], './visualization/source/', s_attention_map, i, level,
                            #                                 save_image=True, save_original_image=True)
                            #
                            #     t_attention_map = t_weight[0, i:i + 1, :, :].permute(1, 2, 0).squeeze(
                            #         dim=2)
                            #     t_attention_map = Variable(t_attention_map, requires_grad=False)
                            #     t_attention_map = np.asarray(t_attention_map.cpu())
                            #     visualize_grid_attention_v2(tar_x[0], './visualization/target/', t_attention_map, i,
                            #                                 level, save_image=True, save_original_image=True)

                        s_pred = torch.cat(source_list, dim=1)
                        t_pred = torch.cat(target_list, dim=1)
                        all_ins_source.append(s_pred)
                        all_ins_target.append(t_pred)
                        # all_reg_source.append(source_reg_pred)
                        # all_reg_target.append(target_reg_pred)
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                all_anchors.append(anchors)
            # for i in range(len(all_anchors)):
            #     print(all_anchors[i].shape)

            # output dict
            outputs = {"pred_cls": all_cls_preds,  # List [B, M, C]
                       "pred_reg": all_reg_preds,  # List [B, M, 4]
                       "pred_ctn": all_ctn_preds,  # List [B, M, 1]
                       'strides': self.stride,
                       "mask": all_masks}  # List [B, M,]
            # print(torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes).shape) // [30804,8]
            # loss
            loss_dict = self.criterion(outputs=outputs,
                                       targets=targets,
                                       anchors=all_anchors)
            da_loss = torch.tensor(0, dtype=torch.float)
            ins_loss = torch.tensor(0, dtype=torch.float)
            reg_loss = torch.tensor(0, dtype=torch.float)


            if tar_x is not None:
                gt_cls = tar_gt
                if gt_cls is not None:
                    cls_p = torch.cat(all_uns_cls,dim=0)
                    gt_cls = torch.cat(gt_cls,dim=0)
                    # gt_ctn = torch.cat(gt_ctn, dim=0)
                    gt_cls = gt_cls.sigmoid()
                    # gt_ctn = gt_ctn.sigmoid()
                    # print(torch.max(tar_gt))
                    uns_loss = torch.dist(cls_p.sigmoid(),gt_cls,p=2)
                    losses = loss_dict['losses'] + uns_loss
                    loss_dict.update({'losses': losses, 'uns_loss': uns_loss})

                da_loss, DA_LOSS = self.computer_da_loss([source_layer4, source_layer3, source_layer2],
                                                         [target_layer4, target_layer3, target_layer2])
                da_loss = da_loss * self.da_alpha
                losses = loss_dict['losses'] + da_loss
                loss_dict.update({'losses': losses, 'da_loss': da_loss})
                if tar_feats_iter is not None and open_ins:
                    ins_loss = self.computer_ins_loss(all_ins_source, all_ins_target, source_ins_weight,
                                                      target_ins_weight)
                    ins_loss = ins_loss * self.ins_beta
                    losses = loss_dict['losses'] + ins_loss
                    loss_dict.update({'losses': losses, 'ins_loss': ins_loss})
                    # reg_loss = self.computer_reg_loss(all_reg_source,all_reg_target)
                return loss_dict, DA_LOSS
            return loss_dict