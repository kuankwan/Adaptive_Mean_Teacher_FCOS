#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
import math

import torch
from torch import nn


class IOULoss(nn.Module):
    """
    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    TODO: Add formulation. @zhubenjin
    """
    def __init__(self, loss_type="iou"):
        """
        Args:
            loss_type (str): candidates are [iou, giou]
        """
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


def iou_loss(
    inputs,
    targets,
    weight=None,
    box_mode="xyxy",
    loss_type="iou",
    smooth=False,
    reduction="none"
):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    if smooth:
        ious = (area_intersect + 1) / (area_union + 1)
    else:
        ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious

    elif loss_type == 'siou':
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        '''
        预测框和真实框坐标形式为xyxy，即左下右上角坐标或左上右下角坐标
        '''
        b1_x1, b1_x2, b1_y1, b1_y2 = inputs[..., 0], inputs[..., 2], inputs[..., 1], inputs[..., 3]
        b2_x1, b2_x2, b2_y1, b2_y2 = targets[..., 0], targets[..., 2], targets[..., 1], targets[..., 3]
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        cw = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])  # convex (smallest enclosing box) width
        ch = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])  # convex height
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5  # 真实框和预测框中心点的宽度差
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5  # 真实框和预测框中心点的高度差
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)  # 真实框和预测框中心点的距离
        sin_alpha_1 = torch.abs(s_cw) / sigma  # 真实框和预测框中心点的夹角β
        sin_alpha_2 = torch.abs(s_ch) / sigma  # 真实框和预测框中心点的夹角α
        threshold = pow(2, 0.5) / 2  # 夹角阈值
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)  # α大于45°则考虑优化β，否则优化α
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)  # 角度损失
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)  # 距离损失
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)  # 形状损失
        iou = ious - 0.5 * (distance_cost + shape_cost)  # siou
        loss = 1.0 - iou
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss