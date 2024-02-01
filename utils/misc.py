import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_loss(inputs, target, beta=1, smooth = 1e-3,weight1=None,weight2=None):
    pos = target > 0.
    tp = torch.sum(inputs[pos])
    fp = torch.sum(inputs)
    fn = torch.sum(target)
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def kl_divergence(p,q):
    return torch.sum(torch.where(p != 0,p * torch.log(p/q),0))


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def condition_focal_loss(logits, targets, gamma=1.0, reduction='none', weight=torch.tensor(0.0)):
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                 target=targets,
                                                 reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)

    loss = ce_loss * ((1.0 - p_t) ** gamma)


    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='none'):
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                 target=targets,
                                                 reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss


def QFLv2(pred_sigmoid,          # (n, 80)
          teacher_sigmoid,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss

def varifocal_loss(pred,
                   target,
                   weight=None,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   reduction='mean',
                   avg_factor=None):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.05).float() + \
                       alpha * (pred_sigmoid - target).abs().pow(gamma) * \
                       (target <= 0.05).float()
    else:
        focal_weight = (target > 0.05).float() + \
                       alpha * (pred_sigmoid - target).abs().pow(gamma) * \
                       (target <= 0.05).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def sigmoid_varifocal_loss(logits, targets,thred=0.05, alpha=0.75, gamma=2.0, reduction='none'):
    p = logits
    pos = (targets > thred).float()
    target = torch.where(targets<=thred,targets,pos)
    ce_loss = F.binary_cross_entropy(input=logits,target=pos,reduction="none")
    scale_factor = (p - pos).abs().pow(gamma)
    loss = ce_loss * scale_factor
    # pos_mask = (targets > 0.05).float()
    # neg_mask = (targets == 0.05).float()
    # pos_loss = ce_loss * pos_mask
    # neg_loss = ce_loss * neg_mask
    #
    # loss = targets * pos_loss + alpha * ((p - targets).abs().pow(gamma)) * neg_loss

    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss


class QFLLoss(nn.Module):
    def __init__(self, qfl_loss_weight=1.0, qfl_loss_beta=2.0):
        super(QFLLoss, self).__init__()
        self.weight = qfl_loss_weight
        self.beta = qfl_loss_beta

    def forward(self, preds, targets, avg_factor=None, reduction='none'):
        loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        preds_sigmoid = preds.sigmoid()
        scale_factor = (preds_sigmoid - targets).abs().pow(self.beta)
        loss = loss * scale_factor
        # loss=loss.sum()*self.weight
        if (not avg_factor is None):
            return loss / avg_factor
        if reduction == "mean":
            loss = loss.mean()

        elif reduction == "sum":
            loss = loss.sum()
        return loss


class DFLLoss(torch.nn.Module):
    def __init__(self, dfl_loss_weight):
        super(DFLLoss, self).__init__()
        self.weight = dfl_loss_weight

    def forward(self, preds, targets, weights=None, avg_factor=None):
        dis_left = targets.long()
        dis_right = dis_left + 1
        weight_left = dis_right.float() - targets
        weight_right = targets - dis_left.float()
        loss = F.cross_entropy(preds, dis_left, reduction='none') * weight_left + F.cross_entropy(preds, dis_right,
                                                                                                  reduction='none') * weight_right
        if (not weights is None):
            loss = loss * weights
        loss = loss.sum() / 4
        if (not avg_factor is None):
            return (loss / avg_factor) * self.weight
        return loss * self.weight

def infoNCE_loss(x,y,sigma=1.,temperature=0.3):
    # 首先，需要对 x 和 y 进行范数计算
    # 这里使用的是 L2 范数，也可以使用其他范数
    x = x.sigmoid()
    y = y.sigmoid()
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)

    # 然后计算 x 和 y 之间的内积
    # 并使用 softmax 函数将结果转化为概率分布
    x_y = torch.matmul(x, y.t()/temperature)
    x_y_prob = nn.functional.softmax(x_y, dim=1)

    # 最后，计算 InfoNCE loss
    # 其中 sigma 是超参数，常常被设置为 1
    # loss = torch.mean(-torch.log(x_y_prob) + sigma * (x_norm + y_norm) ** 2)
    loss = torch.mean(-torch.log(x_y_prob) + sigma * (x_norm + y_norm) ** 2)
    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class NT_Xent(nn.Module):
    def __init__(self, pixel_size, temperature):
        super(NT_Xent, self).__init__()
        self.pixel_size = pixel_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(pixel_size * 2, pixel_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=0)
        z_j = F.normalize(emb_j, dim=0)

        representations = torch.cat([z_i, z_j], dim=1)
        similarity_matrix = F.cosine_similarity(representations.transpose(0,1).unsqueeze(1), representations.transpose(0,1).unsqueeze(0), dim=-1)

        sim_ij = torch.diag(similarity_matrix, self.pixel_size)
        sim_ji = torch.diag(similarity_matrix, -self.pixel_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.pixel_size)
        return loss

class shared_loss(nn.Module):
    def __init__(self,):
        super(shared_loss, self).__init__()

    def forward(self, x, y):
        return torch.pow(F.normalize(x, 2, dim=1) - F.normalize(y, 2, dim=1),2).mean()

class cos_loss(nn.Module):
    def __init__(self,):
        super(cos_loss, self).__init__()

    def forward(self, x, y,p=None):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        similarity_matrix = torch.cosine_similarity(x,y,dim=1)
        cos_loss = (1-similarity_matrix)
        if p is not None:
            cos_loss = p.squeeze(1) * cos_loss
        return cos_loss.mean()


def nms(dets, scores, nms_thresh=0.4):
    """"Pure Python NMS baseline."""
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
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def load_weight(device, model, path_to_ckpt):
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict, strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    return model


class CollateFunc(object):
    def __call__(self, batch):
        targets = []
        images = []
        weak_images = []
        masks = []
        for sample in batch:
            image = sample[0]
            weak_image = sample[1]
            target = sample[2]
            mask = sample[3]

            images.append(image)
            targets.append(target)
            masks.append(mask)
            weak_images.append(weak_image)

        images = torch.stack(images, 0)  # [B, C, H, W]
        weak_images = torch.stack(weak_images,0)
        masks = torch.stack(masks, 0)  # [B, H, W]

        return images,weak_images, targets, masks


class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


# test time augmentation(TTA)
class TestTimeAugmentation(object):
    def __init__(self, num_classes=80, nms_thresh=0.4, scale_range=[320, 640, 32]):
        self.nms = nms
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.scales = np.arange(scale_range[0], scale_range[1] + 1, scale_range[2])

    def __call__(self, x, model):
        # x: Tensor -> [B, C, H, W]
        bboxes_list = []
        scores_list = []
        labels_list = []

        # multi scale
        for s in self.scales:
            if x.size(-1) == s and x.size(-2) == s:
                x_scale = x
            else:
                x_scale = torch.nn.functional.interpolate(
                    input=x,
                    size=(s, s),
                    mode='bilinear',
                    align_corners=False)
            model.set_grid(s)
            bboxes, scores, labels = model(x_scale)
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

            # Flip
            x_flip = torch.flip(x_scale, [-1])
            bboxes, scores, labels = model(x_flip)
            bboxes = bboxes.copy()
            bboxes[:, 0::2] = 1.0 - bboxes[:, 2::-2]
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

        bboxes = np.concatenate(bboxes_list)
        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels
