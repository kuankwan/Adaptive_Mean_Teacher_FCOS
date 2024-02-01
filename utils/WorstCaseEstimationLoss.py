import torch
import torch.nn as nn
import torch.nn.functional as F
def shift_log(x, offset=1e-6):
    """
    First shift, then calculate log for numerical stability.
    """

    return torch.log(torch.clamp(x + offset, max=1.))

class WorstCaseEstimationLoss(nn.Module):
    r"""
    Worst-case Estimation loss from `Debiased Self-Training for Semi-Supervised Learning <https://arxiv.org/abs/2202.07136>`_
    that forces the worst possible head :math:`h_{\text{worst}}` to predict correctly on all labeled samples
    :math:`\mathcal{L}` while making as many mistakes as possible on unlabeled data :math:`\mathcal{U}`. In the
    classification task, it is defined as:
    .. math::
        loss(\mathcal{L}, \mathcal{U}) =
        \eta' \mathbb{E}_{y^l, y_{adv}^l \sim\hat{\mathcal{L}}} -\log\left(\frac{\exp(y_{adv}^l[h_{y^l}])}{\sum_j \exp(y_{adv}^l[j])}\right) +
        \mathbb{E}_{y^u, y_{adv}^u \sim\hat{\mathcal{U}}} -\log\left(1-\frac{\exp(y_{adv}^u[h_{y^u}])}{\sum_j \exp(y_{adv}^u[j])}\right),
    where :math:`y^l` and :math:`y^u` are logits output by the main head :math:`h` on labeled data and unlabeled data,
    respectively. :math:`y_{adv}^l` and :math:`y_{adv}^u` are logits output by the worst-case estimation
    head :math:`h_{\text{worst}}`. :math:`h_y` refers to the predicted label when the logits output is :math:`y`.
    Args:
        eta_prime (float): the trade-off hyper parameter :math:`\eta'`.
    Inputs:
        - y_l: logits output :math:`y^l` by the main head on labeled data
        - y_l_adv: logits output :math:`y^l_{adv}` by the worst-case estimation head on labeled data
        - y_u: logits output :math:`y^u` by the main head on unlabeled data
        - y_u_adv: logits output :math:`y^u_{adv}` by the worst-case estimation head on unlabeled data
    Shape:
        - Inputs: :math:`(minibatch, C)` where C denotes the number of classes.
        - Output: scalar.
    """

    def __init__(self, eta_prime):
        super(WorstCaseEstimationLoss, self).__init__()
        self.eta_prime = eta_prime


    def forward(self, y_l, y_l_adv, y_u, y_u_adv):
        y_l_weight, prediction_l = y_l.max(dim=1)
        loss_l = self.eta_prime * F.cross_entropy(y_l_adv, prediction_l)


        y_u_weight, prediction_u = y_u.max(dim=1)
        loss_u = F.nll_loss(shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u)

        return loss_l + loss_u