import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal

from src.networks.fond import FONDBase


class BlvLoss(nn.Module):
    # cls_nufrequency_list
    def __init__(self, cls_num_list, sigma=4, loss_name="BlvLoss"):
        super(BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        cls_num_list_sum = torch.sum(cls_list)
        self.frequency_list = torch.log(cls_num_list_sum) - frequency_list
        self.reduction = "mean"
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name

    def reduce_loss(self, loss, reduction):
        """Reduce loss as specified.

        Args:
            loss (Tensor): Elementwise loss tensor.
            reduction (str): Options are "none", "mean" and "sum".

        Return:
            Tensor: Reduced loss tensor.
        """
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()

    def weight_reduce_loss(self, loss, weight=None, reduction="mean", avg_factor=None):
        """Apply element-wise weight and reduce loss.

        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Average factor when computing the mean of losses.

        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            assert weight.dim() == loss.dim()
            if weight.dim() > 1:
                assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            loss = self.reduce_loss(loss, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == "mean":
                # Avoid causing ZeroDivisionError when avg_factor is 0.0,
                # i.e., all labels of an image belong to ignore index.
                eps = torch.finfo(torch.float32).eps
                loss = loss.sum() / (avg_factor + eps)
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != "none":
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def forward(
        self,
        pred,
        target,
        weight=None,
        ignore_index=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """
        NOTE: set weight and ignore_index to None, based on my understanding
        of initial paper, might need to revisit implementation again
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        variation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        pred = pred + (
            variation.abs() / self.frequency_list.max() * self.frequency_list
        )

        loss = F.cross_entropy(
            pred,
            target,
            reduction="none",
            # ignore_index=ignore_index
        )

        if weight is not None:
            weight = weight.float()

        loss = self.weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor
        )

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class FOND_BLV(FONDBase):
    """
    Based on FOND however we use the BLV loss instead of regular cross-entropy
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, class_counts):
        super(FOND_BLV, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.blv_loss = BlvLoss(class_counts)

    def update(self, minibatches, unlabeled=None):
        values = self.preprocess(minibatches)

        domains = torch.cat(values["domains"])
        targets = torch.cat(values["targets"])
        classifs = torch.cat(values["classifs"])
        projections = torch.cat(values["projections"])

        masks = self.get_masks(Y=targets, D=domains)

        alpha = (
            ~masks["diff_domain_same_class_mask"]
            + masks["diff_domain_same_class_mask"] * self.xda_alpha
        )

        beta = (
            ~masks["same_domain_diff_class_mask"]
            + masks["same_domain_diff_class_mask"] * self.xda_beta
        )

        xdom_loss, mean_positives_per_sample, num_zero_positives = self.supcon_loss(
            projections=projections,
            positive_mask=masks["same_class_mask"] * masks["self_mask"],
            negative_mask=masks["self_mask"],
            alpha=alpha,
            beta=beta,
        )

        oc_class_loss = F.cross_entropy(
            classifs, targets, weight=self.oc_weight.to(targets.device)
        )
        noc_class_loss = F.cross_entropy(
            classifs, targets, weight=self.noc_weight.to(targets.device)
        )
        error_loss = oc_class_loss - noc_class_loss
        if torch.isnan(error_loss):
            error_loss = torch.tensor(0).to(targets.device)

        blv_loss = self.blv_loss.forward(classifs, targets)

        loss = (
            blv_loss
            + self.xdom_lmbd * xdom_loss
            + self.error_lmbd * torch.abs(error_loss)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "blv_loss": blv_loss.item(),
            "xdom_loss": xdom_loss.item(),
            "error_loss": error_loss.item(),
            "mean_p": mean_positives_per_sample.item(),
            "zero_p": num_zero_positives.item(),
        }
