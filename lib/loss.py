import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-8):
        super(ClassWeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds, binary_labels):
        p = torch.sigmoid(preds)
        p_t = p * binary_labels + (1 - p) * (1 - binary_labels)

        ce_loss = F.binary_cross_entropy_with_logits(preds, binary_labels, reduction='none')
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)

        class_weight_pos = self.alpha
        class_weight_neg = 1 - self.alpha
        alpha_t = class_weight_pos * binary_labels + class_weight_neg * (1 - binary_labels)
        cls_loss = alpha_t * focal_loss

        combined_loss = cls_loss
        return combined_loss.mean()