import torch
from torch import nn

from models.triplet_loss import HardTripletLoss


class LossFun(nn.Module):
    """Combined regression, consistency, and triplet loss."""

    def __init__(self, alpha_mse, alpha, margin, consistency_weight=1.0):
        super(LossFun, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha
        self.alpha_mse = alpha_mse
        self.consistency_weight = consistency_weight

    def forward(self, pred, label, pred1, pred2, feat, args):
        # Main regression loss plus consistency regularization between heads.
        mse_loss = self.mse_loss(pred, label)
        loss_consistency = self.mse_loss(pred1, pred2)

        # Apply triplet loss only when multiple fusion tokens are available.
        if feat is not None and feat.size(1) > 1:
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.reshape(-1, c)
            token_labels = torch.arange(n, device=device).repeat(b)
            t_loss = self.triplet_loss(flat_feat, token_labels)
        else:
            t_loss = torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.alpha_mse * mse_loss
            + self.alpha * t_loss
            + self.consistency_weight * loss_consistency
        )
        return total_loss
