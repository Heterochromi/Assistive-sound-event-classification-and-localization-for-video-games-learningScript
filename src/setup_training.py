import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torchvision

class SigmoidFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2, reduction: str = "mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): The raw, un-activated outputs from the model (logits).
            targets (torch.Tensor): The ground truth labels.
        Returns:
            torch.Tensor: The calculated focal loss.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability is 0
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Apply pos_weights only to positive samples
            F_loss = torch.where(targets == 1, self.alpha * F_loss, F_loss)

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def setup_training(model, learning_rate=1e-3, pos_weights=None, total_steps_scheduler=1000, UseOneCycleLR=False, use_focal_loss=False, focal_loss_gamma=2.0 , T_max=100):
    """
    Setup training components
    
    Args:
        model: MultiLabelCCT model
        learning_rate: Learning rate for optimizer
        pos_weights: Optional tensor for class balancing (used as alpha in FocalLoss)
        UseOneCycleLR: Whether to use OneCycleLR scheduler
        use_focal_loss: Whether to use FocalLoss instead of BCEWithLogitsLoss
        focal_loss_gamma: Gamma parameter for FocalLoss
    """
    # Loss function selection
    if use_focal_loss:
        criterion = SigmoidFocalLoss(alpha=pos_weights, gamma=focal_loss_gamma, reduction='mean')
    elif pos_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    if UseOneCycleLR:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
    max_lr=learning_rate,               # Peak LR (safe for CCT)
    total_steps=total_steps_scheduler,
    pct_start=0.3,             # 30% of steps for warm-up
    anneal_strategy='cos',     # Cosine decay
    div_factor=10,             # 1e-4 -> 1e-3 peak? No, here 1e-4 * 10 = 1e-3 max? Actually: start = max_lr/10
    final_div_factor=1000      # Ends at max_lr / 1000
     )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return criterion, optimizer, scheduler