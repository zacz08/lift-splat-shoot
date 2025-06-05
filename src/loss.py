import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets, weights):
        targets = (targets + 1) / 2  # Normalized to [0,1]
        bce_loss = self.bce(logits, targets).mean(dim=(2, 3))
        
        # apply per-layer weights
        # weighted_loss = weights[:, :, None, None] * bce_loss
        weighted_loss = (bce_loss * weights).sum(dim=1)
        return weighted_loss.mean()

class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, weights):
        logits = (logits + 1) / 2
        targets = (targets + 1) / 2 # Normalized to [0,1]
        
        intersection = (logits * targets).sum(dim=(2, 3))
        union = logits.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth) # shape: [B, C]
        
        # apply per-layer weights
        # weighted_dice_loss = weights[:, :, None, None] * (1 - dice_score)
        dice_loss = (1 - dice_score) * weights
        weighted_dice_loss = dice_loss.sum(dim=1)  # Sum over channels

        return weighted_dice_loss.mean()

class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(SegmentationLoss, self).__init__()
        self.bce = WeightedBCEWithLogitsLoss()
        self.dice = WeightedDiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, weights):
        return self.alpha * self.bce(logits, targets, weights) + \
                self.beta * self.dice(logits, targets, weights)
 

def compute_layer_weights(inputs: torch.Tensor, epsilon=1e-6, clamp_min=1e-3, clamp_max=10):
    """
    inputs: [bs, 4, H, W], mask in [-1, 1]
    returns: [4,] global rec_weight for each layer in batch level
    """
    bs, num_layers, h, w = inputs.shape
    total_pixel = bs * h * w


    mask_bin = (inputs > 0).float()  # [bs, 4, H, W]
    layer_pixel = mask_bin.sum(dim=(0, 2, 3))  # [4]
    layer_pixel = torch.clamp(layer_pixel, min=1.0)

    raw_weights = torch.log(total_pixel / layer_pixel + epsilon)
    raw_weights = raw_weights.clamp(max=clamp_max) 

    weights = raw_weights / (raw_weights.sum() + epsilon) * num_layers

    return weights  # shape: [4]


