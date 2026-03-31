import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets, weights, mask_valid=None):
        """
        logits: [B, C, H, W]
        targets: [B, C, H, W] in [0, 1]
        weights: [C]
        mask_valid: [B, 1, H, W] (optional)
        """
        # targets already in [0, 1], no conversion needed
        # bce_loss = self.bce(logits, targets).mean(dim=(2, 3))
        bce_loss = self.bce(logits, targets)  # shape: [B, C, H, W]

        if mask_valid is not None:
            mask = mask_valid.expand_as(bce_loss)  # shape: [B, C, H, W]
            bce_loss = bce_loss * mask  # mask-out invalid pixels
            denom = mask.sum(dim=(2, 3)) + 1e-6
        else:
            denom = logits.shape[2] * logits.shape[3]

        bce_loss = bce_loss.sum(dim=(2, 3)) / denom  # [B, C]
        
        # apply per-layer weights
        weighted_loss = (bce_loss * weights).sum(dim=1)
        return weighted_loss.mean()

class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, weights, mask_valid=None):
        """
        logits: [B, C, H, W]
        targets: [B, C, H, W] in [0, 1]
        weights: [C]
        mask_valid: [B, 1, H, W] (optional)
        """
        probs = torch.sigmoid(logits)
        # targets already in [0, 1], no conversion needed

        if mask_valid is not None:
            mask = mask_valid.expand_as(probs)
            probs = probs * mask
            targets = targets * mask

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth) # shape: [B, C]
        
        # apply per-layer weights
        # weighted_dice_loss = weights[:, :, None, None] * (1 - dice_score)
        dice_loss = (1 - dice_score) * weights
        weighted_dice_loss = dice_loss.sum(dim=1)  # Sum over channels

        return weighted_dice_loss.mean()

class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """
        Computes gradient of the Jaccard loss w.r.t the sorted error
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
        logits: [P] Variable, logits at each pixel (between -infty and +infty)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
            
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def forward(self, logits, targets, mask_valid=None):
        """
        logits: [B, C, H, W]
        targets: [B, C, H, W] in [0, 1]
        mask_valid: [B, 1, H, W] (optional)
        """
        # targets already in [0, 1], no conversion needed 
        
        # 2. Iterate over each channel (class) separately
        # Lovasz-Hinge is designed for binary classification, so we apply it per-class
        total_loss = 0
        batch_size, C, H, W = logits.shape
        
        for c in range(C):
            # Extract single channel
            c_logits = logits[:, c, :, :] # [B, H, W]
            c_targets = targets[:, c, :, :] # [B, H, W]
            
            # Apply Mask if exists
            if mask_valid is not None:
                # Expand mask to match single channel dims [B, H, W]
                mask = mask_valid.squeeze(1) 
                
                # Select only valid pixels
                # This flattens the tensors automatically
                c_logits_flat = c_logits[mask.bool()]
                c_targets_flat = c_targets[mask.bool()]
            else:
                # Flatten manually if no mask
                c_logits_flat = c_logits.flatten()
                c_targets_flat = c_targets.flatten()
            
            # 3. Compute Lovasz-Hinge for this class
            class_loss = self.lovasz_hinge_flat(c_logits_flat, c_targets_flat)
            
            total_loss += class_loss

        # Average over channels (or sum, depending on preference, but mean keeps scale stable)
        return total_loss / C

class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(SegmentationLoss, self).__init__()
        self.bce = WeightedBCEWithLogitsLoss()
        self.dice = WeightedDiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, weights, mask_valid=None):
        """
        logits: [B, C, H, W]
        targets: [B, C, H, W]
        weights: [C]
        mask_valid: [B, 1, H, W] (optional)
        """
        return self.alpha * self.bce(logits, targets, weights, mask_valid) + \
                self.beta * self.dice(logits, targets, weights, mask_valid)
 

def compute_layer_weights(inputs: torch.Tensor, epsilon=1e-6, clamp_min=1e-3, clamp_max=10):
    """
    inputs: [bs, C, H, W], mask in [0, 1]
    returns: [C,] global rec_weight for each layer in batch level
    """
    bs, num_layers, h, w = inputs.shape
    total_pixel = bs * h * w

    mask_bin = (inputs > 0.5).float()  # [bs, C, H, W], binarize at 0.5
    layer_pixel = mask_bin.sum(dim=(0, 2, 3))  # [C]
    layer_pixel = torch.clamp(layer_pixel, min=1.0)

    raw_weights = torch.log(total_pixel / layer_pixel + epsilon)
    raw_weights = raw_weights.clamp(max=clamp_max) 

    weights = raw_weights / (raw_weights.sum() + epsilon) * num_layers

    return weights  # shape: [4]