import os
import numpy as np
import torch
from glob import glob

def compute_pos_weight(mask_dir, num_classes=6):
    """
    计算每个语义通道的 pos_weight = #negative / #positive
    mask_dir: 包含 .npy 的目录路径
    num_classes: 每个掩码中的通道数（类别数）
    """
    total_pos = torch.zeros(num_classes)
    total_neg = torch.zeros(num_classes)

    npy_files = sorted(glob(os.path.join(mask_dir, "*.npy")))
    assert len(npy_files) > 0, f"No .npy files found in {mask_dir}"

    for path in npy_files:
        mask = np.load(path)  # shape: [C, H, W]
        assert mask.shape[0] == num_classes, f"Unexpected channel count in {path}"

        mask = torch.from_numpy(mask).float()
        total_pos += (mask == 1).sum(dim=[1, 2])
        total_neg += (mask == 0).sum(dim=[1, 2])

    pos_weight = total_neg / (total_pos + 1e-6)  # 加 epsilon 防止除0
    return pos_weight

# 用法示例
if __name__ == "__main__":
    mask_root = "./data/nuscenes/bev_seg_gt_mask_200/mini_train"  
    pos_weight = compute_pos_weight(mask_root, num_classes=6)
    rounded_pos_weight = [round(p.item(), 2) for p in pos_weight]
    normalized = [round(w.item() / max(pos_weight).item(), 5) for w in pos_weight]
    print("pos_weight =", rounded_pos_weight)
    print("normalized =", normalized)

