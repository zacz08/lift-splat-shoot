import torch
import os
import csv
import time
from torch import nn
from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl
from torchvision.models.resnet import resnet18
from src.log_analysis import parse_csv_and_plot
from src.metrics import IntersectionOverUnion
from src.loss import compute_layer_weights

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum, SimpleLoss, get_batch_iou


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.grid_conf = cfg.grid_conf
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.data_aug_conf = cfg.data_aug
        self.downsample = cfg.model.downsample
        self.camC = cfg.model.camC
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=cfg.model.outC) # outC: number of classes/semantic layers

        self.use_quickcumsum = True

        self.loss_fn = SimpleLoss()
        self.lr = cfg.optim.lr
        self.weight_decay = cfg.optim.weight_decay
        self.max_grad_norm = cfg.optim.max_grad_norm

        self.seg_metric = IntersectionOverUnion(cfg.model.outC).to(self.device)

        self.inference_times = []
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, return_feats=False):
        bev_feat = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(bev_feat)
        if return_feats:
            return x, bev_feat
        else:
            return x
    
    def training_step(self, batch, batch_idx):
        imgs, rots, trans, intrins, post_rots, post_trans, bev_seg_gt, _ = batch
        preds = self(imgs, rots, trans, intrins, post_rots, post_trans)
        # rec_weight = compute_layer_weights(bev_seg_gt)
        rec_weight = torch.tensor([0.0671, 1.7914, 0.2680, 1.3406, 1.6461, 0.8869], 
                                  dtype=torch.float32, device=preds.device)
        loss = self.loss_fn(preds, bev_seg_gt, rec_weight)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, rots, trans, intrins, post_rots, post_trans, bev_seg_gt, _ = batch
        preds = self(imgs, rots, trans, intrins, post_rots, post_trans)
        # rec_weight = compute_layer_weights(bev_seg_gt)
        rec_weight = torch.tensor([0.0671, 1.7914, 0.2680, 1.3406, 1.6461, 0.8869], 
                                  dtype=torch.float32, device=preds.device)
        loss = self.loss_fn(preds, bev_seg_gt, rec_weight)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.seg_metric.update((preds > 0), bev_seg_gt)

    def on_validation_epoch_end(self):
        score = self.seg_metric.compute()
        iou= score.mean().item()
        log_dict = {'val/IoU': iou}
        self.log_dict(log_dict, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_index):
        start = time.time()

        imgs, rots, trans, intrins, post_rots, post_trans, bev_seg_gt, _ = batch
        preds = self(imgs, rots, trans, intrins, post_rots, post_trans)
        self.seg_metric.update((preds > 0), bev_seg_gt)

        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        self.inference_times.append(elapsed)

    @torch.no_grad()
    def on_predict_epoch_end(self):
        score = self.seg_metric.compute()
        for index, layer in enumerate(self.cfg.dataset.semantic_layer):
            print(f"IoU {layer}: {score[index].item():.5f}")

        total_time = sum(self.inference_times)
        num_samples = len(self.inference_times)
        avg_time = total_time / num_samples
        fps = 1.0 / avg_time
        print(f"\n✅ Average time per sample = {avg_time:.4f} s, FPS = {fps:.2f}")



    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, **kwargs):
        imgs, rots, trans, intrins, post_rots, post_trans, bev_seg_gt, _ = batch
        preds = self(imgs, rots, trans, intrins, post_rots, post_trans)
        log = dict()
        log["prediction"] = preds
        log["groundtruth"] = bev_seg_gt   
        return log
    
    @torch.no_grad()
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics  # All logged metrics
        epoch = int(self.current_epoch)

        fields = [
            "train/loss_epoch",
            "val/loss_epoch",
            "val/IoU",
        ]

        row = {"epoch": epoch}
        for key in fields:
            val = metrics.get(key)
            row[key] = val.item() if val is not None else None

        # Write to CSV
        csv_file = os.path.join(self.logger.log_dir, "train_log.csv")
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: ("" if v is None else f"{v:.3f}") for k, v in row.items()})

        # Draw learning curve
        learning_curve = os.path.join(self.logger.log_dir, "loss_plot.png")
        parse_csv_and_plot(csv_file, learning_curve)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def on_before_zero_grad(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
