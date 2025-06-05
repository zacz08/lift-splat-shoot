import os
import torch
import torchvision
import datetime
import numpy as np
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        now = datetime.datetime.now()
        self.log_folder_tag = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute)))
        self.val_sample_input = None    # to save sample during validation

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split + "_" + self.log_folder_tag)
        processed_images = []
        spacing = 8
        for k in images:
            if isinstance(images[k], np.ndarray):
                images[k] = torch.from_numpy(images[k])
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            # grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            # grid = grid.numpy()
            # grid = (grid * 255).astype(np.uint8)
            grid = (grid > 0).numpy().astype(np.uint8)
            processed_images.append(grid)

        if len(processed_images) > 0:

            # add blank gap between images
            images_with_spacing = []
            for img in processed_images:
                # img = np.transpose(img, (2, 0, 1))  # [512, 512, 4] -> [4, 512, 512]
                horizontal_mask = np.concatenate(img, axis=1)   # [4, 512, 512] -> [512, 2048]
                rgb_mask = np.stack([horizontal_mask] * 3, axis=2)  # [512, 2048] -> [512, 2048, 3] to show in RGB
                img = Image.fromarray(rgb_mask * 255)  # convert to PIL Image, rescale to [0, 255]
                images_with_spacing.append(img)
                _, img_width = horizontal_mask.shape
                blank_image = np.ones((spacing, img_width, 3), dtype=np.uint8) * 255  # blank gap
                images_with_spacing.append(blank_image)
            # remove last blank gap of the last image
            images_with_spacing = images_with_spacing[:-1]

            stacked_image = np.vstack(images_with_spacing)  # stack vertically

            # save the combined image
            filename = "combined_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(stacked_image).save(path)


    def log_local_rgb(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        """
        Log images as RGB masks, combining multiple masks into a single image.
        """
        root = os.path.join(save_dir, "image_log", split + "_" + self.log_folder_tag)
        os.makedirs(root, exist_ok=True)

        spacing = 8
        processed_images = []

        for k in images:
            masks = images[k]
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)

            # Ensure shape is [B, 4, H, W]
            if masks.ndim == 3:
                masks = masks.unsqueeze(0)  # [4, H, W] -> [1, 4, H, W]
            elif masks.ndim != 4:
                raise ValueError(f"Unexpected mask shape: {masks.shape}")

            for i in range(masks.shape[0]):
                mask4 = masks[i].cpu().numpy()  # [4, H, W]
                rgb_img = combine_masks_to_rgb(mask4)  # [H, W, 3], float32 in [0,1]
                rgb_img = (rgb_img * 255).astype(np.uint8)
                processed_images.append(rgb_img)

        if len(processed_images) == 0:
            return

        # Add blank gap between images
        images_with_spacing = []
        for img in processed_images:
            images_with_spacing.append(img)
            blank_gap = np.ones((img.shape[0], spacing, 3), dtype=np.uint8) * 255
            images_with_spacing.append(blank_gap)
        images_with_spacing = images_with_spacing[:-1]  # Remove last blank

        stacked_image = np.concatenate(images_with_spacing, axis=1)  # stack horizontally

        filename = "combined_rgb_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        Image.fromarray(stacked_image).save(path)


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            ## two log styles for ldm and cldm
            if hasattr(pl_module, "control_key"):   # for cldm
                self.log_local(pl_module.logger.save_dir, split, images, 
                               pl_module.global_step, pl_module.current_epoch, batch_idx)
            else:   # for ldm
                self.log_local_rgb(pl_module.logger.save_dir, split, images,
                                   pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     if not self.disabled:
    #         self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="val")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="predict")



def combine_masks_to_rgb(mask):
    h, w = mask.shape[1:]
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    colors = [
    [0, 128, 255],      # drivable_area - 鲜艳的蓝色
    [255, 255, 0],      # lane_divider - 高亮的黄色
    [255, 0, 0],        # vehicle - 鲜艳的红色
    [0, 255, 0],        # pedestrian - 鲜艳的绿色
    [255, 0, 255],      # other1 - 紫色
    [255, 165, 0],      # other2 - 橙色
]

    for i in range(mask.shape[0]):
        mask_i = (mask[i] > 0)  # binary mask: shape [H, W]
        color = np.array(colors[i], dtype=np.uint8).reshape(1, 1, 3)  # shape [1, 1, 3]
        rgb_img[mask_i] = color  # overwrite only where mask==1

    rgb_img = rgb_img.astype(np.float32) / 255.0  # normalize for matplotlib
    return rgb_img