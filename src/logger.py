import os
import torch
import numpy as np
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def combine_masks_to_rgb(mask):
    h, w = mask.shape[1:]
    background_color = np.array([30, 30, 30], dtype=np.uint8)
    rgb_img = np.ones((h, w, 3), dtype=np.uint8) * background_color.reshape(1, 1, 3)

    colors = [
    (100, 100, 100),      # drivable_area - 柏油灰
    (255, 178, 102),      # ped_crossing - 橙黄
    (220, 220, 100),      # walkway - 浅黄
    (255, 50, 50),        # stop_line - 鲜红
    (160, 120, 60),       # carpark_area - 棕褐色
    (240, 240, 240),      # lane_divider - 白色
]

    for i in range(mask.shape[0]):
        mask_i = (mask[i] > 0)  # binary mask: shape [H, W]
        color = np.array(colors[i], dtype=np.uint8).reshape(1, 1, 3)  # shape [1, 1, 3]
        rgb_img[mask_i] = color  # overwrite only where mask==1

    rgb_img = rgb_img.astype(np.float32) / 255.0  # normalize for matplotlib
    return rgb_img


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, image_style='normal', data_split='val', 
                 log_images_kwargs=None, log_folder=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.save_dir = log_folder

        assert image_style in ['normal', 'video_frame'], \
            f"Invalid image_style: {image_style}. Choose from ['normal', 'video_frame']."
        self.image_style = image_style
        if self.image_style == 'video_frame':
            assert data_split in ['train','val','test','mini_train','mini_val'], \
            f"Invalid data_split: {data_split}"
            if 'mini' in data_split:
                ds_version = 'v1.0-mini'
            else:
                ds_version = 'v1.0-trainval'
            # Load nuScenes dataset
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(version=ds_version, dataroot='./data/nuscenes', verbose=False)

    @staticmethod
    def _as_bchw(tensor_like):
        if isinstance(tensor_like, np.ndarray):
            tensor_like = torch.from_numpy(tensor_like)

        if tensor_like.ndim == 2:
            tensor_like = tensor_like.unsqueeze(0).unsqueeze(0)
        elif tensor_like.ndim == 3:
            tensor_like = tensor_like.unsqueeze(0)
        elif tensor_like.ndim != 4:
            raise ValueError(f"Unexpected tensor shape: {tensor_like.shape}")

        return tensor_like

    @staticmethod
    def _heatmap_to_rgb(heatmap_hw):
        heatmap_hw = np.nan_to_num(heatmap_hw.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        heatmap_hw = np.clip(heatmap_hw, 0.0, 1.0)
        colored = cm.get_cmap('turbo')(heatmap_hw)[..., :3]
        return (colored * 255).astype(np.uint8)

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx, render=True):
        root = os.path.join(self.save_dir, "image_log_" + split)
        processed_images = []
        spacing = 8

        for k in images:
            if isinstance(images[k], np.ndarray):
                images[k] = torch.from_numpy(images[k])

            masks = images[k]  # tensor of shape [B, C, H, W] or [C, H, W]
            if masks.ndim == 3:
                masks = masks.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

            B, C, H, W = masks.shape
            row_images = []

            for c in range(C):
                for b in range(B):
                    bin_mask = (masks[b, c] > 0).cpu().numpy().astype(np.uint8)
                    rgb_mask = np.stack([bin_mask] * 3, axis=2) * 255  # [H, W, 3]
                    row_images.append(Image.fromarray(rgb_mask))

            if render:
                for b in range(B):
                    rgb_combined = combine_masks_to_rgb(masks[b].cpu().numpy())  # [H, W, 3]
                    rgb_combined = (rgb_combined * 255).astype(np.uint8)
                    row_images.append(Image.fromarray(rgb_combined))

            # combine row_image and rgb_image horizontally
            total_width = sum(im.width for im in row_images) + spacing * (len(row_images) - 1)
            row_height = row_images[0].height
            row_image = Image.new("RGB", (total_width, row_height), (255, 255, 255))

            x_offset = 0
            for idx, im in enumerate(row_images):
                row_image.paste(im, (x_offset, 0))
                x_offset += im.width
                if idx < len(row_images) - 1:
                    x_offset += spacing  # add spacing between images

            processed_images.append(row_image)

        if len(processed_images) == 0:
            return

        # Add vertical spacing between rows
        images_with_spacing = []
        for img in processed_images:
            images_with_spacing.append(img)
            blank_image = Image.new("RGB", (img.width, spacing), (255, 255, 255))
            images_with_spacing.append(blank_image)
        images_with_spacing = images_with_spacing[:-1]  # Remove last blank

        # Stack vertically
        total_height = sum(img.height for img in images_with_spacing)
        max_width = max(img.width for img in images_with_spacing)
        stacked_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

        y_offset = 0
        for img in images_with_spacing:
            stacked_image.paste(img, (0, y_offset))
            y_offset += img.height

        # Save image
        filename = "combined_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stacked_image.save(path)


    def log_local_rgb(self, split, images, global_step, current_epoch, batch_idx):
        """
        Log images as RGB masks, combining multiple masks into a single image.
        """
        root = os.path.join(self.save_dir, "image_log_" + split)
        os.makedirs(root, exist_ok=True)

        spacing = 8
        required_keys = ["prediction", "evidential_heatmap", "groundtruth"]

        # Preferred evidential layout: each sample as one row [pred | heatmap | gt], rows stacked vertically.
        if all(k in images for k in required_keys):
            pred_masks = self._as_bchw(images["prediction"]).detach().cpu()
            heatmaps = self._as_bchw(images["evidential_heatmap"]).detach().cpu()
            gt_masks = self._as_bchw(images["groundtruth"]).detach().cpu()

            batch_size = min(pred_masks.shape[0], heatmaps.shape[0], gt_masks.shape[0], self.max_images)
            if batch_size == 0:
                return

            row_images = []
            for i in range(batch_size):
                pred_rgb = (combine_masks_to_rgb(pred_masks[i].numpy()) * 255).astype(np.uint8)

                heat_i = heatmaps[i, 0].numpy() if heatmaps.shape[1] > 0 else heatmaps[i].squeeze(0).numpy()
                heat_rgb = self._heatmap_to_rgb(heat_i)

                gt_rgb = (combine_masks_to_rgb(gt_masks[i].numpy()) * 255).astype(np.uint8)

                panel_imgs = [pred_rgb, heat_rgb, gt_rgb]
                panel_with_spacing = []
                for panel in panel_imgs:
                    panel_with_spacing.append(panel)
                    panel_with_spacing.append(np.ones((panel.shape[0], spacing, 3), dtype=np.uint8) * 255)
                panel_with_spacing = panel_with_spacing[:-1]

                row = np.concatenate(panel_with_spacing, axis=1)
                row_images.append(row)

            stacked_rows = []
            for row in row_images:
                stacked_rows.append(row)
                stacked_rows.append(np.ones((spacing, row.shape[1], 3), dtype=np.uint8) * 255)
            stacked_rows = stacked_rows[:-1]

            stacked_image = np.concatenate(stacked_rows, axis=0)

            filename = "combined_rgb_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            Image.fromarray(stacked_image).save(path)
            return

        # Fallback to legacy behavior if expected keys are absent.
        processed_images = []
        for k in images:
            masks = self._as_bchw(images[k]).detach().cpu()
            for i in range(masks.shape[0]):
                rgb_img = combine_masks_to_rgb(masks[i].numpy())
                rgb_img = (rgb_img * 255).astype(np.uint8)
                processed_images.append(rgb_img)

        if len(processed_images) == 0:
            return

        images_with_spacing = []
        for img in processed_images:
            images_with_spacing.append(img)
            images_with_spacing.append(np.ones((img.shape[0], spacing, 3), dtype=np.uint8) * 255)
        images_with_spacing = images_with_spacing[:-1]

        stacked_image = np.concatenate(images_with_spacing, axis=1)

        filename = "combined_rgb_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        Image.fromarray(stacked_image).save(path)


    def log_frame(self, split, images, batch_idx, sample_token):
        """
        Log superimposed mask with camera images.
        """
        root = os.path.join(self.save_dir, "image_log_" + split)
        os.makedirs(root, exist_ok=True)

        for k in images:
            if k != 'samples':  # only visulise sample results (predictions)
                continue
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
                rgb_mask = combine_masks_to_rgb(mask4)  # [H, W, 3], float32 in [0,1]
                rgb_mask = (rgb_mask * 255).astype(np.uint8)

        # get camera image from nuScenes dataset
        sample = self.nusc.get('sample', sample_token[0])
        cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        img_dict = {}
        for cam in cam_names:
            sd_token = sample['data'][cam]
            sd_record = self.nusc.get('sample_data', sd_token)
            img_path = os.path.join(self.nusc.dataroot, sd_record['filename'])
            img = Image.open(img_path).convert("RGB")
            img_dict[cam] = img

        combined_img = self.stack_camera_imgs_with_bev(img_dict, rgb_mask)

        filename = "frame--{:05}.png".format(batch_idx)
        path = os.path.join(root, filename)
        combined_img.save(path, format="JPEG", quality=80, optimize=True)

    @staticmethod
    def stack_camera_imgs_with_bev(img_dict, rgb_mask, gap=10):
        W, H = img_dict['CAM_FRONT'].size
        gap_color = (255, 255, 255)
        row_width = W * 3 + gap * 2
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=48)

        # --- Step 1: define camera label height ---
        label_height = 60  # reserve space for top and bottom camera names

        # --- Step 2: create top and bottom camera label areas ---
        top_label_area = Image.new('RGB', (row_width, label_height), gap_color)
        bottom_label_area = Image.new('RGB', (row_width, label_height), gap_color)

        # --- Step 3: stitch camera images into two rows ---
        row1 = Image.new('RGB', (row_width, H), gap_color)
        row1.paste(img_dict['CAM_FRONT_LEFT'], (0, 0))
        row1.paste(img_dict['CAM_FRONT'], (W + gap, 0))
        row1.paste(img_dict['CAM_FRONT_RIGHT'], (2 * W + 2 * gap, 0))

        row2 = Image.new('RGB', (row_width, H), gap_color)
        row2.paste(img_dict['CAM_BACK_LEFT'], (0, 0))
        row2.paste(img_dict['CAM_BACK'], (W + gap, 0))
        row2.paste(img_dict['CAM_BACK_RIGHT'], (2 * W + 2 * gap, 0))

        # --- Step 4: stack all camera parts: top label + rows + bottom label ---
        cam_panel_height = label_height + H + gap + H + label_height
        cam_panel = Image.new('RGB', (row_width, cam_panel_height), gap_color)
        cam_panel.paste(top_label_area, (0, 0))
        cam_panel.paste(row1, (0, label_height))
        cam_panel.paste(row2, (0, label_height + H + gap))
        cam_panel.paste(bottom_label_area, (0, cam_panel_height - label_height))

        # --- Step 5: compute BEV layout: label + map + legend ---
        if isinstance(rgb_mask, np.ndarray):
            rgb_mask = Image.fromarray(rgb_mask.astype('uint8'))

        # Use only the raw image height (without label) for BEV map alignment
        raw_img_height = cam_panel.height - 2 * label_height  # 2H + gap
        bev_label_height = label_height  # blank space above BEV
        legend_height = 160
        mask_target_height = raw_img_height

        mask_orig_w, mask_orig_h = rgb_mask.size
        scale = mask_target_height / mask_orig_h
        mask_new_w = int(mask_orig_w * scale)
        rgb_mask_resized = rgb_mask.resize((mask_new_w, mask_target_height), Image.BILINEAR)

        # --- Step 6: compute final canvas size ---
        bev_panel_height = bev_label_height + mask_target_height + legend_height
        final_height = bev_panel_height
        final_width = cam_panel.width + gap + mask_new_w

        # --- Step 7: create canvas and paste camera + BEV components ---
        canvas = Image.new('RGB', (final_width, final_height), gap_color)
        canvas.paste(cam_panel, (0, 0))
        canvas.paste(rgb_mask_resized, (cam_panel.width + gap, bev_label_height)) 
        draw = ImageDraw.Draw(canvas)

        # --- Step 8: draw camera labels outside the images ---
        top_labels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
        bottom_labels = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        for i, label in enumerate(top_labels):
            x = i * (W + gap)
            y = 5  # or center in label_height
            text_bbox = draw.textbbox((0, 0), label, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (W - text_width) // 2
            draw.text((text_x, y), label, fill=(0, 0, 0), font=label_font)

        for i, label in enumerate(bottom_labels):
            x = i * (W + gap)
            y = cam_panel_height - label_height + 5
            text_bbox = draw.textbbox((0, 0), label, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (W - text_width) // 2
            draw.text((text_x, y), label, fill=(0, 0, 0), font=label_font)

        # --- Step 9: draw semantic legend below BEV ---
        colors = [
            (100, 100, 100),      # drivable_area
            (255, 178, 102),      # ped_crossing
            (220, 220, 100),      # walkway
            (255, 50, 50),        # stop_line
            (160, 120, 60),       # carpark_area
            (240, 240, 240),      # lane_divider
        ]
        labels = [
            "Drivable Area", "Ped Crossing", "Walkway",
            "Stop Line", "Carpark Area", "Lane Divider"
        ]
        box_w = 80
        box_h = 40
        spacing_x = 10
        spacing_y = 30
        legend_start_x = cam_panel.width + gap
        legend_start_y = bev_label_height + mask_target_height + spacing_x
        col_w = mask_new_w // 3

        for i, (color, label) in enumerate(zip(colors, labels)):
            row = i // 3
            col = i % 3
            x = legend_start_x + col * col_w + spacing_x
            y = legend_start_y + row * (box_h + spacing_y)
            draw.rectangle([x, y, x + box_w, y + box_h], fill=color, outline=(0, 0, 0))
            draw.text((x + box_w + 10, y), label, fill=(0, 0, 0), font=label_font)

        return canvas


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
            if self.image_style == 'video_frame':
                # Convert each image to a video frame
                sample_token = batch.get('sample_token', None)
                self.log_frame(split, images, batch_idx, sample_token)
            elif hasattr(pl_module, "control_key"):   # for cldm
                self.log_local_rgb(split, images,
                               pl_module.global_step, pl_module.current_epoch, batch_idx)
            else:   # for ldm
                self.log_local_rgb(split, images,
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