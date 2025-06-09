import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm
from src.models_pl import LiftSplatShoot
from src.data import compile_data

ckpt_path = './ckpts/6_layer_epoch=30-step=13299.ckpt'

def save_predictions(model, data_loader, data_split='train', dataroot='./data/nuscenes/'):
    
    # LSS predicts BEV segmentation maps saving path
    save_folder_predmap = os.path.join(dataroot, 'bev_pred_lss', data_split)
    if not os.path.exists(save_folder_predmap):
            os.makedirs(save_folder_predmap)

    # LSS predicts BEV features saving path
    save_folder_bevfeat = os.path.join(dataroot, 'bev_feat_lss', data_split)
    if not os.path.exists(save_folder_bevfeat):
        os.makedirs(save_folder_bevfeat)
    
    # json file saving path
    json_name = 'prompt_lss_' + data_split + '.json'
    json_path = os.path.join(dataroot, json_name)

    bev_seg_gt_folder = os.path.join(dataroot, 'bev_seg_gt_mask_192', data_split)

    
    with open(json_path, 'w') as json_file:
        for index, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                imgs, rots, trans, intrins, post_rots, post_trans, bev_seg_gt, bev_token = batch
                assert len(bev_token) == 1, "Only one bev_token is expected per batch"
                bev_token = bev_token[0]  # Get the single token from the batch
                preds, bev_feat = model(imgs, rots, trans, 
                                        intrins, post_rots, post_trans,
                                        return_feats=True)
            preds = (preds > 0)
            bev_feat_name = f"{index:05d}_bev_feat_{bev_token}.pt"
            feat_save_path = os.path.join(save_folder_bevfeat, bev_feat_name)
            pred_save_path = os.path.join(save_folder_predmap, f"{index:05d}_bev_pred_{bev_token}.npy")
            torch.save(bev_feat.squeeze(0).half(), feat_save_path)
            np.save(pred_save_path, preds.squeeze(0).numpy())

            # write data to json file
            gt_save_path = get_seg_map_name_by_sample_token(
                bev_seg_gt_folder,
                bev_token)
            assert gt_save_path is not None, f"Can't find bev gt image with sample_token: {bev_token}"
            pred_seg_map_path = f"{index:05d}_bev_pred_{bev_token}.npy"

            data = {
                "bev_feat": bev_feat_name,
                "pred_map": pred_seg_map_path,
                "bev_map_gt": gt_save_path
            }

            json.dump(data, json_file)
            json_file.write('\n')

def get_seg_map_name_by_sample_token(
        folder_path, 
        sample_token: str) -> str:
        
    for filename in os.listdir(folder_path):
        # get token by image name
        if filename.endswith(".jpg") or filename.endswith(".npy"):
            current_token = filename[len("00262_bev_gt_"):-len(".jpg")]
            if current_token == sample_token:
                return filename
    
    # return none if can't find the image
    return None


def main():
    cfg = OmegaConf.load('./configs/lss.yaml')

    model = LiftSplatShoot(cfg)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
    
    model.eval()

    cfg.loader.batch_size = 1  # Set batch size to 1 to generate predictions for each image individually
    cfg.dataset.version = 'trainval'
    train_loader, val_loader = compile_data(cfg=cfg, parser_name='segmentationdata')
    
    save_predictions(model, train_loader, data_split='train', dataroot=cfg.dataset.dataroot)
    save_predictions(model, val_loader, data_split='val', dataroot=cfg.dataset.dataroot)


if __name__ == "__main__":
    main()
