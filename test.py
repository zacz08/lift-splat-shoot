import torch
import os
import multiprocessing
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.models_pl import LiftSplatShoot
from src.data import compile_data
from src.logger import ImageLogger
from train_pl import seed_everything
import datetime

ckpt_path = 'logs/01_12_15_08_miou=34.4/best-ckpt-epoch=37-step=32642.ckpt'

# Scene classification filter (set to None to use all scenes)
# Options: None, 'normal', 'rainy', 'night', or ['rainy', 'night']
scene_class_file = 'data/nuscenes/scene_classes.json'  # path to classification JSON
scene_class_filter = 'night'  # e.g. 'night' to only evaluate night scenes

def main():
    multiprocessing.set_start_method('spawn')
    cfg = OmegaConf.load('./configs/lss.yaml')
    seed_everything(cfg.trainer_config.get('seed', 42))

    # create logger path
    now = datetime.datetime.now()
    log_folder_path ='logs/' + '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute)))
    os.makedirs(log_folder_path, exist_ok=True)
    print(f"[Log] Logging to {log_folder_path}")
    
    # save config used for this experiment
    config_save_path = os.path.join(log_folder_path, "config.yaml")
    OmegaConf.save(cfg, config_save_path)
    print(f"[Config] Saved to {config_save_path}")

    model = LiftSplatShoot(cfg)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"], strict=False)
    
    model.eval()

    # Determine scene class filter arguments
    sc_file = scene_class_file if scene_class_filter is not None else None
    sc_filter = scene_class_filter

    logger = ImageLogger(batch_frequency=cfg.trainer.log_freq, 
                         rescale=False,
                         log_folder=log_folder_path)
    _, val_dataloader = compile_data(cfg=cfg, parser_name='segmentationdata',
                                     scene_class_file=sc_file,
                                     scene_class_filter=sc_filter)

    trainer = pl.Trainer(
        strategy="auto", 
        accelerator='gpu',
        devices=1,
        precision=cfg.trainer_config.precision, 
        callbacks=[logger],
        logger=False)
    
    trainer.predict(model, val_dataloader)


if __name__ == "__main__":
    main()
