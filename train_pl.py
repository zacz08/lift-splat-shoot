import os
import torch
import random
import datetime
import numpy as np
import multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from src.models_pl import LiftSplatShoot
from src.data import compile_data
from src.logger import ImageLogger

def seed_everything(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set seed for PyTorch Lightning
    pl.seed_everything(seed, workers=True)
    
    print(f"[Seed] All random seeds set to {seed} for reproducibility.")

config_path = './configs/lss.yaml'

def main():
    multiprocessing.set_start_method('spawn')
    cfg = OmegaConf.load(config_path)

    seed_everything(cfg.trainer.get('seed', 42))
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
    model.log_dir = log_folder_path
    # model.load_state_dict(torch.load('./ckpts/lss_init.ckpt')["state_dict"], strict=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_folder_path,
        monitor='val/IoU',
        filename='best-ckpt-{epoch}-{step}',
        mode='max')
    
    logger = ImageLogger(
        batch_frequency=cfg.trainer.log_freq, 
        rescale=False,
        log_folder=log_folder_path)
    
    train_dataloader, val_dataloader = compile_data(cfg=cfg, parser_name='segmentationdata')

    trainer = pl.Trainer(
        strategy="auto", 
        accelerator='gpu',
        devices=cfg.trainer.gpus,
        precision=cfg.trainer.precision, 
        callbacks=[logger, checkpoint_callback],
        logger=False,
        max_epochs=cfg.trainer.epochs)
    
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
