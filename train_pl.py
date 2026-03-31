import os

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

import datetime
import pytorch_lightning as pl
import multiprocessing
from src.logger import ImageLogger
from src.model_summary import ModelSummary
from omegaconf import OmegaConf
from src.models_pl import LiftSplatShoot
from src.data import compile_data
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import torch
import numpy as np
import random


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


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        
        # format learning rate to scientific notation
        if 'lr' in items:
            items['lr'] = f"{items['lr']:.2e}"

        if 'global_step' in items:
            items['global_step'] = int(items['global_step'])
                
        items.pop("v_num", None)    # remove version number from progress bar
        return items


## Configs
config_path = './configs/lss.yaml'

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # load config
    cfg = OmegaConf.load(config_path)
    trainer_cfg = cfg.trainer_config

    seed_everything(trainer_cfg.get('seed', 42))

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

    train_dataloader, val_dataloader = compile_data(cfg=cfg, parser_name='segmentationdata')

    # image logger callback
    logger = ImageLogger(
        batch_frequency=trainer_cfg.logger_freq, 
        rescale=False,      # rescale=False when using mask, =True when using rgb image
        log_folder=log_folder_path)

    # model summary callback
    model_summary = ModelSummary(log_folder=log_folder_path)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_folder_path,
        monitor='val/IoU',
        filename='best-ckpt-{epoch}-{step}',
        mode='max')
    
    trainer = pl.Trainer(strategy="auto", 
                         accelerator="gpu", 
                         devices=trainer_cfg.gpu_num, 
                         precision=trainer_cfg.precision, 
                         callbacks=[logger, model_summary, checkpoint_callback, CustomProgressBar()],
                         logger=False, 
                         max_epochs=trainer_cfg.epochs,
                         accumulate_grad_batches=trainer_cfg.acc_grad_batches)
        

    resume_path = trainer_cfg.get('resume_path', None)
    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint from {resume_path}")
        ckpt_path = resume_path
    else:
        ckpt_path = None

    # Train!
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)