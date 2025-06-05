import torch
import multiprocessing
import pytorch_lightning as pl
# from cldm.logger import ImageLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from src.models_pl import LiftSplatShoot
from src.data import compile_data
from src.logger import ImageLogger

config_path = './configs/lss.yaml'

def main():
    multiprocessing.set_start_method('spawn')
    cfg = OmegaConf.load(config_path)

    model = LiftSplatShoot(cfg)
    model.load_state_dict(torch.load('./ckpts/lss_init.ckpt')["state_dict"], strict=False)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/IoU',
        filename='best-ckpt-{epoch}-{step}',
        mode='max')
    logger = ImageLogger(batch_frequency=cfg.trainer.log_freq, rescale=False)
    
    train_dataloader, val_dataloader = compile_data(cfg=cfg, parser_name='segmentationdata')

    trainer = pl.Trainer(
        strategy="auto", 
        accelerator='gpu',
        devices=cfg.trainer.gpus,
        precision=cfg.trainer.precision, 
        callbacks=[logger, checkpoint_callback],
        max_epochs=cfg.trainer.epochs)
    
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
