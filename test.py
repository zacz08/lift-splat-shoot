import torch
import multiprocessing
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.models_pl import LiftSplatShoot
from src.data import compile_data
from src.logger import ImageLogger

ckpt_path = './lightning_logs/version_2/checkpoints/best-ckpt-epoch=30-step=13299.ckpt'

def main():
    multiprocessing.set_start_method('spawn')
    cfg = OmegaConf.load('./configs/lss.yaml')

    model = LiftSplatShoot(cfg)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["state_dict"], strict=True)
    
    model.eval()


    logger = ImageLogger(batch_frequency=cfg.trainer.log_freq, rescale=False)
    train_dataloader, val_dataloader = compile_data(cfg=cfg, parser_name='segmentationdata')

    trainer = pl.Trainer(
        strategy="auto", 
        accelerator='gpu',
        devices=cfg.trainer.gpus,
        precision=cfg.trainer.precision, 
        callbacks=[logger])
    
    trainer.predict(model, val_dataloader)


if __name__ == "__main__":
    main()
