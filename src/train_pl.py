import multiprocessing
import pytorch_lightning as pl
from cldm.logger import ImageLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from cldm.model import create_model

config_path = './configs/lss.yaml'

def main():
    multiprocessing.set_start_method('spawn')
    cfg = OmegaConf.load(config_path)

    model = create_model(config_path).cpu()
    model.load_separate_ckpt(cfg.cldm_ckpt, cfg.bevnet_ckpt)

    # save_dir = os.path.join(
    #     cfg.LOG_DIR, time.strftime('%d%B%Yat%H_%M_%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    # )
    # tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)

    cldm_cfg = cfg.controlnet_config
    cfg = OmegaConf.load(cldm_cfg).model.params.trainer_config

    logger = ImageLogger(batch_frequency=cfg.logger_freq, 
                         rescale=False)    # rescale=False when using mask, =True when using rgb image

    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/IoU',
        filename='best-ckpt-{epoch}-{step}',
        mode='max')

    trainer = pl.Trainer(
        strategy="auto", 
        accelerator='gpu',
        devices=cfg.gpu_num,
        precision=cfg.precision, 
        # sync_batchnorm=True,
        # weights_summary='full',
        profiler='simple',
        callbacks=[logger, checkpoint_callback],
        max_epochs=cfg.epochs)
 
    trainer.fit(model)


if __name__ == "__main__":
    main()
