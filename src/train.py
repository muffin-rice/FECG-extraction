import os
from load_data import ECGDataModule
from models.unet import UNet
from models.wnet import WNet
from models.FECGMem import FECGMem
from pytorch_lightning import Trainer
from pytorch_lightning.loops import Loop
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from hyperparams import *

from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar

import torch

torch.set_default_dtype(torch.float32)

class LitProgressBar(TQDMProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar

class NoiseLoop(Loop):
    '''loop to gradually add noise to the mecg_sig if necessary '''

    def __init__(self, model, optimizer, dataloader):
        super().__init__()
        self.model=model
        self.optimizer=optimizer
        self.dataloader = dataloader
        self.batch_idx=0
        self.steps = 0
        self.STEP_THRESHOLD = 1000

    @property
    def done(self):
        return self.batch_idx >= len (self.dataloader)

    def reset(self):
        self.dataloader_iter = iter(self.dataloader)

    def noise_step(self):
        return

    def advance(self, *args, **kwargs):
        batch = next(self.dataloader)
        self.optimizer.zero_grad()
        loss = self.model.training_step(batch, self.batch_idx)
        if loss < LOSS_THRESHOLD:
            self.noise_step()
        loss.backward()
        self.optimizer.step()

def get_loss_param_dict():
    return {
        'fecg' : FECG_RATIO,
        'mecg' : MECG_RATIO,
        'fecg_peak' : FECG_PEAK_LOSS_RATIO,
    }

def make_unet(path : str = ''):
    print('=====Making UNet Model=====')
    if path:
        return UNet.load_from_checkpoint(path,
                                         sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                                         fecg_down_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                                         fecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                         batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                                         decoder_skips=SKIP, initial_conv_planes=INITIAL_CONV_PLANES,
                                         linear_layers=LINEAR_LAYERS, pad_length=PAD_LENGTH,
                                         )

    return UNet(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                fecg_down_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                fecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                decoder_skips=SKIP, initial_conv_planes=INITIAL_CONV_PLANES,
                linear_layers=LINEAR_LAYERS, pad_length=PAD_LENGTH,)

def make_wnet(path : str = ''):
    print('=====Making WNet Model=====')
    if path:
        return WNet.load_from_checkpoint(path,
                                         sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                                         fecg_down_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                                         fecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                         mecg_down_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                                         mecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                         batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                                         )

    return WNet(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                fecg_down_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                fecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                mecg_down_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                mecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,)

def make_fecgmem(path : str = '', unet_path : str = PRETRAINED_UNET_CKPT):
    print('=====Making FECGMem Model=====')
    if path:
        return FECGMem.load_from_checkpoint(path,
                                            sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                                            query_encoder_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                                            value_encoder_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                                            decoder_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                            key_dim=KEY_DIM, val_dim=VAL_DIM, memory_length=MEMORY_LENGTH,
                                            batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                                            window_length=WINDOW_LENGTH, pretrained_unet = None,
                                            decoder_skips=SKIP, initial_conv_planes=INITIAL_CONV_PLANES,
                                            linear_layers=LINEAR_LAYERS,  pad_length=PAD_LENGTH,
                                            )

    if unet_path:
        pretrained_unet = make_unet(path)
    else:
        pretrained_unet = None

    return FECGMem(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                   query_encoder_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                   value_encoder_params=(DOWN_PLANES, DOWN_KERNELS, DOWN_STRIDES),
                   decoder_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                   key_dim=KEY_DIM, val_dim=VAL_DIM, memory_length=MEMORY_LENGTH,
                   batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, window_length=WINDOW_LENGTH,
                   pretrained_unet=pretrained_unet, decoder_skips=SKIP,
                   initial_conv_planes=INITIAL_CONV_PLANES, linear_layers=LINEAR_LAYERS,
                   pad_length=PAD_LENGTH,)

def main(**kwargs):
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name=MODEL_NAME)

    # For reproducibility
    seed_everything(SEED, True)

    if MODEL == 'unet':
        model = make_unet()
    elif MODEL == 'wnet':
        model = make_wnet()
    elif MODEL == 'fecgmem':
        model = make_fecgmem()
    else:
        raise NotImplementedError

    data = ECGDataModule(data_dir=DATA_DIR, window_size=500, num_workers=NUM_DATA_WORKERS,
                         batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))
    bar = LitProgressBar()

    runner = Trainer(logger=tb_logger,
                     auto_scale_batch_size=True,
                     callbacks=[
                         LearningRateMonitor(logging_interval='step'),
                         ModelCheckpoint(save_top_k=3,
                                         every_n_train_steps = SAVE_N_STEPS,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_total_loss",
                                         save_last=True),
                         bar,
                     ],
                     log_every_n_steps=LOG_STEPS,
                     check_val_every_n_epoch=TRAIN_PER_VAL_RUN,
                     accelerator=DEVICE,
                     devices=NUM_TRAINER_WORKERS,
                     max_epochs=NUM_EPOCHS,
                     auto_lr_find=True,)
                     # profiler='pytorch',)

    print(f"======= Training {MODEL_NAME} =======")
    if MODEL_VER:
        runner.fit(model, datamodule=data, ckpt_path=f'Run/Logging/{MODEL_NAME}/version_{MODEL_VER}/checkpoints/last.ckpt',)
    else:
        runner.fit(model, datamodule=data)

if __name__ == '__main__':
    main()
