import os
from load_data import ECGDataModule
from models.unet import UNet
from models.wnet import WNet
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
        # self.steps += 1
        # if self.steps == self.STEP_THRESHOLD:
        #     self.steps = 0
        # else:
        #     return
        # global NOISE_STD
        # NOISE_STD += 0.0004
        # if NOISE_STD == 0.0012:
        #     quit()


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
        'fp_bce' : FECG_BCE_RATIO,
        'fecg' : FECG_RATIO,
        'mecg' : MECG_RATIO,
        'fp_bce_class' : FECG_BCE_CLASS_RATIO-1
    }

def make_unet():
    return UNet(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                fecg_down_params=(NUM_PLANES_DOWN, NUM_KERNELS, NUM_STRIDES),
                fecg_up_params=(NUM_PLANES_UP, DECODER_KERNELS, DECODER_STRIDES),
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

def make_wnet():
    return WNet(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                fecg_down_params=(NUM_PLANES_DOWN, NUM_KERNELS, NUM_STRIDES),
                fecg_up_params=(NUM_PLANES_UP, DECODER_KERNELS, DECODER_STRIDES),
                mecg_down_params=(NUM_PLANES_DOWN, NUM_KERNELS, NUM_STRIDES),
                mecg_up_params=(NUM_PLANES_UP, DECODER_KERNELS, DECODER_STRIDES),
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

def main(**kwargs):
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name=MODEL_NAME)

    # For reproducibility
    seed_everything(SEED, True)

    if MODEL == 'unet':
        model = make_unet()
    elif MODEL == 'wnet':
        model = make_wnet()
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
                     auto_lr_find=True)

    print(f"======= Training {MODEL_NAME} =======")
    runner.fit(model, datamodule=data)#, ckpt_path=f'Run/Logging/{MODEL_NAME}/version_2/checkpoints/last.ckpt',)

if __name__ == '__main__':
    main()
