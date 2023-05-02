import os
from load_data import ECGDataModule
from models.unet import UNet
from models.wnet import WNet
from models.FECGMem import FECGMem
from models.LSTM_baseline import LSTM_baseline
from pytorch_lightning import Trainer
from pytorch_lightning.loops import Loop
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.saving import save_hparams_to_yaml
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
        'fecg_peak' : FECG_PEAK_LOSS_RATIO,
        'fecg_peak_mask' : FECG_PEAK_CLASS_RATIO - 1
    }

def make_unet(path : str = ''):
    print('=====Making UNet Model=====')
    if path:
        return UNet.load_from_checkpoint(path,
                                         sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                                         fecg_down_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                                         fecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                         batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                                         decoder_skips=SKIP, initial_conv_planes=INITIAL_CONV_PLANES,
                                         linear_layers=LINEAR_LAYERS, pad_length=PAD_LENGTH,
                                         embed_dim=EMBED_DIM, peak_downsamples=PEAK_DOWNSAMPLES,
                                         include_rnn=INCLUDE_RNN,
                                         )

    return UNet(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                fecg_down_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                fecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                decoder_skips=SKIP, initial_conv_planes=INITIAL_CONV_PLANES,
                linear_layers=LINEAR_LAYERS, pad_length=PAD_LENGTH,
                embed_dim=EMBED_DIM, peak_downsamples=PEAK_DOWNSAMPLES,
                include_rnn=INCLUDE_RNN,)

def make_lstm(path : str = '', unet_path : str = PRETRAINED_UNET_CKPT):
    print('=====Making LSTM Model=====')
    if path:
        return LSTM_baseline.load_from_checkpoint(path,
                                                  sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                                                  value_encoder_params=(
                                                  VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                                                  decoder_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                                  batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                                                  decoder_skips=SKIP, num_layers=2, embed_dim=EMBED_DIM,
                                                  window_length=WINDOW_LENGTH, pretrained_unet=None
                                                  )

    if unet_path:
        pretrained_unet = make_unet(path)
    else:
        pretrained_unet = None

    return LSTM_baseline(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                         value_encoder_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                         decoder_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                         batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                         decoder_skips=SKIP, num_layers=2, embed_dim=EMBED_DIM,
                         window_length=WINDOW_LENGTH, pretrained_unet=pretrained_unet)


def make_wnet(path : str = ''):
    print('=====Making WNet Model=====')
    if path:
        return WNet.load_from_checkpoint(path,
                                         sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                                         fecg_down_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                                         fecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                         mecg_down_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                                         mecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                         batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                                         )

    return WNet(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                fecg_down_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                fecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                mecg_down_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                mecg_up_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, )

def make_fecgmem(path : str = '', unet_path : str = PRETRAINED_UNET_CKPT):
    print('=====Making FECGMem Model=====')
    if path:
        return FECGMem.load_from_checkpoint(path,
                                            sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                                            query_encoder_params=(MEMORY_DOWN_PLANES, MEMORY_DOWN_KERNELS, MEMORY_DOWN_STRIDES),
                                            value_encoder_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                                            decoder_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                                            embed_dim=EMBED_DIM, memory_length=MEMORY_LENGTH,
                                            batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                                            window_length=WINDOW_LENGTH, pretrained_unet = None,
                                            decoder_skips=SKIP, initial_conv_planes=INITIAL_CONV_PLANES,
                                            linear_layers=LINEAR_LAYERS, pad_length=PAD_LENGTH,
                                            peak_downsamples=PEAK_DOWNSAMPLES, include_rnn=INCLUDE_RNN,
                                            )

    if unet_path:
        pretrained_unet = make_unet(path)
    else:
        pretrained_unet = None

    return FECGMem(sample_ecg=SAMPLE_ECG, loss_ratios=get_loss_param_dict(),
                   query_encoder_params=(MEMORY_DOWN_PLANES, MEMORY_DOWN_KERNELS, MEMORY_DOWN_STRIDES),
                   value_encoder_params=(VALUE_DOWN_PLANES, VALUE_DOWN_KERNELS, VALUE_DOWN_STRIDES),
                   decoder_params=(UP_PLANES, UP_KERNELS, UP_STRIDES),
                   embed_dim=EMBED_DIM, memory_length=MEMORY_LENGTH,
                   batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, window_length=WINDOW_LENGTH,
                   pretrained_unet=pretrained_unet, decoder_skips=SKIP,
                   initial_conv_planes=INITIAL_CONV_PLANES, linear_layers=LINEAR_LAYERS,
                   pad_length=PAD_LENGTH, peak_downsamples=PEAK_DOWNSAMPLES,
                   include_rnn=INCLUDE_RNN, )

def main(**kwargs):
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name=MODEL_NAME)

    # For reproducibility
    seed_everything(SEED, True)

    model_registry = {
        'unet' : make_unet,
        'wnet' : make_wnet,
        'fecgmem' : make_fecgmem,
        'lstm_baseline' : make_lstm,
    }

    if MODEL not in model_registry:
        raise NotImplementedError('Model not implemented')
    else:
        model = model_registry[MODEL]()

    model.to(DEVICE)

    data = ECGDataModule(data_dir=DATA_DIR, window_size=500, num_workers=NUM_DATA_WORKERS,
                         batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))
    bar = LitProgressBar()

    save_hparams_to_yaml(f'Run/Logging/{MODEL_NAME}/hyperparams.yaml', vars(args))

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
                     auto_lr_find=True,
                     profiler='pytorch',)

    print(f"======= Training {MODEL_NAME} =======")
    if MODEL_VER:
        runner.fit(model, datamodule=data, ckpt_path=f'Run/Logging/{MODEL_NAME}/version_{MODEL_VER}/checkpoints/last.ckpt',)
    else:
        runner.fit(model, datamodule=data)

if __name__ == '__main__':
    main()
