import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything

import load_data
from vae_backbone import VAE

SEED = 1
LOG_DIR = 'Run/Logging'
MODEL_NAME = 'modelv0.4'
DATA_DIR = 'Data/preprocessed_data/paired_data'
LOG_STEPS = 10
LEARNING_RATE = 0.02
Z_DIM = 20
LOSS_RATIO = 10000
NUM_TRAINER_WORKERS = 1
NUM_DATA_WORKERS = 1
BATCH_SIZE = 1
FIND_UNUSED = False
USE_GPU = 'cpu'


def calc_loss(recon_loss, kl_loss):
    return recon_loss + kl_loss


def main(num_epochs=10, **kwargs):
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name=MODEL_NAME)

    # For reproducibility
    seed_everything(SEED, True)

    model = VAE(z_dim=Z_DIM, learning_rate=LEARNING_RATE, mode='STFT')

    data = load_data.ECGDataModule(data_dir=DATA_DIR, window_size=500, num_workers=NUM_DATA_WORKERS,
                                   batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))

    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=5,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     log_every_n_steps=LOG_STEPS,
                     strategy=DDPPlugin(find_unused_parameters=FIND_UNUSED),
                     accelerator=USE_GPU,
                     devices=NUM_TRAINER_WORKERS,
                     auto_lr_find=True)

    print(f"======= Training {MODEL_NAME} =======")
    runner.fit(model, datamodule=data)


if __name__ == '__main__':
    main()
