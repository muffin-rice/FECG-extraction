import os

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar

from tqdm import tqdm

from load_data import ECGDataModule
from vae import VAE

SEED = 1
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run', 'logging')
MODEL_NAME = 'modelv1.1'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
LOG_STEPS = 10
LEARNING_RATE = 1e-4
Z_DIM = 64
LOSS_RATIO = 1000
NUM_TRAINER_WORKERS = 1
NUM_DATA_WORKERS = 8
BATCH_SIZE = 128
FIND_UNUSED = False
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 2000

if __name__ == '__main__':
    model_log_path = os.path.join(LOG_DIR, MODEL_NAME)
    if not os.path.exists(model_log_path):
        os.mkdir(os.path.join(LOG_DIR, MODEL_NAME))
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name=MODEL_NAME)

    # For reproducibility
    seed_everything(SEED, True)

    class LitProgressBar(TQDMProgressBar):

        def init_validation_tqdm(self):
            bar = tqdm(
                disable=True,
            )
            return bar

    data = ECGDataModule(data_dir=DATA_DIR, window_size=500, dataset_type='', num_workers=NUM_DATA_WORKERS,
                         batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))
    model = VAE(sample_ecgs=[])
    bar = LitProgressBar()

    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=5,
                                         dirpath=tb_logger.log_dir + "/checkpoints",
                                         monitor="val_loss",
                                         save_last=True),
                         bar,
                     ],
                     log_every_n_steps=LOG_STEPS,
                     check_val_every_n_epoch=5,
                     # strategy=DDPPlugin(find_unused_parameters=FIND_UNUSED),
                     accelerator=DEVICE,
                     devices=NUM_TRAINER_WORKERS,
                     max_epochs=NUM_EPOCHS,
                     auto_lr_find=True)

    print(f"======= Training {MODEL_NAME} =======")
    runner.fit(model, datamodule=data)
