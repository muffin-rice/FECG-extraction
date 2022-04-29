import os
from load_data import ECGDataModule
from vae_backbone import VAE
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from hyperparams import *

from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LitProgressBar(TQDMProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar

def main(**kwargs):
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name=MODEL_NAME)

    # For reproducibility
    seed_everything(SEED, True)

    model = VAE(sample_ecg=SAMPLE_ECG)

    data = ECGDataModule(data_dir=DATA_DIR, window_size=500, num_workers=NUM_DATA_WORKERS,
                         batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))
    bar = LitProgressBar()

    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=3,
                                         every_n_train_steps = SAVE_N_STEPS,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                         bar,
                     ],
                     log_every_n_steps=LOG_STEPS,
                     check_val_every_n_epoch=5,
                     accelerator=device,
                     devices=NUM_TRAINER_WORKERS,
                     max_epochs=NUM_EPOCHS,
                     auto_lr_find=True)

    print(f"======= Training {MODEL_NAME} =======")
    runner.fit(model, datamodule=data)

if __name__ == '__main__':
    main()
