import os

import torch

from scipy.io import savemat

from load_data import ECGDataModule
from models.vae import VAE
from train import DATA_DIR, NUM_DATA_WORKERS, BATCH_SIZE, NUM_TRAINER_WORKERS

run_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')
model_name = 'modelv1.0'
model_version = 'version_24'
model_path = f'{run_root}/logging/{model_name}/{model_version}/checkpoints/last.ckpt'
output_root = f'{run_root}/output/{model_name}/{model_version}'

if __name__ == '__main__':
    model = VAE.load_from_checkpoint(model_path)
    model.eval()

    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, dataset_type='', num_workers=NUM_DATA_WORKERS,
                         batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))

    model.eval()
    with torch.no_grad():
        print('start train eval')

        dl = dm.train_dataloader()
        index = 0
        for j, d in enumerate(dl):
            model_output = model(d)

            for i in range(d['fecg_sig'].shape[0]):
                out = {'fecg_sig': d['fecg_sig'][i].detach().cpu().numpy(),
                       'mecg_sig': d['mecg_sig'][i].detach().cpu().numpy(),
                       'mecg_recon': model_output['x_recon'][i].detach().cpu().numpy()}

                savemat(output_root + f'/train/ecg_{index}.mat', out)
                index += 1

        print('train eval done\nstart validation eval')

        dl = dm.val_dataloader()
        index = 0
        for d in dl:
            model_output = model(d)

            for i in range(d['fecg_sig'].shape[0]):
                out = {'fecg_sig': d['fecg_sig'][i].detach().cpu().numpy(),
                       'mecg_sig': d['mecg_sig'][i].detach().cpu().numpy(),
                       'mecg_recon': model_output['x_recon'][i].detach().cpu().numpy()}

                savemat(output_root + f'/validation/ecg_{index}.mat', out)
                index += 1

        print('validation eval done\nstart test eval')

        dl = dm.test_dataloader()
        index = 0
        for d in dl:
            model_output = model(d)

            for i in range(d['fecg_sig'].shape[0]):
                out = {'fecg_sig': d['fecg_sig'][i].detach().cpu().numpy(),
                       'mecg_sig': d['mecg_sig'][i].detach().cpu().numpy(),
                       'mecg_recon': model_output['x_recon'][i].detach().cpu().numpy()}

                savemat(output_root + f'/test/ecg_{index}.mat', out)
                index += 1

        print('all done')
