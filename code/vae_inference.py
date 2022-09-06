import os
import io
import base64

import torch

from scipy.io import savemat

from load_data import ECGDataModule, ffilenames
from vae import VAE
from vae_train import DATA_DIR, NUM_DATA_WORKERS, BATCH_SIZE, NUM_TRAINER_WORKERS

run_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')
model_name = 'modelv1.2'
model_version = 'version_0'
model_path = f'{run_root}/logging/{model_name}/{model_version}/checkpoints/last.ckpt'
output_root = f'{run_root}/output/{model_name}/{model_version}'

if __name__ == '__main__':
    model = VAE.load_from_checkpoint(model_path)
    model.eval()

    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, dataset_type='', num_workers=1,
                         batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))

    model.eval()
    with torch.no_grad():
        print('start train eval')

        dl = dm.train_dataloader()
        index = 0
        for j, d in enumerate(dl):
            model_output = model(d)

            for i in [-1]:#range(d['fecg_sig'].shape[0]):
                out = {'fecg_sig': d['fecg_sig'][i].detach().cpu().numpy(),
                       'mecg_sig': d['mecg_sig'][i].detach().cpu().numpy(),
                       'mecg_recon': model_output['x_recon'][i].detach().cpu().numpy(),
                       'snr': d['snr'][i].detach().cpu().numpy(),
                       'fecg_fnames': base64.b64decode(bytes(d['fecg_fname'][i])).decode('utf8')}

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
                       'mecg_recon': model_output['x_recon'][i].detach().cpu().numpy(),
                'snr': d['snr'][i].detach().cpu().numpy()}

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
                       'mecg_recon': model_output['x_recon'][i].detach().cpu().numpy(),
                'snr': d['snr'][i].detach().cpu().numpy()}

                savemat(output_root + f'/test/ecg_{index}.mat', out)
                index += 1

        print('all done')
