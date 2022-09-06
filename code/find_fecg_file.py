import os
import base64
import pickle

import torch
from scipy.io import loadmat

from load_data import ECGDataModule
from vae import VAE
from vae_train import DATA_DIR, NUM_DATA_WORKERS, BATCH_SIZE, NUM_TRAINER_WORKERS

run_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')
model_name = 'modelv1.2'
model_version = 'version_0'
model_path = f'{run_root}/logging/{model_name}/{model_version}/checkpoints/last.ckpt'
output_root = f'{run_root}/output/{model_name}/{model_version}'
sig_to_name = {}

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
            for i in range(d['fecg_sig'].shape[0]):
                intermediate_path = base64.b64decode(bytes(d['fecg_fname'][i])).decode('utf8').rstrip()
                inter_data = loadmat(intermediate_path)['fecg_fname'][0].rsplit('/', 1)[1].split('_')
                sub = inter_data[0][1:]
                snr = inter_data[1]

                sig_to_name[tuple(map(lambda x: round(x, 4), d['fecg_sig'][i][0][:100].detach().cpu().numpy()))] =
                print(sig_to_name[tuple(map(lambda x: round(x, 4), d['fecg_sig'][i][0][:100].detach().cpu().numpy()))])
                raise SystemExit

        print('train eval done\nstart validation eval')

        dl = dm.val_dataloader()
        index = 0
        for d in dl:
            for i in range(d['fecg_sig'].shape[0]):
                sig_to_name[tuple(map(lambda x: round(x, 4), d['fecg_sig'][i][0][:100].detach().cpu().numpy()))] = base64.b64decode(bytes(d['fecg_fname'][i])).decode('utf8')[0].rpartition('/')


        print('validation eval done\nstart test eval')

        dl = dm.test_dataloader()
        index = 0
        for d in dl:
            for i in range(d['fecg_sig'].shape[0]):
                sig_to_name[tuple(map(lambda x: round(x, 4), d['fecg_sig'][i][0][:100].detach().cpu().numpy()))] = base64.b64decode(bytes(d['fecg_fname'][i])).decode('utf8')[0].rpartition('/')

        print('all done')

    # save fecg signal snippet to file name dictionary
    with open('/Users/Richard/git/FECG-extraction/run/fecg_fnames.pickle', 'wb') as file:
        pickle.dump(sig_to_name, file, protocol=pickle.HIGHEST_PROTOCOL)

