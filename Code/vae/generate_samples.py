import pickle as pkl

import numpy as np
import torch
from scipy.signal import istft

from load_data import ECGDataModule
from train_vae import DATA_DIR
from vae_backbone import STFT_VAE

BATCH_SIZE = 8


def generate_stft_image2(sig, **stft_args):
    stft_image = torch.stft(torch.from_numpy(sig), n_fft=32, normalized=True, hop_length=1,
                            onesided=True, return_complex=True, center=False)
    return torch.cat((stft_image.real, stft_image.imag), axis=1)


def invert_stft_batch_torch(stft_sig):  # batch_size x num_c x 34 x 469
    stft_sig1 = stft_sig  # [:, 0, :, :]
    complex_stft = torch.complex(stft_sig1[:, :17, :], stft_sig1[:, 17:, :])
    return torch.istft(complex_stft, n_fft=32, hop_length=1, onesided=True, return_complex=False, length=500,
                       center=False)  # win_length=32,


def invert_stft(stft_sig):
    x = stft_sig[:, :17, :] + 1j * stft_sig[:, 17:, :]
    orig_sig = istft(x, fs=125, nperseg=32, noverlap=31, input_onesided=True, boundary=None)[1]
    return orig_sig


def invert_stft_batch(stft_sig):  # batch_size x 1 x num_c x 34 x 469
    x = []
    for ent in range(stft_sig.shape[0]):
        x.append(invert_stft(stft_sig[ent, 0, :, :, :]))

    return np.array(x)


def main():
    model = STFT_VAE(20)
    model.load_from_checkpoint('Run/Logging/modelv0.3/version_3/checkpoints/epoch=6-step=6146.ckpt')

    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, num_workers=0)
    dataloader = dm.test_dataloader()

    for i, (aecg_stft, fecg_stft, orig_fecg) in enumerate(dataloader):
        if i > 8:
            break
        orig_aecg = invert_stft_batch_torch(aecg_stft)
        model_output = model(aecg_stft, fecg_stft, orig_fecg)
        model_fecg = invert_stft_batch_torch(model_output['x_recon'].detach().numpy())

        d = {'orig_aecg': orig_aecg, 'orig_fecg': orig_fecg, 'model_fecg': model_fecg,
             'recon_loss': model_output['recon_loss'], 'kl_loss': model_output['kl_loss']}

        with open(f'Run/Reconstructions/test_{i}.pkl', 'wb') as f:
            pkl.dump(d, f)


if __name__ == '__main__':
    main()
