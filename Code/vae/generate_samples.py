from vae_backbone import VAE
from load_data import ECGDataModule
from train_vae import DATA_DIR
from scipy.signal import istft
import pickle as pkl
import numpy as np
import torch

BATCH_SIZE = 8

def stft(sig: np.array):
    if len(sig.shape) == 3:
        x = []
        for isig in range(sig.shape[0]):  # iterate through batch
            stft_image = torch.stft(torch.from_numpy(sig[isig, ...]), n_fft=32, normalized=True, hop_length=1,
                                    onesided=True, return_complex=True, center=False)

            x.append(torch.cat((stft_image.real, stft_image.imag), dim=1).unsqueeze(0))

        return torch.cat(x)

    else:
        stft_image = torch.stft(torch.from_numpy(sig), n_fft=32, normalized=True, hop_length=1,
                                onesided=True, return_complex=True, center=False)

        return torch.cat((stft_image.real, stft_image.imag), dim=1)


def invert_stft_batch(stft_sig):  # batch_size x 1 (optional) x 34 x 469

    if len(stft_sig.shape) == 4:
        x = []
        for isig in range(stft_sig.shape[0]):  # iterate through batch
            stft_sig1 = stft_sig[isig, ...]
            complex_stft = torch.complex(stft_sig1[:, :17, :], stft_sig1[:, 17:, :])
            sig = torch.istft(complex_stft, n_fft=32, hop_length=1, onesided=True, return_complex=False, length=500,
                              center=False)

            x.append(sig.unsqueeze(0))

        return torch.cat(x)

    else:
        stft_sig1 = stft_sig
        complex_stft = torch.complex(stft_sig1[:, :17, :], stft_sig1[:, 17:, :])

        return torch.istft(complex_stft, n_fft=32, hop_length=1, onesided=True, return_complex=False, length=500,
                           center=False)


def main():
    model = VAE(z_dim = 40, mode = 'DIRECT')
    model.load_from_checkpoint('Run/Logging/modelv0.5/version_8/checkpoints/epoch=22-step=10189.ckpt')

    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, num_workers=0)
    dataloader = dm.test_dataloader()

    for i, d in enumerate(dataloader):
        if i > 8:
            break
1
        fecg_stft, orig_fecg = d['fecg_stft'], d['fecg_sig']
        orig_aecg = d['fecg_sig'] + d['mecg_sig']

        model_output = model(d)
        model_fecg = invert_stft_batch(model_output['x_recon'])

        d = {'orig_aecg' : orig_aecg.numpy(), 'orig_fecg' : orig_fecg.numpy(), 'model_fecg' : model_fecg.detach().numpy(),
             'recon_loss' : model_output['recon_loss'].detach().numpy()}#, 'kl_loss' : model_output['kl_loss']}

        with open(f'Run/Reconstructions/test_{i}.pkl', 'wb') as f:
            pkl.dump(d, f)


if __name__ == '__main__':
    main()