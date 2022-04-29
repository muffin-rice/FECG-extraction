import torch
from hyperparams import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal
import pytorch_lightning as pl
import math
import numpy as np
import matplotlib.pyplot as plt


class EncoderX(nn.Module):

    def __init__(self, im_chan : int = START_CHANNELS, z_dim : int = Z_DIM, hidden_dim : (int) = NUM_PLANES, kernels : (int) = NUM_KERNELS,
                 strides=NUM_STRIDES):
        super(EncoderX, self).__init__()
        self.z_dim = z_dim
        self.encode_x = nn.Sequential(
            self.encoderX_block(im_chan, NUM_PLANES[0], kernel_size=kernels[0], stride=strides[0]),
            self.encoderX_block(NUM_PLANES[0], NUM_PLANES[1], kernel_size=kernels[1], stride=strides[1]),
            self.encoderX_block(NUM_PLANES[1], NUM_PLANES[2], kernel_size=kernels[2], stride=strides[2]),
            self.encoderX_block(NUM_PLANES[2], z_dim * 2, kernel_size=kernels[3], stride=strides[3], final_layer=True),
        )

    def encoderX_block(self, input_channels : int, output_channels : int, kernel_size : int = 4, stride : int = 2, final_layer : bool = False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm1d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, signal : torch.Tensor):
        encoding_x = self.encode_x(signal)
        encoding_x = encoding_x.view(len(encoding_x), -1)

        # The stddev output is treated as the log of the variance of the normal
        # distribution by convention and for numerical stability
        return encoding_x[:, :self.z_dim], encoding_x[:, self.z_dim:].exp()


def stft_batch(sig: np.array):
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


def invert_stft_batch(stft_sig : torch.Tensor): #batch_size x 34 x 469

    if len(stft_sig.shape) == 4:
        x = []
        for isig in range(stft_sig.shape[0]): #iterate through batch
            stft_sig1 = stft_sig[isig, ...]
            complex_stft = torch.complex(stft_sig1[:,:17,:], stft_sig1[:,17:,:])
            sig = torch.istft(complex_stft, n_fft=32, hop_length=1, onesided=True, return_complex=False, length=500, center=False)

            x.append(sig.unsqueeze(0))


        return torch.cat(x)

    else:

        stft_sig1 = stft_sig
        complex_stft = torch.complex(stft_sig1[:,:17,:], stft_sig1[:,17:,:])

        return torch.istft(complex_stft, n_fft=32, hop_length=1, onesided=True, return_complex=False, length=500, center=False)


class DecoderX(nn.Module):

    def __init__(self, im_chan=START_CHANNELS, z_dim=Z_DIM, hidden_dim=NUM_PLANES, kernels=DECODER_KERNELS,
                 strides=DECODER_STRIDES, recon_length=500):
        super(DecoderX, self).__init__()
        self.z_dim, self.recon_length = z_dim, recon_length
        #         self.gen = nn.Sequential(
        #             self.make_decoderX_block(z_dim, NUM_PLANES[-1], kernel_size=kernels[-1], stride=strides[-1]),
        #             self.make_decoderX_block(NUM_PLANES[-1], NUM_PLANES[-2], kernel_size=kernels[-2], stride=strides[-2]),
        #             self.make_decoderX_block(NUM_PLANES[-2], NUM_PLANES[-3], output_padding=1, kernel_size=kernels[-3], stride=strides[-3]),
        #             self.make_decoderX_block(NUM_PLANES[-3], im_chan, kernel_size=kernels[-4]+4, stride=strides[-4], final_layer=True),
        #         )

        self.gen = nn.Sequential(
            self.decoderX_block(z_dim, NUM_PLANES[-1], kernel_size=kernels[0], stride=strides[0]),
            self.decoderX_block(NUM_PLANES[-1], NUM_PLANES[-2], kernel_size=kernels[1], stride=strides[1]),
            self.decoderX_block(NUM_PLANES[-2], NUM_PLANES[-3], output_padding=0, kernel_size=kernels[2],
                                     stride=strides[2]),
            self.decoderX_block(NUM_PLANES[-3], im_chan, kernel_size=kernels[3], stride=strides[3],
                                     final_layer=True),
        )

    def decoderX_block(self, input_channels, output_channels, kernel_size=8, stride=4, output_padding=0,
                            final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride, output_padding=output_padding),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride, output_padding=output_padding),
                nn.Sigmoid(),
            )

    def forward(self, noise):

        x = torch.unsqueeze(noise, 2)
        x = self.gen(x)
        return x[:, :, :self.recon_length]

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + F.softplus(-2. * x) - math.log(2.0)
    return torch.sum(_log_cosh(y_pred - y_true))

# class MaskedDiff(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, recon, target, mask, order = 'l2'):
#         if order == 'l2':
#             loss = (torch.flatten(recon) - torch.flatten(target)) ** 2.0
#
#         elif order == 'l1':
#             loss = torch.flatten(recon) - torch.flatten(target)
#
#         else:
#             raise Exception
#
#         flat_mask = torch.flatten(mask)
#         assert(len(flat_mask) == len(loss))
#
#         loss_masked = loss * flat_mask
#         return loss_masked

class VAE(pl.LightningModule):
    def __init__(self, sample_ecg, loss_ratio=LOSS_RATIO, learning_rate=LEARNING_RATE):
        super().__init__()
        self.encoderX = EncoderX()
        self.decoderX = DecoderX()
        self.loss_ratio = loss_ratio
        self.learning_rate = learning_rate
        self.double()
        self.curr_device = None
        self.sample_ecg = sample_ecg

    def vae_stft(self, sig: torch.Tensor):
        if len(sig.shape) == 3:
            x = []
            for isig in range(sig.shape[0]):  # iterate through batch
                stft_image = torch.stft(sig[isig, ...], n_fft=32, normalized=True, hop_length=1,
                                        onesided=True, return_complex=True, center=False)

                x.append(torch.cat((stft_image.real, stft_image.imag), dim=1).unsqueeze(0))

            return torch.cat(x)

        else:
            stft_image = torch.stft(sig, n_fft=32, normalized=True, hop_length=1,
                                    onesided=True, return_complex=True, center=False)

            return torch.cat((stft_image.real, stft_image.imag), dim=1)

    def loss_function(self, results):
        # return PEAK_MASK_LOSS_FACTOR * torch.sum(results['loss_mse'] * results['peak_mask'])
        return 10 * results['loss_log_cosh'] + PEAK_MASK_LOSS_FACTOR * torch.sum(results['loss_mse'] * results['peak_mask']) / BATCH_SIZE

    def loss_function_ss(self, results, alpha):
        return self.loss_function(results) + alpha * ()

    @staticmethod
    def calc_mse(x_recon, x):
        return F.mse_loss(x_recon, x, reduction='none')

    @staticmethod
    def calc_mae(x_recon, x):
        return F.l1_loss(x_recon, x, reduction='none')

    @staticmethod
    def calc_logcosh(x_recon, x):
        return log_cosh_loss(x_recon, x)

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):

        fecg_sig, fecg_stft = x['fecg_sig'][:, :1, :], x['fecg_stft'][:, :1, :, :]
        mecg_sig, mecg_stft = x['mecg_sig'][:, :1, :], x['mecg_stft'][:, :1, :, :]
        offset = x['offset'][:, :1, :]
        peak_mask = x['fetal_mask'][:, :1, :]
        aecg_sig, aecg_stft = fecg_sig + mecg_sig - offset, fecg_stft + mecg_stft

        mean, std = self.encoderX(aecg_sig)
        z = Normal(mean, std)

        z_sample = z.rsample()  # Equivalent to torch.randn(z_dim) * stddev + mean
        x_recon = self.decoderX(z_sample)

        recon_loss_mse = self.calc_mse(self.vae_stft(x_recon), fecg_stft)
        recon_loss_mae = self.calc_mae(self.vae_stft(x_recon), fecg_stft)
        recon_loss_raw_mse = self.calc_mse(x_recon, fecg_sig - offset)
        recon_loss_raw_mae = self.calc_mae(x_recon, fecg_sig - offset)
        recon_log_cosh = self.calc_logcosh(x_recon, fecg_sig - offset)

        kl_loss = torch.mean(self.kl_divergence(z_sample, mean, std))

        return {'x_recon': x_recon, 'stft_loss_mse': torch.sum(recon_loss_mse), 'loss_log_cosh': recon_log_cosh,
                'kl_loss': kl_loss, 'stft_loss_mae': torch.sum(recon_loss_mae), 'loss_mse': torch.sum(recon_loss_raw_mse),
                'loss_mae': torch.sum(recon_loss_raw_mae), 'orig_fecg': fecg_sig, 'orig_mecg': mecg_sig,
                'peak_mask' : peak_mask, 'mu' : mean, 'sigma' : std}

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        d = self(batch)

        d['loss'] = self.loss_function(d)

        self.log_dict({f'train_{k}': v for k, v in d.items() if 'loss' in k}, sync_dist=True)

        return d['loss']

    def training_step_self_supervised(self, batch, batch_idx, optimizer_idx=0):
        # step 1: normal loss, recon
        # step 2: FIX_FECG_ALPHA
        # step 3: SWITCH_FECG_WINDOW_ALPHA
        # step 4: SWITCH_FECG_ALPHA
        pass

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        d = self(batch)
        d['loss'] = self.loss_function(d)

        self.log_dict({f'val_{k}': v for k, v in d.items()}, sync_dist=True)

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        curr_device = batch['fecg_sig'].device
        d = self(batch)
        d['loss'] = self.loss_function(d)

        self.log_dict({f'test_{k}': v for k, v in d.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_epoch_end(self, training_step_outputs):
        results = self(self.sample_ecg)
        recon = results['x_recon'].detach().numpy()
        orig_sig = self.sample_ecg['fecg_sig']  # self.sample_ecg['fecg_sig']

        plt.plot(orig_sig[0, 0, :], color='b')
        plt.plot(recon[0, 0, :], color='r')

        plt.show()

        super().training_epoch_end(training_step_outputs)