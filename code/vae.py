import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.normal import Normal

import pytorch_lightning as pl


LOG_STEPS = 10
LEARNING_RATE = 1e-4
Z_DIM = 64
LOSS_RATIO = 1000

NUM_BLOCKS = (8,8,8)
NUM_STRIDES = (4,6,4,2)
NUM_KERNELS = (8,8,8,4)
DECODER_KERNELS = (7,8,8,6)
DECODER_STRIDES = (4,4,4,4)
NUM_PLANES = (16, 64, 128)

assert len(NUM_PLANES) == len(NUM_BLOCKS)
START_CHANNELS = 34


def make_encoderX_block(input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

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


class EncoderX(nn.Module):

    def __init__(self, im_chan=1, z_dim=Z_DIM, hidden_dim=NUM_PLANES, kernels=NUM_KERNELS, strides=NUM_STRIDES):
        super(EncoderX, self).__init__()
        self.z_dim = z_dim
        self.encode_x = nn.Sequential(
            make_encoderX_block(im_chan, NUM_PLANES[0], kernel_size=kernels[0], stride=strides[0]),
            make_encoderX_block(NUM_PLANES[0], NUM_PLANES[1], kernel_size=kernels[1], stride=strides[1]),
            make_encoderX_block(NUM_PLANES[1], NUM_PLANES[2], kernel_size=kernels[2], stride=strides[2]),
            make_encoderX_block(NUM_PLANES[2], z_dim * 2, kernel_size=kernels[3], stride=strides[3], final_layer=True),
        )

    def forward(self, signal):
        encoding_x = self.encode_x(signal)
        encoding_x = encoding_x.view(len(encoding_x), -1)

        # The stddev output is treated as the log of the variance of the normal
        # distribution by convention and for numerical stability
        return encoding_x[:, :self.z_dim], encoding_x[:, self.z_dim:].exp()


class DecoderX(nn.Module):

    def __init__(self, im_chan=1, z_dim=Z_DIM, hidden_dim=NUM_PLANES, kernels=DECODER_KERNELS, strides=DECODER_STRIDES,
                 recon_length=500):
        super(DecoderX, self).__init__()
        self.z_dim, self.recon_length = z_dim, recon_length
        #         self.gen = nn.Sequential(
        #             self.make_decoderX_block(z_dim, NUM_PLANES[-1], kernel_size=kernels[-1], stride=strides[-1]),
        #             self.make_decoderX_block(NUM_PLANES[-1], NUM_PLANES[-2], kernel_size=kernels[-2], stride=strides[-2]),
        #             self.make_decoderX_block(NUM_PLANES[-2], NUM_PLANES[-3], output_padding=1, kernel_size=kernels[-3], stride=strides[-3]),
        #             self.make_decoderX_block(NUM_PLANES[-3], im_chan, kernel_size=kernels[-4]+4, stride=strides[-4], final_layer=True),
        #         )

        self.gen = nn.Sequential(
            self.make_decoderX_block(z_dim, NUM_PLANES[-1], kernel_size=kernels[0], stride=strides[0]),
            self.make_decoderX_block(NUM_PLANES[-1], NUM_PLANES[-2], kernel_size=kernels[1], stride=strides[1]),
            self.make_decoderX_block(NUM_PLANES[-2], NUM_PLANES[-3], output_padding=0, kernel_size=kernels[2],
                                     stride=strides[2]),
            self.make_decoderX_block(NUM_PLANES[-3], im_chan, kernel_size=kernels[3], stride=strides[3],
                                     final_layer=True),
        )

    def make_decoderX_block(self, input_channels, output_channels, kernel_size=8, stride=4, output_padding=0,
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
                nn.Tanh(),
            )

    def forward(self, noise):

        x = torch.unsqueeze(noise, 2)
        x = self.gen(x)
        return x[:, :, :self.recon_length]


class VAE(pl.LightningModule):
    def __init__(self, loss_ratio=LOSS_RATIO, learning_rate=LEARNING_RATE, sample_ecgs=[]):
        super().__init__()
        self.encoderX = EncoderX()
        self.decoderX = DecoderX()
        self.loss_ratio = loss_ratio
        self.learning_rate = learning_rate
        self.double()
        self.curr_device = None
        self.sample_ecgs = sample_ecgs

    def stft(self, sig: torch.Tensor):
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
        return results['loss_mse']

    @staticmethod
    def calc_mse(x_recon, x):
        return torch.mean(F.mse_loss(x_recon, x, reduction='sum'))

    @staticmethod
    def calc_mae(x_recon, x):
        return torch.mean(F.l1_loss(x_recon, x, reduction='sum'))

    @staticmethod
    def kl_divergence(z, mu, std):
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
        aecg_sig, aecg_stft = fecg_sig + mecg_sig, fecg_stft + mecg_stft

        mean, std = self.encoderX(aecg_sig)
        z = Normal(mean, std)

        z_sample = z.rsample()  # Equivalent to torch.randn(z_dim) * stddev + mean
        x_recon = self.decoderX(z_sample)

        recon_loss_mse = self.calc_mse(self.stft(x_recon), mecg_stft)
        recon_loss_mae = self.calc_mae(self.stft(x_recon), mecg_stft)
        recon_loss_raw_mse = self.calc_mse(x_recon, mecg_sig)
        recon_loss_raw_mae = self.calc_mae(x_recon, mecg_sig)
        kl_loss = torch.mean(self.kl_divergence(z_sample, mean, std))

        return {'x_recon': x_recon, 'stft_loss_mse': recon_loss_mse,
                'kl_loss': kl_loss, 'stft_loss_mae': recon_loss_mae, 'loss_mse': recon_loss_raw_mse,
                'loss_mae': recon_loss_raw_mae}

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        d = self(batch)
        d['loss'] = self.loss_function(d)

        self.log_dict({f'train_{k}': v for k, v in d.items() if 'loss' in k}, sync_dist=True)

        return d['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        d = self(batch)
        d['loss'] = self.loss_function(d)

        self.log_dict({f'val_{k}': v for k, v in d.items()}, sync_dist=True)

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        d = self(batch)
        d['loss'] = self.loss_function(d)

        self.log_dict({f'test_{k}': v for k, v in d.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_epoch_end(self, training_step_outputs):
        if self.sample_ecgs:
            fig, axs = plt.subplots(nrows=len(self.sample_ecgs), ncols=1)

            for i, sample_ecg in enumerate(self.sample_ecgs):
                results = self(sample_ecg)
                recon = results['x_recon'].detach().numpy()
                orig_sig = self.sample_ecg['mecg_sig']  # self.sample_ecg['fecg_sig']

                axs[i, 0].plot(orig_sig[0, 0, :], color='b')
                axs[i, 0].plot(recon[0, 0, :], color='r')

            fig.show()

        super().training_epoch_end(training_step_outputs)
