import math

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
NUM_STRIDES = (4,4,4,4)
NUM_KERNELS = (8,8,10,6)
DECODER_KERNELS = (6,10,8,8)
DECODER_STRIDES = (4,4,4,4)
NUM_PLANES = (16, 64, 128)

SKIP = False

assert len(NUM_PLANES) == len(NUM_BLOCKS)
START_CHANNELS = 34


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + F.softplus(-2. * x) - math.log(2.0)
    return _log_cosh(y_pred - y_true)


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

    def __init__(self, im_chan=1, z_dim=Z_DIM, hidden_dim=NUM_PLANES, kernels=NUM_KERNELS, strides=NUM_STRIDES, skip=SKIP):
        super(EncoderX, self).__init__()
        self.z_dim = z_dim
        self.skip = skip
        self.skip_out = [0, 0]
        self.encode_x = nn.Sequential(
            make_encoderX_block(im_chan, NUM_PLANES[0], kernel_size=kernels[0], stride=strides[0]),
            make_encoderX_block(NUM_PLANES[0], NUM_PLANES[1], kernel_size=kernels[1], stride=strides[1]),
            make_encoderX_block(NUM_PLANES[1], NUM_PLANES[2], kernel_size=kernels[2], stride=strides[2]),
            make_encoderX_block(NUM_PLANES[2], z_dim * 2, kernel_size=kernels[3], stride=strides[3], final_layer=True),
        )

    def forward(self, signal):
        for i, layer in enumerate(self.encode_x):
            encoding_x = layer(encoding_x if i else signal)
            if self.skip and i > 0 and i < len(self.encode_x) - 1:
                self.skip_out[i-1] = encoding_x
        encoding_x = encoding_x.view(len(encoding_x), -1)

        # The stddev output is treated as the log of the variance of the normal
        # distribution by convention and for numerical stability
        return encoding_x[:, :self.z_dim], encoding_x[:, self.z_dim:].exp()


class DecoderX(nn.Module):

    def __init__(self, im_chan=1, z_dim=Z_DIM, hidden_dim=NUM_PLANES, kernels=DECODER_KERNELS, strides=DECODER_STRIDES,
                 recon_length=500, skip=SKIP):
        super(DecoderX, self).__init__()
        self.skip = skip
        self.skip_in = [0, 0]
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
        for i, layer in enumerate(self.gen):
            if self.skip and i > 0 and i < len(self.gen) - 1:
                x = layer(x + self.skip_in[2 - i])
            else:
                x = layer(x)

        return x[:, :, :self.recon_length]


class VAE(pl.LightningModule):
    def __init__(self, loss_ratio=LOSS_RATIO, learning_rate=LEARNING_RATE, sample_ecgs=[], mode='extract_fecg'):
        super().__init__()
        self.encoderX = EncoderX()
        self.decoderX = DecoderX()
        self.loss_ratio = loss_ratio
        self.learning_rate = learning_rate
        self.double()
        self.curr_device = None
        self.sample_ecgs = sample_ecgs
        self.mode = mode

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
        if self.mode == 'recreate':
            return results['logcosh_loss'] + results['kl_loss'] * 0.1
        return results['loss_mse'] * 500 + results['kl_loss'] * 50

    @staticmethod
    def calc_mse(x_recon, x, mask=1):
        return torch.mean(F.mse_loss(x_recon, x, reduction='sum') * mask)

    @staticmethod
    def calc_mae(x_recon, x):
        return torch.mean(F.l1_loss(x_recon, x, reduction='sum'))

    @staticmethod
    def calc_logcosh(x_recon, x):
        return torch.mean(log_cosh_loss(x_recon, x))

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
        if self.mode == 'recreate':
            mean, std = self.encoderX(F.pad(x, (0, 0)))
            z = Normal(mean, std)
            z_sample = z.rsample()  # Equivalent to torch.randn(z_dim) * stddev + mean
            self.decoderX.skip_in = self.encoderX.skip_out
            x_recon = self.decoderX(z_sample)
            recon_loss_logcosh = self.calc_logcosh(x_recon, x)
            kl_loss = torch.mean(self.kl_divergence(z_sample, mean, std))
            return {'logcosh_loss': recon_loss_logcosh, 'kl_loss': kl_loss}


        maternal_mask = x['maternal_mask'][:, :1, :]
        fetal_mask = x['fetal_mask'][:, :1, :]
        fecg_sig, fecg_stft = x['fecg_sig'][:, :1, :], x['fecg_stft'][:, :1, :, :]
        mecg_sig, mecg_stft = x['mecg_sig'][:, :1, :], x['mecg_stft'][:, :1, :, :]
        aecg_sig, aecg_stft = fecg_sig + mecg_sig, fecg_stft + mecg_stft
        # mean, std = self.encoderX(F.pad(aecg_sig, (30, 30)))
        mean, std = self.encoderX(F.pad(aecg_sig, (0, 0)))
        z = Normal(mean, std)

        z_sample = z.rsample()  # Equivalent to torch.randn(z_dim) * stddev + mean
        self.decoderX.skip_in = self.encoderX.skip_out
        x_recon = self.decoderX(z_sample)

        recon_loss_mse = self.calc_mse(self.stft(x_recon), fecg_stft)
        recon_loss_mae = self.calc_mae(self.stft(x_recon), fecg_stft)
        # recon_loss_raw_mse = self.calc_mse(x_recon, fecg_sig, fetal_mask + 1)
        recon_loss_raw_mse = self.calc_logcosh(x_recon, aecg_sig)
        recon_loss_raw_mae = self.calc_mae(x_recon, fecg_sig)
        kl_loss = torch.mean(self.kl_divergence(z_sample, mean, std))

        return {'x_recon': x_recon, 'stft_loss_mse': recon_loss_mse,
                'kl_loss': kl_loss, 'stft_loss_mae': recon_loss_mae, 'loss_mse': recon_loss_raw_mse,
                'loss_mae': recon_loss_raw_mae, 'maternal_mask': maternal_mask, 'fetal_mask': fetal_mask}

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
