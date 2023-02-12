import torch
from torch import nn, optim
import pytorch_lightning as pl
from .network_modules import Encoder, Decoder, PeakHead, KeyProjector
from .losses import *

class UNet(pl.LightningModule):
    def __init__(self, sample_ecg : torch.Tensor, learning_rate : float, fecg_down_params : ((int,),),
                 fecg_up_params : ((int,),), loss_ratios : {str : int}, batch_size : int, decoder_skips : bool,
                 initial_conv_planes : int, linear_layers : (int,), pad_length : int, embed_dim : int):
        # params in the format ((num_planes), (kernel_width), (stride))
        super().__init__()

        self.learning_rate = learning_rate
        self.sample_ecg = sample_ecg

        self.fecg_encode = Encoder(fecg_down_params)

        self.fecg_decode = Decoder(fecg_up_params, ('tanh',), skips=decoder_skips)

        # for pretraining purposes, otherwise is just another conv layer
        self.value_key_proj = KeyProjector(fecg_down_params[0][-1], embed_dim)
        self.value_unprojer = KeyProjector(embed_dim, fecg_up_params[0][0])

        self.fecg_peak_head = PeakHead(starting_planes=fecg_down_params[0][-1], ending_planes=initial_conv_planes,
                                       hidden_layers=linear_layers, output_length=pad_length)

        self.loss_params, self.batch_size = loss_ratios, batch_size
        # change dtype
        self.float()

    def loss_function(self, results):
        # return all the losses with hyperparameters defined earlier
        return self.loss_params['fecg'] * torch.sum(results['loss_fecg_mse']) / self.batch_size +\
               self.loss_params['fecg_peak'] * torch.sum(results['loss_peaks_mse']) / self.batch_size

    def calculate_losses_into_dict(self, recon_fecg: torch.Tensor, gt_fecg: torch.Tensor, recon_peaks: torch.Tensor,
                                   gt_fetal_peaks: torch.Tensor) -> {str: torch.Tensor}:
        # If necessary: peak-weighted losses, class imablance in BCE loss

        fecg_loss_mse = calc_mse(recon_fecg, gt_fecg)
        # fecg_mask_loss_bce = calc_bce_loss(recon_binary_fetal_mask, gt_binary_fetal_mask)
        # masked loss to weigh peaks on the bce loss
        # fecg_mask_loss_masked_bce = calc_bce_loss(recon_binary_fetal_mask * gt_binary_fetal_mask, gt_binary_fetal_mask)

        peak_loss_mse = calc_mse(recon_peaks, gt_fetal_peaks)

        aggregate_loss = {'loss_fecg_mse': fecg_loss_mse, 'loss_peaks_mse': peak_loss_mse}

        loss_dict = {}

        # calculate loss with loss weights
        loss_dict['total_loss'] = self.loss_function(aggregate_loss)

        # loss from array to scalar
        loss_dict.update({k: torch.sum(loss) / self.batch_size for k, loss in aggregate_loss.items()})

        return loss_dict

    def remap_input(self, x):
        # print(x.keys())
        fecg_sig, mecg_sig = x['fecg_sig'][:, :1, :], x['mecg_sig'][:, :1, :]
        offset, noise = x['offset'][:, :1, :], x['noise'][:, :1, :]
        aecg_sig = fecg_sig + mecg_sig - offset + noise
        fecg_peaks = x['fecg_peaks'][:,0,:]

        binary_fetal_mask = x['binary_fetal_mask'][:, :1, :]
        binary_maternal_mask = x['binary_maternal_mask'][:, :1, :]

        d = {'aecg_sig': aecg_sig, 'binary_fetal_mask': binary_fetal_mask,
             'binary_maternal_mask': binary_maternal_mask, 'fecg_peaks' : fecg_peaks,
             'fecg_sig': fecg_sig, 'mecg_sig': mecg_sig, 'offset': offset}

        for k, v in d.items():
            d[k] = v.type(torch.float32)

        return d

    def forward(self, aecg_sig):

        # fecg_encode_outs : [fecg_sig (template), layer1, layer2, ..., layern, inner_layer]
        fecg_encode_outs = self.fecg_encode(aecg_sig, None)

        value_proj = self.value_key_proj(fecg_encode_outs[-1])
        value_unproj = self.value_unprojer(value_proj)

        (fecg_recon,), _ = self.fecg_decode(value_unproj, None)

        fecg_peak_recon = self.fecg_peak_head(value_unproj)

        return {'fecg_recon' : fecg_recon, 'fecg_peak_recon' : fecg_peak_recon}

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['fecg_sig'],
                                                    model_outputs['fecg_peak_recon'], d['fecg_peaks'])

        self.log_dict({f'train_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)

        return loss_dict['total_loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0, log=False):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['fecg_sig'],
                                                    model_outputs['fecg_peak_recon'], d['fecg_peaks'])

        self.log_dict({f'val_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)
        model_outputs.update(loss_dict)

        return model_outputs

    def test_step(self, batch, batch_idx, optimizer_idx=0, log=False):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['fecg_sig'],
                                                    model_outputs['fecg_peak_recon'], d['fecg_peaks'])

        model_outputs.update(loss_dict)

        return model_outputs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def print_summary(self):
        from torchinfo import summary
        random_input = torch.rand((self.batch_size, 1, 250))
        return summary(self, input_data=random_input, depth=7)