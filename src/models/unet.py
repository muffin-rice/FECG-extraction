import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal
import pytorch_lightning as pl
import math
import numpy as np
import matplotlib.pyplot as plt

class UNet(pl.LightningModule):
    def __init__(self, sample_ecg : torch.Tensor, learning_rate : float, fecg_down_params : ((int,),),
                 fecg_up_params : ((int,),), loss_ratios : {str : int}, batch_size : int):
        # params in the format ((num_planes), (kernel_width), (stride))
        super().__init__()

        self.learning_rate = learning_rate
        self.sample_ecg = sample_ecg

        self.fecg_encode = nn.ParameterList()
        for i in range(len(fecg_down_params[0])-2):
            self.fecg_encode.append(self.make_encoder_block(fecg_down_params[0][i], fecg_down_params[0][i + 1],
                                                            kernel_size=fecg_down_params[1][i], stride=fecg_down_params[2][i]))

        self.fecg_encode.append(self.make_encoder_block(fecg_down_params[0][-2], fecg_down_params[0][-1],
                                                        kernel_size=fecg_down_params[1][-1], stride=fecg_down_params[2][-1]))

        self.fecg_decode = nn.ParameterList()
        for i in range(len(fecg_up_params[0])-2):
            self.fecg_decode.append(self.make_decoder_block(fecg_up_params[0][i], fecg_up_params[0][i + 1],
                                                            kernel_size=fecg_up_params[1][i], stride=fecg_up_params[2][i]))

        self.fecg_decode_final = self.make_decoder_block(fecg_up_params[0][-2], 1, fecg_up_params[1][-1],
                                                        fecg_up_params[2][-1], final_layer=True)

        self.fecg_peak_head = nn.Sequential(
            nn.ConvTranspose1d(fecg_up_params[0][-2], 1, fecg_up_params[1][-1], fecg_up_params[2][-1], output_padding=1),
            nn.Sigmoid(),
        )

        self.loss_params, self.batch_size = loss_ratios, batch_size

        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        # change dtype
        self.float()

    def make_decoder_block(self, input_channels, output_channels, kernel_size=8, stride=4, output_padding=1,
                           final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride,
                                   output_padding=min(output_padding, stride - 1)),
                nn.BatchNorm1d(output_channels),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm1d(output_channels),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride, output_padding=output_padding),
                nn.Tanh(),
            )

    def make_encoder_block(self, input_channels : int, output_channels : int, kernel_size : int = 4, stride : int = 2,
                           final_layer : bool = False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm1d(output_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm1d(output_channels),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
            )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def make_skip_connection(self, a, b):
        b_shape = b.shape
        return self.leaky(a[:b_shape[0], :b_shape[1], :b_shape[2]]+b)

    @staticmethod
    def calc_mse(x_recon, x):
        return F.mse_loss(x_recon, x, reduction='none')

    @staticmethod
    def calc_mae(x_recon, x):
        return F.l1_loss(x_recon, x, reduction='none')

    @staticmethod
    def calc_peak_loss(mse, mask):
        return mse * mask

    @staticmethod
    def calc_bce_loss(mask_recon, mask):
        return F.binary_cross_entropy(mask_recon, mask, reduction='none')

    def loss_function(self, results):
        # return all the losses with hyperparameters defined earlier
        return self.loss_params['fp_bce'] * torch.sum(results['loss_fecg_mask_bce']) / self.batch_size + \
               self.loss_params['fecg'] * torch.sum(results['loss_fecg_mae']) / self.batch_size + \
               self.loss_params['fp_bce'] * self.loss_params['fp_bce_class'] * torch.sum(results['loss_fecg_mask_masked_bce'])

    def calculate_losses_into_dict(self, recon_fecg, gt_fecg, recon_binary_fetal_mask,
                                   gt_binary_fetal_mask):

        # If necessary: peak-weighted losses, class imablance in BCE loss

        fecg_loss_mae = self.calc_mse(recon_fecg, gt_fecg)
        fecg_mask_loss_bce = self.calc_bce_loss(recon_binary_fetal_mask, gt_binary_fetal_mask)
        # masked loss to weigh peaks on the bce loss
        fecg_mask_loss_masked_bce = self.calc_bce_loss(recon_binary_fetal_mask * gt_binary_fetal_mask, gt_binary_fetal_mask)

        aggregate_loss = {'loss_fecg_mae' : fecg_loss_mae, 'loss_fecg_mask_bce' : fecg_mask_loss_bce,
                          'loss_fecg_mask_masked_bce' : fecg_mask_loss_masked_bce}

        loss_dict = {}

        # calculate loss with loss weights
        loss_dict['total_loss'] = self.loss_function(aggregate_loss)

        # loss from array to scalar
        loss_dict.update({k: torch.sum(loss) / self.batch_size for k, loss in aggregate_loss.items()})

        return loss_dict

    def remap_input(self, x):
        fecg_sig, mecg_sig = x['fecg_sig'][:, :1, :], x['mecg_sig'][:, :1, :]
        offset, noise = x['offset'][:, :1, :], x['noise'][:, :1, :]
        fetal_mask, maternal_mask = x['fetal_mask'][:, :1, :], x['maternal_mask'][:, :1, :]
        aecg_sig = fecg_sig + mecg_sig - offset
        gt_fecg_sig = x[f'gt_fecg_sig'][:, :1, :]

        binary_fetal_mask = x['binary_fetal_mask'][:, :1, :]
        binary_maternal_mask = x['binary_maternal_mask'][:, :1, :]

        d = {'aecg_sig' : aecg_sig, 'binary_fetal_mask' : binary_fetal_mask,
             'binary_maternal_mask' : binary_maternal_mask, 'gt_fecg_sig' : gt_fecg_sig,
             'fetal_mask' : fetal_mask, 'maternal_mask' : maternal_mask, 'fecg_sig' : fecg_sig,
             'mecg_sig' : mecg_sig}

        for k, v in d.items():
            d[k] = v.type(torch.float32)

        return d


    def forward(self, aecg_sig):

        # fecg_encode_outs : [fecg_sig (template), layer1, layer2, ..., layern]
        fecg_encode_outs = [aecg_sig]
        for i in range(len(self.fecg_encode)-1):
            encoder_output = self.fecg_encode[i](fecg_encode_outs[-1])
            # fecg_encode_outs.append(self.make_skip_connection(encoder_output, mecg_nonskip_outs[-i-1]))
            fecg_encode_outs.append(encoder_output)

        inner_layer = self.fecg_encode[-1](fecg_encode_outs[-1])

        # fecg_decode_outs: [encoded_layer, layern, ..., layer1]
        fecg_decode_outs = [inner_layer]
        for i, layer in enumerate(self.fecg_decode):
            decoder_output = layer(fecg_decode_outs[-1])
            fecg_decode_outs.append(self.make_skip_connection(-decoder_output, fecg_encode_outs[-i-1]))

        fecg_peak_recon = self.fecg_peak_head(fecg_decode_outs[-1])[:, :, :500]
        fecg_recon = self.fecg_decode_final(fecg_decode_outs[-1])[:, :, :500]

        return {'fecg_recon' : fecg_recon, 'fecg_mask_recon' : fecg_peak_recon}

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['gt_fecg_sig'],
                                                    model_outputs['fecg_mask_recon'], d['binary_fetal_mask'])

        self.log_dict({f'train_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)

        return loss_dict['total_loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0, log=False):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['gt_fecg_sig'],
                                                    model_outputs['fecg_mask_recon'], d['binary_fetal_mask'])

        self.log_dict({f'val_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)
        model_outputs.update(loss_dict)

        return model_outputs

    def test_step(self, batch, batch_idx, optimizer_idx=0, log=False):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['gt_fecg_sig'],
                                                    model_outputs['fecg_mask_recon'], d['binary_fetal_mask'])

        model_outputs.update(loss_dict)

        return model_outputs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def print_summary(self):
        from torchinfo import summary
        random_input = torch.rand((2, 1, 500))
        return summary(self, input_data=random_input, depth=7)