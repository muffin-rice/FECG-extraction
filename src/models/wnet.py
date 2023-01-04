import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from .network_modules import Encoder, Decoder

class WNet(pl.LightningModule):
    def __init__(self, sample_ecg : torch.Tensor, learning_rate : float, mecg_down_params : ((int,),),
                 mecg_up_params : ((int,),), fecg_down_params : ((int,),), fecg_up_params : ((int,),), loss_ratios : {str : int},
                 batch_size : int):
        # params in the format ((num_planes), (kernel_width), (stride))
        super().__init__()

        self.learning_rate = learning_rate
        self.sample_ecg = sample_ecg

        self.mecg_encode = Encoder(mecg_down_params, encoder_skip=False)

        self.mecg_decode = Decoder(mecg_up_params, ('sigmoid',))

        # no peak head for mecg – focus entirely on the reconstruction
        # self.mecg_peak_head = nn.Sequential(
        #     nn.ConvTranspose1d(mecg_up_params[0][-2], 1, mecg_up_params[1][-1], mecg_up_params[2][-1], output_padding=1),
        #     nn.Sigmoid(),
        # )

        self.fecg_encode = Encoder(fecg_down_params)

        self.fecg_decode = Decoder(fecg_up_params, ('sigmoid', 'tanh'))

        self.loss_params, self.batch_size = loss_ratios, batch_size

        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        # change dtype
        self.float()

    def make_skip_connection(self, a, b):
        '''makes a skip connection and applies leaky relu'''
        b_shape = b.shape
        return self.leaky(a[:, :, :b_shape[2]]+b)

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
               self.loss_params['mecg'] * torch.sum(results['loss_mecg_mse']) / self.batch_size + \
               self.loss_params['fecg'] * torch.sum(results['loss_fecg_mae']) / self.batch_size + \
               self.loss_params['fp_bce'] * self.loss_params['fp_bce_class'] * torch.sum(results['loss_fecg_mask_masked_bce'])

    def calculate_losses_into_dict(self, recon_fecg, gt_fecg, recon_mecg, gt_mecg, recon_binary_fetal_mask,
                                   gt_binary_fetal_mask, offset):

        # If necessary: peak-weighted losses, class imablance in BCE loss

        fecg_loss_mae = self.calc_mae(recon_fecg, gt_fecg)
        fecg_mask_loss_bce = self.calc_bce_loss(recon_binary_fetal_mask, gt_binary_fetal_mask)
        mecg_loss_mse = self.calc_mse(recon_mecg+offset, gt_mecg)
        fecg_mask_loss_masked_bce = self.calc_bce_loss(recon_binary_fetal_mask * gt_binary_fetal_mask, gt_binary_fetal_mask)

        aggregate_loss = {'loss_fecg_mae' : fecg_loss_mae, 'loss_fecg_mask_bce' : fecg_mask_loss_bce,
                          'loss_mecg_mse' : mecg_loss_mse, 'loss_fecg_mask_masked_bce' : fecg_mask_loss_masked_bce}

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
        aecg_sig = fecg_sig + mecg_sig - offset + noise

        binary_fetal_mask = x['binary_fetal_mask'][:, :1, :]
        binary_maternal_mask = x['binary_maternal_mask'][:, :1, :]

        d = {'aecg_sig': aecg_sig, 'binary_fetal_mask': binary_fetal_mask,
             'binary_maternal_mask': binary_maternal_mask, 'fetal_mask': fetal_mask, 'maternal_mask': maternal_mask,
             'fecg_sig': fecg_sig, 'mecg_sig': mecg_sig, 'offset' : offset}

        for k, v in d.items():
            d[k] = v.type(torch.float32)

        return d

    def forward(self, aecg_sig):

        # mecg_encode_outs: [aecg, layer1, layer2, ..., encoded_layer]
        mecg_encode_outs = self.mecg_encode(aecg_sig, None)

        # decoder outputs mecg recon
        (mecg_recon,), mecg_decode_skips = self.mecg_decode(mecg_encode_outs[-1], mecg_encode_outs)

        # fecg_encode_outs : [fecg_sig (noisy), layer1, layer2, ..., layern, encoded_layer]
        noisy_fecg = aecg_sig - mecg_recon
        fecg_encode_outs = self.fecg_encode(noisy_fecg, None)

        (fecg_peak_recon, fecg_recon), fecg_decode_skips = self.fecg_decode(fecg_encode_outs[-1], fecg_encode_outs)

        return {'mecg_recon' : mecg_recon, 'fecg_recon' : fecg_recon, 'fecg_mask_recon' : fecg_peak_recon}

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['fecg_sig'], model_outputs['mecg_recon'],
                                                    d['mecg_sig'], model_outputs['fecg_mask_recon'], d['binary_fetal_mask'],
                                                    d['offset'])

        self.log_dict({f'train_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)

        return loss_dict['total_loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0, log=False):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['fecg_sig'], model_outputs['mecg_recon'],
                                                    d['mecg_sig'], model_outputs['fecg_mask_recon'], d['binary_fetal_mask'],
                                                    d['offset'])

        self.log_dict({f'val_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)
        model_outputs.update(loss_dict)

        return model_outputs

    def test_step(self, batch, batch_idx, optimizer_idx=0, log=False):
        d = self.remap_input(batch)
        model_outputs = self.forward(d['aecg_sig'])

        loss_dict = self.calculate_losses_into_dict(model_outputs['fecg_recon'], d['fecg_sig'], model_outputs['mecg_recon'],
                                                    d['mecg_sig'], model_outputs['fecg_mask_recon'], d['binary_fetal_mask'],
                                                    d['offset'])

        model_outputs.update(loss_dict)

        return model_outputs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def print_summary(self):
        from torchinfo import summary
        random_input = torch.rand((2, 1, 500))
        return summary(self, input_data=random_input, depth=7)