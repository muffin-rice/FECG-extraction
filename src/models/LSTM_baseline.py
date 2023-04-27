import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from .network_modules import Encoder, KeyProjector, Decoder#, PeakHead
from numpy import sqrt
from .losses import *
from .unet import UNet

class LSTM_baseline(pl.LightningModule):
    def __init__(self, sample_ecg, window_length, embed_dim : int, value_encoder_params : ((int,),),
                 decoder_params : ((int,),), batch_size : int, learning_rate : float, loss_ratios : {str : int},
                 pretrained_unet : UNet, decoder_skips : bool, num_layers : int):
        super().__init__()
        self.window_length = window_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_params = loss_ratios

        if pretrained_unet is not None:
            self.encoder = pretrained_unet.fecg_encode
            self.decoder = pretrained_unet.fecg_decode
            self.key_proj = pretrained_unet.value_key_proj
            self.unprojer = pretrained_unet.value_unprojer

            print('Using pretrained value encoder/decoder')
        else:
            self.decoder = Decoder(decoder_params, head_params=('tanh', 'sigmoid'), skips=decoder_skips)
            self.encoder = Encoder(value_encoder_params)

            self.key_proj = KeyProjector(value_encoder_params[0][-1], embed_dim)
            self.unprojer = KeyProjector(embed_dim, decoder_params[0][0])

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=int(embed_dim/2), num_layers=num_layers, batch_first = True,
                            bidirectional=True, )

    def forward(self, aecg_sig : torch.Tensor, hidden_pair : (torch.Tensor, torch.Tensor) = None):
        '''data is in the format of B x num_windows x window_size'''
        fecg_recon = torch.zeros_like(aecg_sig).to(self.device)
        fecg_peak_recon = torch.zeros_like(aecg_sig).to(self.device)
        for i in range(aecg_sig.shape[1]):
            encoding_outs = self.encoder.forward(aecg_sig[:,[i],:])
            proj_encoding = self.key_proj(encoding_outs[-1])

            if hidden_pair is None:
                pp_encoding, hidden_pair = self.lstm.forward(proj_encoding.transpose(1,2))

            else:
                pp_encoding, hidden_pair = self.lstm.forward(proj_encoding.transpose(1,2), hidden_pair)

            unproj_encoding = self.unprojer(pp_encoding.transpose(1,2))
            decoded_seq, _ = self.decoder.forward(unproj_encoding, encoding_outs)

            fecg_recon[:,i,:] = decoded_seq[0][:, 0, :self.window_length]
            fecg_peak_recon[:,i,:] = decoded_seq[1][:,0,:self.window_length]

        return {'fecg_recon' : fecg_recon, 'fecg_peak_recon' : fecg_peak_recon}

    def train_forward(self, aecg_sig : torch.Tensor):
        return self.forward(aecg_sig)

    def loss_function(self, results):
        # return all the losses with hyperparameters defined earlier
        return self.loss_params['fecg'] * torch.sum(results['loss_fecg_mse']) / self.batch_size + \
               self.loss_params['fecg_peak'] * torch.sum(results['loss_peaks_bce']) / self.batch_size + \
               self.loss_params['fecg_peak_mask'] * self.loss_params['fecg_peak'] * torch.sum(results['loss_peaks_bce_masked']) / self.batch_size
        # self.loss_params['fecg_peak'] * torch.sum(results['loss_peaks_mse']) / self.batch_size

    def calculate_losses_into_dict(self, recon_fecg: torch.Tensor, gt_fecg: torch.Tensor, recon_peaks: torch.Tensor,
                                   gt_fetal_peaks: torch.Tensor) -> {str: torch.Tensor}:
        # If necessary: peak-weighted losses, class imablance in BCE loss

        assert torch.any(gt_fetal_peaks > 0), 'The binary fetal mask is all zeros.'

        fecg_loss_mse = calc_mse(recon_fecg, gt_fecg)
        fecg_mask_loss_bce = calc_bce_loss(recon_peaks, gt_fetal_peaks)
        # masked loss to weigh peaks on the bce loss
        fecg_mask_loss_masked_bce = calc_bce_loss(recon_peaks * gt_fetal_peaks, gt_fetal_peaks)

        # peak_loss_mse = calc_mse(recon_peaks, gt_fetal_peaks)

        aggregate_loss = {'loss_fecg_mse': fecg_loss_mse, 'loss_peaks_bce': fecg_mask_loss_bce,
                          'loss_peaks_bce_masked': fecg_mask_loss_masked_bce}

        loss_dict = {}

        # calculate loss with loss weights
        loss_dict['total_loss'] = self.loss_function(aggregate_loss)

        # loss from array to scalar
        loss_dict.update({k: torch.sum(loss) / self.batch_size for k, loss in aggregate_loss.items()})

        return loss_dict

    def convert_to_float(self, d: {}):
        # TODO: make better solution
        for k, v in d.items():
            if 'Tensor' in str(type(v)):
                d[k] = v.float()

    def training_step(self, d : {}, batch_idx):
        self.convert_to_float(d)
        aecg_sig = d['mecg_sig'] + d['fecg_sig'] + d['noise']
        # performs backwards on the last segment only to avoid inplace operations with the masking
        # self.peak_shape = d['fecg_peaks'].shape
        model_output = self.train_forward(aecg_sig)

        try:
            loss_dict = self.calculate_losses_into_dict(model_output['fecg_recon'], d['fecg_sig'],
                                                        model_output['fecg_peak_recon'],
                                                        d['binary_fetal_mask'])  # d['fecg_peaks'][:,-1,:])
        except Exception as e:
            print(f'{d["fname"]} failed somehow:')
            import pickle as pkl
            with open('dev/fails.pkl', 'wb') as f:
                pkl.dump(d, f)
            raise e

        self.log_dict({f'train_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)

        return loss_dict['total_loss']

    def validation_step(self, d : {}, batch_idx):
        self.memory_initialized = False
        self.convert_to_float(d)
        aecg_sig = d['mecg_sig'] + d['fecg_sig'] + d['noise']
        # self.peak_shape = d['fecg_peaks'].shape
        model_output = self.forward(aecg_sig)

        loss_dict = self.calculate_losses_into_dict(model_output['fecg_recon'][:,-1,:], d['fecg_sig'][:,-1,:],
                                                    model_output['fecg_peak_recon'][:,-1,:],  d['binary_fetal_mask'][:,-1,:]) # d['fecg_peaks'])

        self.log_dict({f'val_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)
        model_output.update(loss_dict)

        return model_output

    def test_step(self, d : {}, batch_idx):
        self.memory_initialized = False
        self.convert_to_float(d)
        aecg_sig = d['mecg_sig'] + d['fecg_sig'] + d['noise']
        # self.peak_shape = d['fecg_peaks'].shape
        model_output = self.forward(aecg_sig)

        loss_dict = self.calculate_losses_into_dict(model_output['fecg_recon'][:,-1,:], d['fecg_sig'][:,-1,:],
                                                    model_output['fecg_peak_recon'][:,-1,:], d['binary_fetal_mask'][:,-1,:]) # d['fecg_peaks'])

        model_output.update(loss_dict)

        return model_output

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def print_summary(self, depth = 7):
        from torchinfo import summary
        random_input = torch.rand((self.batch_size, 5, 250)) # window len 5 will make summary long, change to 1 if too long
        self.peak_shape = (self.batch_size, 5, self.pad_length) # only a single peak for the entire window
        return summary(self, input_data=random_input, depth=depth)

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size