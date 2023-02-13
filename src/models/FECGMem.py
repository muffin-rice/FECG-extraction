import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from .network_modules import Encoder, KeyProjector, Decoder, PeakHead
from numpy import sqrt
from .losses import *
from .unet import UNet

class FECGMem(pl.LightningModule):
    '''FECG with memory storage'''
    def __init__(self, sample_ecg, window_length, query_encoder_params : ((int,),), embed_dim : int,
                 value_encoder_params : ((int,),), decoder_params : ((int,),), memory_length : int,
                 batch_size : int, learning_rate : float, loss_ratios : {str : int}, pretrained_unet : UNet,
                 decoder_skips : bool, initial_conv_planes : int, linear_layers : (int,), pad_length : int):
        super().__init__()
        self.window_length = window_length
        if pretrained_unet is not None:
            self.value_encoder = pretrained_unet.fecg_encode
            self.value_decoder = pretrained_unet.fecg_decode
            self.fecg_peak_head = pretrained_unet.fecg_peak_head
            self.value_key_proj = pretrained_unet.value_key_proj
            self.value_unprojer = pretrained_unet.value_unprojer

            # TODO: freeze or not
            # self.value_encoder.freeze()
            # self.value_decoder.freeze()

            print('Using pretrained value encoder/decoder')
        else:
            self.value_decoder = Decoder(decoder_params, head_params=('tanh',), skips=decoder_skips)
            self.value_encoder = Encoder(value_encoder_params)
            self.fecg_peak_head = PeakHead(starting_planes=embed_dim, ending_planes=initial_conv_planes,
                                           hidden_layers=linear_layers, output_length=pad_length)
            self.value_key_proj = KeyProjector(value_encoder_params[0][-1], embed_dim)
            self.value_unprojer = KeyProjector(embed_dim, decoder_params[0][0])

        self.key_encoder = Encoder(query_encoder_params)
        self.embed_dim = embed_dim
        self.memory_initialized = False
        self.batch_size = batch_size
        self.memory_length = memory_length
        self.memory_iteration = 0
        self.loss_params = loss_ratios
        self.learning_rate = learning_rate
        self.pad_length = pad_length

        self.float()

        # TODO: project the memory
        self.query_key_proj = KeyProjector(query_encoder_params[0][-1], embed_dim)

        self.attention_layer = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        # self.memory_key_proj

    def encode_value(self, segment : torch.Tensor) -> (torch.Tensor, (torch.Tensor,)):
        '''encodes the value, returns the encoded value with projection (if any) and the skips'''
        assert segment.shape[2] == self.window_length
        encoded_values = self.value_encoder(segment)
        return encoded_values[-1], encoded_values

    def encode_query(self, segment : torch.Tensor) -> (torch.Tensor, (torch.Tensor,)):
        '''gets the encoded key given aecg segment'''
        assert segment.shape[2] == self.window_length
        encoded_key = self.key_encoder(segment)
        return encoded_key[-1], encoded_key

    def decode_value(self, value : torch.Tensor, value_skips) -> torch.Tensor:
        value = self.value_unprojer(value)
        initial_guess, _ = self.value_decoder(value, value_skips)

        return initial_guess

    def get_key_memory(self):
        '''create functions for training inplace behavior'''
        return self.key_memory

    def get_value_memory(self):
        '''create functions for training inplace behavior'''
        return self.value_memory

    def _initialize_memory(self, memory_key : torch.Tensor, memory_value : torch.Tensor):
        '''initializes the key and value memories
        memory has shape B x key_dim x memory_length * planes (features)'''
        assert self.memory_initialized is False
        # inputs are B x C x W
        self.key_memory = torch.zeros((self.batch_size, self.embed_dim, self.memory_length * memory_key.shape[2])).to(self.device)
        self.value_memory = torch.zeros((self.batch_size, self.embed_dim, self.memory_length * memory_value.shape[2])).to(self.device)

        self.memory_initialized = True
        self.memory_iteration = 0

        self.add_to_memory(memory_value, memory_key)

    def add_to_memory(self, memory_value : torch.Tensor, memory_key : torch.Tensor):
        '''adds value/key to memory'''
        # TODO: better way of adding to memory
        if self.memory_iteration < self.memory_length:
            self.key_memory[:, :, self.memory_iteration: self.memory_iteration + memory_key.shape[2]] = memory_key
            self.value_memory[:, :, self.memory_iteration: self.memory_iteration + memory_value.shape[2]] = memory_value
        else:
            replace_i = self.memory_iteration % self.memory_length

            self.key_memory[:, :, replace_i: replace_i + memory_key.shape[2]] = memory_key
            self.value_memory[:, :, replace_i: replace_i + memory_value.shape[2]] = memory_value

        self.memory_iteration += 1

    def retrieve_memory_value(self, query : torch.Tensor) -> torch.Tensor:
        '''retrieves the value in memory using affinity'''
        value_memory = self.get_value_memory()
        atn = self.attention_layer.forward(query.transpose(1,2), query.transpose(1,2), value_memory.transpose(1,2))

        return atn[0].transpose(1,2)

    def softmax_affinity(self, affinity : torch.Tensor) -> torch.Tensor:
        '''softmaxes affinity matrix S across second dimension'''
        return nn.Softmax(dim=1)(affinity) / sqrt(self.embed_dim)

    def compute_affinity(self, query : torch.Tensor) -> torch.Tensor:
        '''computes affinity between current query and key in memory
        currently uses dot product'''
        # input is B x Ck x W, output is B x L*W x W
        key_memory = self.get_key_memory()
        assert query.shape[1] == key_memory.shape[1]
        # dot product is bmm of transpose
        return torch.bmm(key_memory.transpose(1,2), query)

    def loss_function(self, results) -> torch.Tensor:
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

    def forward(self, aecg_sig : torch.Tensor) -> {str : torch.Tensor}:
        '''data in the format of B x num_windows (set at 100 during training) x window_size'''
        fecg_recon = torch.zeros_like(aecg_sig).to(self.device)
        fecg_peak_recon = torch.zeros(self.peak_shape).to(self.device)

        # get initial guess and initialize memory
        initial_aecg_segment = aecg_sig[:, [0], :]
        initial_value, initial_value_outs = self.encode_value(initial_aecg_segment)
        initial_query, _ = self.encode_query(initial_aecg_segment)

        value_proj = self.value_key_proj(initial_value)
        query_proj = self.query_key_proj(initial_query)
        self._initialize_memory(query_proj, value_proj)

        memory_value = self.retrieve_memory_value(query_proj)
        initial_guess = self.decode_value(memory_value, initial_value_outs)

        fecg_peak_recon[:,0,:] = self.fecg_peak_head(value_proj)
        fecg_recon[:, 0, :] = initial_guess[0][:, 0, :self.window_length]

        for i in range(aecg_sig.shape[1]):
            if i == 0:
                continue

            segment = aecg_sig[:,[i],:]

            query, _ = self.encode_query(segment)
            value, value_outs = self.encode_value(segment)

            value_proj = self.value_key_proj(value)
            query_proj = self.query_key_proj(query)
            self.add_to_memory(query_proj, value_proj)

            memory_value = self.retrieve_memory_value(query_proj)
            guess = self.decode_value(memory_value, value_outs)

            fecg_peak_recon[:, i, :] = self.fecg_peak_head(memory_value)
            fecg_recon[:,i,:] = guess[0][:, 0, :self.window_length]

        # TODO: abolish memory
        self.memory_initialized = False

        return {'fecg_recon' : fecg_recon, 'fecg_peak_recon' : fecg_peak_recon}

    def train_forward(self, aecg_sig: torch.Tensor) -> {str : torch.Tensor}:
        '''same thing as forward but only outputs the last segment'''
        # fecg_recon, fecg_peak_recon = torch.zeros_like(aecg_sig), torch.zeros_like(aecg_sig)

        # get initial guess and initialize memory
        initial_aecg_segment = aecg_sig[:, [0], :]
        initial_value, initial_value_outs = self.encode_value(initial_aecg_segment)
        initial_query, _ = self.encode_query(initial_aecg_segment)

        value_proj = self.value_key_proj(initial_value)
        query_proj = self.query_key_proj(initial_query)
        self._initialize_memory(query_proj, value_proj)

        for i in range(aecg_sig.shape[1]):
            if i == 0:
                continue

            segment = aecg_sig[:, [i], :]

            query, _ = self.encode_query(segment)
            value, value_outs = self.encode_value(segment)

            value_proj = self.value_key_proj(value)
            query_proj = self.query_key_proj(query)
            self.add_to_memory(query_proj, value_proj)

            if i == aecg_sig.shape[1] - 1:
                memory_value = self.retrieve_memory_value(query_proj)
                guess = self.decode_value(memory_value, value_outs)

                fecg_peak_recon = self.fecg_peak_head(memory_value)
                fecg_recon = guess[0][:,0,:self.window_length]

        # TODO: abolish memory
        self.memory_initialized = False

        return {'fecg_recon': fecg_recon, 'fecg_peak_recon': fecg_peak_recon}

    def convert_to_float(self, d : {}):
        # TODO: make better solution
        for k, v in d.items():
            if 'Tensor' in str(type(v)):
                d[k] = v.float()

    def training_step(self, d: {}, batch_idx):
        self.is_training = False
        self.memory_initialized = False
        self.convert_to_float(d)
        aecg_sig = d['mecg_sig'] + d['fecg_sig'] + d['noise']
        # performs backwards on the last segment only to avoid inplace operations with the masking
        self.peak_shape = d['fecg_peaks'].shape
        model_output = self.train_forward(aecg_sig)

        loss_dict = self.calculate_losses_into_dict(model_output['fecg_recon'], d['fecg_sig'][:,-1,:],
                                                    model_output['fecg_peak_recon'], d['fecg_peaks'][:,-1,:])

        self.log_dict({f'train_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)

        return loss_dict['total_loss']

    def validation_step(self, d : {}, batch_idx):
        self.is_training = False
        self.memory_initialized = False
        self.convert_to_float(d)
        aecg_sig = d['mecg_sig'] + d['fecg_sig'] + d['noise']
        self.peak_shape = d['fecg_peaks'].shape
        model_output = self.forward(aecg_sig)

        loss_dict = self.calculate_losses_into_dict(model_output['fecg_recon'], d['fecg_sig'],
                                                    model_output['fecg_peak_recon'], d['fecg_peaks'])

        self.log_dict({f'val_{k}': v for k, v in loss_dict.items()}, sync_dist=True, batch_size=self.batch_size)
        model_output.update(loss_dict)

        return model_output

    def test_step(self, d : {}, batch_idx):
        self.is_training = False
        self.memory_initialized = False
        self.convert_to_float(d)
        aecg_sig = d['mecg_sig'] + d['fecg_sig'] + d['noise']
        self.peak_shape = d['fecg_peaks'].shape
        model_output = self.forward(aecg_sig)

        loss_dict = self.calculate_losses_into_dict(model_output['fecg_recon'], d['fecg_sig'],
                                                    model_output['fecg_peak_recon'], d['fecg_peaks'])

        model_output.update(loss_dict)

        return model_output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def print_summary(self, depth = 7):
        from torchinfo import summary
        random_input = torch.rand((self.batch_size, 1, 250))
        self.peak_shape = (self.batch_size, 5, self.pad_length)
        return summary(self, input_data=random_input, depth=depth)