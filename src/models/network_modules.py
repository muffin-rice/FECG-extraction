from torch import nn
import torch
import math

class EncoderBlock(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, kernel_size : int = 4, stride : int = 2,):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.lr1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.lr2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x):
        x = self.lr1(self.bn1(self.conv1(x)))
        y = self.lr2(self.bn2(self.conv2(x)))

        return x+y

class Encoder(nn.Module):
    def __init__(self, down_params : ((int,),), encoder_skip : bool = False):
        # params in the format ((num_planes), (kernel_width), (stride))
        super().__init__()
        self.encoder_skip = encoder_skip
        self.leaky = nn.LeakyReLU(negative_slope=0.1)

        self.encodes = nn.ParameterList()
        for i in range(len(down_params[0]) - 1):
            self.encodes.append(EncoderBlock(input_channels= down_params[0][i], output_channels= down_params[0][i + 1],
                                             kernel_size=down_params[1][i], stride=down_params[2][i]))

    def make_skip_connection(self, a, b):
        '''makes a skip connection and applies leaky relu'''
        b_shape = b.shape
        return self.leaky(a[:, :, :b_shape[2]]+b)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, aecg_sig, skips = None) -> [torch.Tensor]:
        '''returns encode_outs: [aecg, layer1, layer2, ..., encoded_layer]'''
        encode_outs = [aecg_sig]
        for layer in self.encodes:
            layer_output = layer(encode_outs[-1])
            if self.encoder_skip:
                assert skips is not None, 'Encoder skip is true but skips is empty'
                layer_output = layer(self.make_skip_connection(skips[-1], layer_output))
            encode_outs.append(layer_output)

        return encode_outs

class Decoder(nn.Module):
    def __init__(self, up_params : ((int,),), head_params : (str,), signal_length = 500, skips : bool = True):
        super().__init__()
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.signal_length = signal_length
        self.skips = skips

        self.decode_layers = nn.ParameterList()
        for i in range(len(up_params[0]) - 2):
            self.decode_layers.append(self.make_decoder_block(up_params[0][i], up_params[0][i + 1],
                                                              kernel_size=up_params[1][i], stride=up_params[2][i]))

        self.heads = nn.ParameterList()
        for head in head_params:
            if head == 'tanh':
                final_activate = nn.Tanh() # [-1,1]
            elif head == 'sigmoid':
                final_activate = nn.Sigmoid() # [0,1]
            else:
                raise NotImplementedError('No other head activations exist')

            self.heads.append(self.final_layer(up_params[0][-2], final_activate, up_params[1][-1], up_params[2][-1],))

    def make_skip_connection(self, a, b):
        '''makes a skip connection and applies leaky relu'''
        b_shape = b.shape
        return self.leaky(a[:, :, :b_shape[2]]+b)

    def make_decoder_block(self, input_channels, output_channels, kernel_size=8, stride=4, output_padding=1,):
        return nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride,
                               output_padding=min(output_padding, stride - 1)),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(output_channels),
        )

    def final_layer(self, input_channels, activation : nn.Module, kernel_size=8, stride=4, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose1d(input_channels, 1, kernel_size, stride, output_padding=output_padding),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv1d(1, 1, kernel_size=5, stride=1, padding='same'), # minor smoothing
            nn.BatchNorm1d(1),
            activation,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inner_layer, skips = None) -> ([torch.Tensor], [torch.Tensor]):
        '''returns [heads] (num_heads) and
        decode_outs: [encoded_layer, layern, ..., layer1]'''
        if not self.skips:
            self.skips = None
        # skips are in the format of [input, layer1, layer2, ..., inner_layer]
        encode_outs = [inner_layer]

        # decode_outs: [encoded_layer, layern, ..., layer1]
        decode_outs = [encode_outs[-1]]
        for i, layer in enumerate(self.decode_layers):
            decoder_output = layer(decode_outs[-1])
            if skips is not None:
                decode_outs.append(self.make_skip_connection(-decoder_output, skips[-i - 2]))
            else:
                decode_outs.append(decoder_output)

        heads = []
        for layer in self.heads:
            heads.append(layer(decode_outs[-1])[:,:,:self.signal_length])

        return heads, decode_outs

class KeyProjector(nn.Module):
    def __init__(self, input_dim, key_dim):
        super().__init__()

        self.input_dim = input_dim
        self.key_dim = key_dim

        self.key_proj = nn.Conv1d(input_dim, key_dim, kernel_size=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        # TODO: shrinkage and selection
        return self.key_proj(x)

class RNN(nn.Module):
    def __init__(self, val_dim, batch_size):
        super().__init__()

        self.val_dim = val_dim
        self.batch_size = batch_size

        self.hidden_state = None

        self.hidden_conv = nn.Sequential(*[
            nn.Conv1d(2 * self.val_dim, 2*self.val_dim, kernel_size=3, stride=1, padding='same',),
            nn.BatchNorm1d(2*self.val_dim),
            nn.LeakyReLU(negative_slope=0.1)
        ])

    def _reinitialize(self, seq_length, device):
        self.hidden_state = torch.zeros(self.batch_size, self.val_dim, seq_length).to(device)

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, value):
        total_hidden = torch.concatenate((self.hidden_state, value), dim=1) # concat along feature dim

        out_rnn = self.hidden_conv(total_hidden)

        self.hidden_state = out_rnn[:,:self.val_dim,:]

        return out_rnn[:,self.val_dim:,:]


class NaiveMemory:
    '''Naive memory is just storing the memory as a plain matrix without any sophisticated
    memory management'''
    def __init__(self, device, memory_length, embed_dim):
        self.device = device
        self.memory_initialized = False
        self.memory_length = memory_length
        self.embed_dim = embed_dim

    def _reinitialize_memory(self, batch_size, seq_length, ):
        '''initializes the key and value memories
                memory has shape B x K x N * P'''
        self.seq_length = seq_length
        self.key_memory = torch.zeros((batch_size, self.embed_dim, self.memory_length * seq_length)).to(self.device)
        self.value_memory = torch.zeros((batch_size, self.embed_dim, self.memory_length * seq_length)).to(self.device)

        self.memory_initialized = True
        self.memory_iteration = 0

    def add_to_memory(self, memory_value : torch.Tensor, memory_key : torch.Tensor):
        '''adds value/key to memory'''
        assert self.memory_initialized
        if self.memory_iteration < self.memory_length:
            start_i = self.memory_iteration * self.seq_length
            self.key_memory[:, :, start_i : start_i + self.seq_length] = memory_key
            self.value_memory[:, :, start_i : start_i + self.seq_length] = memory_value
        else:
            # shift matrix, then append to end
            shift_length = (self.memory_length - 1) * self.seq_length
            self.key_memory[:,:,:shift_length] = self.key_memory[:,:,self.seq_length:]
            self.value_memory[:,:,:shift_length] = self.value_memory[:,:,self.seq_length:]

            self.key_memory[:,:,shift_length:] = memory_key
            self.value_memory[:,:,shift_length:] = memory_value

        self.memory_iteration += 1

    def get_key_memory(self) -> torch.Tensor:
        '''returns value memory in shape of B x QK x N*P'''
        return self.key_memory

    def get_value_memory(self) -> torch.Tensor:
        '''returns value memory in shape of B x Vk x N*P'''
        return self.value_memory

    def get_memory_copy(self):
        return torch.clone(self.key_memory).cpu(), torch.clone(self.value_memory).cpu()

    def is_memory_initialized(self):
        return self.memory_initialized

    def process_affinity(self, affinity : torch.Tensor) -> None:
        pass

    def abolish_memory(self):
        del self.key_memory
        del self.value_memory
        self.memory_initialized = False
        self.memory_iteration = 0

class MemoryRanker:
    '''Naive memory is just storing the memory as a plain matrix without any sophisticated
    memory management'''
    def __init__(self, device, memory_length, embed_dim):
        self.device = device
        self.memory_initialized = False
        self.memory_length = memory_length
        self.embed_dim = embed_dim
        self.indices_to_replace = None
        self.memory_is_full = False

    def _reinitialize_memory(self, batch_size, seq_length, ):
        '''initializes the key and value memories
                memory has shape B x K x N * P'''
        self.seq_length = seq_length
        self.key_memory = torch.zeros((batch_size, self.embed_dim, self.memory_length * seq_length)).to(self.device)
        self.value_memory = torch.zeros((batch_size, self.embed_dim, self.memory_length * seq_length)).to(self.device)

        self.memory_initialized = True
        self.memory_iteration = 0

    def add_to_memory(self, memory_value : torch.Tensor, memory_key : torch.Tensor):
        '''adds value/key to memory'''
        assert self.memory_initialized
        if self.memory_iteration < self.memory_length:
            start_i = self.memory_iteration * self.seq_length
            self.key_memory[:, :, start_i : start_i + self.seq_length] = memory_key
            self.value_memory[:, :, start_i : start_i + self.seq_length] = memory_value
        else:
            assert self.memory_is_full
            assert self.indices_to_replace is not None
            batch_size = memory_key.shape[0]
            temp_km = self.key_memory.view((batch_size, self.embed_dim, self.memory_length, self.seq_length))
            temp_vm = self.key_memory.view((batch_size, self.embed_dim, self.memory_length, self.seq_length))

            temp_km[torch.arange(batch_size), :, self.indices_to_replace, :] = memory_key
            temp_vm[torch.arange(batch_size), :, self.indices_to_replace, :] = memory_value
            # self.key_memory[torch.arange(batch_size),:,self.indices_to_replace] = memory_key
            # self.value_memory[torch.arange(batch_size),:,self.indices_to_replace] = memory_value

        self.memory_iteration += 1
        if self.memory_iteration >= self.memory_length:
            self.memory_is_full = True

    def get_key_memory(self) -> torch.Tensor:
        '''returns value memory in shape of B x QK x N*P'''
        return self.key_memory

    def get_value_memory(self) -> torch.Tensor:
        '''returns value memory in shape of B x Vk x N*P'''
        return self.value_memory

    def get_memory_copy(self):
        return torch.clone(self.key_memory).cpu(), torch.clone(self.value_memory).cpu()

    def is_memory_initialized(self):
        return self.memory_initialized

    def process_affinity(self, affinity : torch.Tensor) -> None:
        '''should be called after computing affinity; removes an entry from memory'''
        # softmaxed affinity as a B x NQk x Qk
        B, NQk, Qk = affinity.shape
        if not self.memory_is_full:
            return

        reshaped_aff = affinity.transpose(1,2).resize(B, self.memory_length, Qk, Qk)

        # rank reshaped_aff
        summed_aff = reshaped_aff.sum(dim=(2,3)) # size B x N, sort

        self.indices_to_replace = summed_aff.argmin(dim=1)

    def abolish_memory(self):
        del self.key_memory
        del self.value_memory
        self.memory_initialized = False
        self.memory_iteration = 0
        self.indices_to_replace = None
        self.memory_is_full = False

class PositionalEmbedder(nn.Module):
    def __init__(self, positional_type = 'none', add = True):
        super().__init__()
        self.embedding_type = positional_type
        if self.embedding_type == 'none':
            add = True
        self.inited = False
        self.add = add # if not add, concat
        self.get_embedding = {
            'baseline' : self.get_baseline_embedding,
            'none' : self.get_no_embedding,
        }

    def move_device(self, device):
        self.device = device

    def forward(self, x):
        embedding = self.get_embedding[self.embedding_type](x)
        if self.add:
            return x + embedding
        else:
            return torch.concat((x, embedding), dim=1)

    def get_baseline_embedding(self, x, c = 10000):
        B, K, L = x.shape
        if K == 1:
            K = 2
        k_range = torch.arange(K // 2) + 1
        l_range = torch.arange(L)
        sin_exps = torch.sin(l_range[None, :] / c ** (2 * k_range[:, None] / K))
        cos_exps = torch.cos(l_range[None, :] / c ** ((2 * k_range[:, None] + 1) / K))
        embedding = torch.zeros((B, K, L)).to(self.device)
        embedding[:, ::2, :] = sin_exps
        embedding[:, 1::2, :] = cos_exps

        return embedding

    def get_no_embedding(self, x):
        return torch.zeros_like(x).to(self.device)

# class PeakHead(nn.Module):
#     '''Downsample sequence further, then flatten and perform regression'''
#     def __init__(self, starting_planes : int, ending_planes : int, hidden_layers : (int,), output_length : int,
#                  num_downsampling : int):
#         super().__init__()
#
#         initial_conv = self.downsampler(starting_planes, ending_planes, 3, 0, 2)
#
#         for i in range(num_downsampling-1):
#             initial_conv += self.downsampler(ending_planes, ending_planes, 3, 0, 2)
#
#         # if ending_planes:
#         #     initial_conv.append(nn.Conv1d(ending_planes, ending_planes, kernel_size=1, padding=0))
#         #     initial_conv.append(nn.BatchNorm1d(ending_planes))
#
#         self.initial_conv = nn.Sequential(*initial_conv[:-1])
#
#         self.flatten = nn.Flatten()
#         linears = [nn.LeakyReLU(negative_slope=0.1)]
#
#         for i in range(len(hidden_layers) - 1):
#             linears.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
#             linears.append(nn.LeakyReLU(negative_slope=0.1))
#
#         linears.append(nn.Linear(hidden_layers[-1], output_length))
#         linears.append(nn.ReLU())
#
#         self.linears = nn.Sequential(*linears)
#
#     def downsampler(self, starting_planes, ending_planes, k, p, s):
#         return [
#             nn.Conv1d(starting_planes, ending_planes, kernel_size=k, padding=p, stride=s),
#             nn.BatchNorm1d(ending_planes),
#             nn.LeakyReLU(negative_slope=0.1)
#         ]
#
#     def forward(self, x):
#         x = self.initial_conv(x)
#         x = self.flatten(x)
#         return self.linears(x)
