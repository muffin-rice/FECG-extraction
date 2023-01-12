from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, down_params : ((int,),), encoder_skip : bool = False):
        # params in the format ((num_planes), (kernel_width), (stride))
        super().__init__()
        self.encoder_skip = encoder_skip
        self.leaky = nn.LeakyReLU(negative_slope=0.1)

        self.encodes = nn.ParameterList()
        for i in range(len(down_params[0]) - 1):
            self.encodes.append(self.make_encoder_block(down_params[0][i], down_params[0][i + 1],
                                                            kernel_size=down_params[1][i], stride=down_params[2][i]))

    def make_skip_connection(self, a, b):
        '''makes a skip connection and applies leaky relu'''
        b_shape = b.shape
        return self.leaky(a[:, :, :b_shape[2]]+b)

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
    def __init__(self, up_params : ((int,),), head_params : (str,), decoder_skip : bool = True, signal_length = 500):
        super().__init__()
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.signal_length = signal_length

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
        # skips are in the format of [input, layer1, layer2, ..., inner_layer]
        encode_outs = [inner_layer]

        # decode_outs: [encoded_layer, layern, ..., layer1]
        decode_outs = [encode_outs[-1]]
        for i, layer in enumerate(self.decode_layers):
            decoder_output = layer(decode_outs[-1])
            decode_outs.append(self.make_skip_connection(-decoder_output, skips[-i - 2]))

        heads = []
        for layer in self.heads:
            heads.append(layer(decode_outs[-1])[:,:,:self.signal_length])

        return heads, decode_outs

class KeyProjector(nn.Module):
    def __init__(self, input_dim, key_dim):
        super().__init__()

        self.input_dim = input_dim
        self.key_dim = key_dim

        self.key_proj = nn.Conv1d(input_dim, key_dim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        # TODO: shrinkage and selection
        return self.key_proj(x)