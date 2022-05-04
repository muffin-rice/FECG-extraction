import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.signal import istft
from torch import nn
from torch.distributions import Normal


def invert_stft(stft_sig):
    x = stft_sig[:, :17, :] + 1j * stft_sig[:, 17:, :]
    orig_sig = istft(x, fs=125, nperseg=32, noverlap=31, input_onesided=True, boundary=False)[1]
    return orig_sig


def invert_stft_batch(stft_sig):  # batch_size x num_c x 34 x 469
    stft_sig1 = stft_sig[:, 0, :, :]
    complex_stft = torch.complex(stft_sig1[:, :17, :], stft_sig1[:, 17:, :])
    return torch.istft(complex_stft, n_fft=32, hop_length=1, onesided=True, return_complex=False,
                       length=500)  # win_length=32,
    # x = []
    # for ent in range(stft_sig.shape[0]):
    #     x.append(invert_stft(stft_sig[ent, :, :, :]))
    #
    # return torch.from_numpy(np.array(x))


# ========================================================================
# ================================= STFT =================================
# ========================================================================

class ResizeConv2d(nn.Module):
    '''upsamples through interpolation and applies a conv (as opposed to deconv filters)'''

    # scales by scale_factor = stride

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    '''double conv + skip connection'''

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), num_blocks=2):
        super().__init__()

        convs = []
        curr_channels = in_channels
        for i in range(num_blocks):
            convs.append(nn.Conv2d(curr_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding='same'))
            curr_channels = out_channels
            convs.append(nn.BatchNorm2d(curr_channels))
            convs.append(nn.ReLU())

        self.convs = nn.Sequential(*convs)

        if stride != (1, 1):
            first_padding = 'valid'
        else:
            first_padding = 'same'

        downconv = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=first_padding,
                             bias=False)

        self.downconvs = nn.Sequential(downconv, nn.BatchNorm2d(out_channels), nn.ReLU())

        # shortcut allows for skip connection when channel size is diff
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            shortcut_kernel = (1, 1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=shortcut_kernel, stride=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        a = self.convs(x) + self.shortcut(x)
        out = self.downconvs(a)
        return out


class ResNet18Enc(nn.Module):
    '''applies 4 downsampling basic resnet blocks'''

    def __init__(self, z_dim=30, start_channels=2, num_blocks=(2, 2, 2, 1)):
        super().__init__()
        channels = [start_channels, 64, 128, 256, 512]
        self.z_dim = z_dim
        self.layer1 = self._make_layer(BasicBlockEnc, (channels[0], channels[1]), num_blocks[0], stride=2)
        self.layer2 = self._make_layer(BasicBlockEnc, (channels[1], channels[2]), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, (channels[2], channels[3]), num_blocks[2], stride=1)
        # self.layer4 = self._make_layer(BasicBlockEnc, (channels[3], channels[4]), num_blocks[3], stride=2)
        self.pool = nn.MaxPool2d((5, 14))
        self.linears = nn.Sequential(*[nn.Linear(channels[3], 512), nn.ReLU()])
        self.mu_linear = nn.Linear(512, z_dim)
        y = nn.Linear(512, z_dim)
        self.var_linear = nn.Sequential(*[y, nn.BatchNorm1d(z_dim), nn.LeakyReLU(negative_slope=.2)])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = (stride, 2 * stride)
        layers = block(planes[0], planes[1], kernel_size=(3, 6), stride=strides, num_blocks=num_blocks)
        return layers

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.pool(x)
        x = nn.Flatten()(x).view(x.size(0), -1)
        x = self.linears(x)  # x is 2 * z_dim, one of which is mu and the other var
        mu = self.mu_linear(x)
        logvar = self.var_linear(x)
        return mu, logvar


class BasicBlockDec(nn.Module):
    '''double conv + upsample'''

    def __init__(self, in_channels, out_channels=None, kernel_size=(3, 3), stride=(1, 1), num_blocks=1):
        # when upsampling, use stride to avoid artifacts as well
        super().__init__()

        if not out_channels:
            out_channels = int(in_channels / stride[1])

        convs = []
        for i in range(num_blocks):
            convs.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=(1, 1), padding='same', bias=False))
            convs.append(nn.BatchNorm2d(in_channels))
            convs.append(nn.ReLU())

        self.convs = nn.Sequential(*convs)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == (1, 1):
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_channels, out_channels, kernel_size=kernel_size, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.convs(x) + x
        out = torch.relu(self.bn1(self.conv1(out)))
        return out


class ResNet18Dec(nn.Module):

    def __init__(self, num_blocks=(2, 2, 2, 1), z_dim=10, nc=1):
        super().__init__()
        planes = [256, 128, 64, nc]
        self.in_planes = planes[0]

        self.linear = nn.Linear(z_dim, 10240)

        self.layer4 = self._make_layer(BasicBlockDec, planes[1], num_blocks[0], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, planes[2], num_blocks[1], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, planes[3], num_blocks[2], stride=2)
        # self.layer1 = self._make_layer(BasicBlockDec, planes[4], num_blocks[3], stride=1)
        # self.conv1 = ResizeConv3d(nc, 1, kernel_size=(1,1,1), scale_factor=1)

    def _make_layer(self, BasicBlockDec, planes, num_blocks, stride):
        strides = (stride, 2 * stride)
        layers = []
        layers += [BasicBlockDec(self.in_planes, planes, num_blocks=num_blocks, stride=strides)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = torch.nn.ReLU()(self.linear(z))
        x = x.view(z.size(0), 256, 5, 8)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        # x = self.layer1(x)
        x = torch.sigmoid(x)
        x = x[:, :, :34, :469]
        return x


class STFT_VAE(pl.LightningModule):

    def __init__(self, log_dict, z_dim=20, learning_rate=0.02, loss_ratio=10000):
        super().__init__()
        self.log_dict = log_dict
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)
        self.loss_ratio = loss_ratio
        self.learning_rate = learning_rate
        self.curr_device = None

    @staticmethod
    def reparameterize(mean, std):
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def loss_function(self, results):
        return (self.loss_ratio * results['recon_loss_mse'] + results['kl_loss'])

    @staticmethod
    def calc_recon_loss_mse(stft_recon, stft):
        return torch.mean(F.mse_loss(stft_recon, stft, reduction='none'))

    @staticmethod
    def calc_recon_loss_mae(stft_recon, stft):
        return torch.mean(F.l1_loss(stft_recon, stft, reduction='none'))

    @staticmethod
    def calc_recon_loss_raw_mse(stft_recon, raw):
        return torch.mean(F.mse_loss(invert_stft_batch(stft_recon)[:, :, 1:], raw[:, :, 1:], reduction='none'))

    @staticmethod
    def calc_recon_loss_raw_mae(stft_recon, raw):
        return torch.mean(F.l1_loss(invert_stft_batch(stft_recon)[:, :, 1:], raw[:, :, 1:], reduction='none'))

    def forward(self, x):
        aecg_stft, fecg_stft, fecg_sig = x['aecg_stft'].float(), x['fecg_stft'].float(), x['fecg_sig'].float()
        mean, logvar = self.encoder(aecg_stft)
        std = torch.exp(logvar / 2)

        dist = Normal(mean, std)
        z = dist.rsample()

        x_recon = self.decoder(z)

        recon_loss_mse = self.calc_recon_loss_mse(x_recon, fecg_stft)
        recon_loss_mae = self.calc_recon_loss_mae(x_recon, fecg_stft)
        recon_loss_raw_mse = self.calc_recon_loss_raw_mse(x_recon.detach().numpy(), fecg_sig)
        recon_loss_raw_mae = self.calc_recon_loss_raw_mae(x_recon.detach().numpy(), fecg_sig)
        kl_loss = torch.mean(self.kl_divergence(z, mean, std))

        return {'x_recon': x_recon, 'recon_loss': recon_loss_mse, 'kl_loss': kl_loss, 'recon_loss_mse': recon_loss_mse,
                'recon_loss_mae': recon_loss_mae, 'recon_loss_raw_mse': recon_loss_raw_mse,
                'recon_loss_raw_mae': recon_loss_raw_mae}

    def training_step(self, batch, batch_idx, device, optimizer_idx=0):
        self.curr_device = device

        results = self.forward(batch)
        results['loss'] = self.loss_function(results)

        return results['loss'], {f'train_{key}': val.item() for key, val in results.items() if 'loss' in key}

    def validation_step(self, batch, batch_idx, device, optimizer_idx=0):
        self.curr_device = device

        results = self.forward(batch)
        results['loss'] = self.loss_function(results)

        return {f"val_{key}": val.item() for key, val in results.items() if 'loss' in key}

    def test_step(self, batch, batch_idx, device, optimizer_idx=0):
        self.curr_device = device

        results = self.forward(batch)
        results['loss'] = self.loss_function(results)

        return {f"test_{key}": val.item() for key, val in results.items() if 'loss' in key}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ========================================================================
# ================================== 2D ==================================
# ========================================================================


class ResizeConv1d(nn.Module):
    '''upsamples through interpolation (as opposed to deconv filters) and applies a conv'''

    # scales by scale_factor = stride

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # self.linear = nn.Linear()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=(1,), padding=(1,))
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = nn.ReLU()(self.bn(self.conv(x)))
        return x


class Basic1DEnc(nn.Module):
    def __init__(self, start_planes, end_planes, blocks=2, kernel_sizes=(13, 9, 5),
                 downsample=(False, (None,), (None,)),
                 final_bn_relu=False):
        super().__init__()

        bn = lambda planes: nn.BatchNorm1d(planes)
        relu = lambda: nn.ReLU()
        conv = lambda in_planes, out_planes, kernel_size: \
            nn.Conv1d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                      padding='same', stride=(1,))

        convs = [conv(start_planes, end_planes, kernel_sizes[0]), bn(end_planes), relu()]

        for block in range(blocks):
            for kernel_size in kernel_sizes[1:]:
                convs.extend([conv(end_planes, end_planes, kernel_size), bn(end_planes), relu()])

        self.convs = nn.Sequential(*convs[:-2])
        self.bn_relu = nn.Sequential(*convs[-2:])
        self.shortcut = nn.Conv1d(in_channels=start_planes, out_channels=end_planes, kernel_size=(1,),
                                  padding='same', stride=(1,))

        if downsample[0]:
            downsamples = [nn.Conv1d(in_channels=end_planes, out_channels=end_planes,
                                     kernel_size=downsample[1], stride=downsample[2])]
            if not final_bn_relu:
                downsamples.append(bn(end_planes))
                downsamples.append(relu())

            self.downsample = nn.Sequential(*downsamples)

        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        return self.downsample(self.bn_relu(self.shortcut(x) + self.convs(x)))


class Encoder2D(nn.Module):
    def __init__(self, z_dim=30, start_channels=2):
        super().__init__()
        self.in_planes = start_channels  # initial number of planes
        self.z_dim = z_dim
        planes = [start_channels, 32, 64, 128, 256, 512]
        strides = [5, 2, 2, 2, 2]

        layers = []
        for i, curr_planes in enumerate(planes[:-1]):
            layers.append(Basic1DEnc(curr_planes, planes[i + 1], downsample=(True, 5, strides[i]),
                                     final_bn_relu=(i == len(planes) - 2)))

        self.enc_layers = nn.Sequential(*layers)

        # ending should be 512 x 3 = 1536
        linears = [1536, 512, 256, 2 * z_dim]
        layers = []

        for i, l in enumerate(linears[:-1]):
            layers.append(nn.Linear(l, linears[i + 1]))
            layers.append(nn.BatchNorm1d(linears[i + 1]))
            layers.append(nn.ReLU())

        self.linear_layers = nn.Sequential(*layers[:-2])

    def forward(self, x):
        x = self.enc_layers(x)
        x = nn.Flatten()(x)
        x = self.linear_layers(x)  # x is 2 * z_dim, one of which is mu and the other var
        mu = x[:, :self.z_dim]
        logvar = nn.LeakyReLU(negative_slope=.2)(x[:, self.z_dim:])
        return mu, logvar


class Decoder2D(nn.Module):
    def __init__(self, z_dim=30):
        super().__init__()
        self.z_dim = z_dim
        planes = [1, 128, 64, 32, 16, 1]
        up_strides = [2, 2, 4, 2, 2]

        layers = []
        for i, curr_planes in enumerate(planes[:-1]):
            layers.append(ResizeConv1d(curr_planes, planes[i + 1], 7, up_strides[i]))

        self.dec_layers = nn.Sequential(*layers)
        self.final_linear = nn.Linear(1060, 500)

        layers = []
        linear_layers = [z_dim, 64, 256, 500]
        for i, curr_planes in enumerate(linear_layers[:-1]):
            layers.append(nn.Linear(curr_planes, linear_layers[i + 1]))
            layers.append(nn.BatchNorm1d(1))
            layers.append(nn.ReLU())

        self.linear_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x1 = self.final_linear(self.dec_layers(x))
        x2 = self.linear_layers(x)
        x = x1 + x2
        return x


class DIRECT_VAE(pl.LightningModule):

    def __init__(self, log_dict, z_dim=20, learning_rate=0.02, loss_ratio=1000):
        super().__init__()
        self.log_dict = log_dict
        self.encoder = Encoder2D(z_dim=z_dim)
        self.decoder = Decoder2D(z_dim=z_dim)
        self.loss_ratio = loss_ratio
        self.learning_rate = learning_rate
        self.curr_device = None

    @staticmethod
    def reparameterize(mean, std):
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def loss_function(self, results):
        return self.loss_ratio * results['recon_loss'] + results['recon_loss_raw_mae'] + results['kl_loss']

    @staticmethod
    def calc_recon_loss_raw_mse(sig_recon, sig):
        return torch.mean(F.mse_loss(sig_recon, sig, reduction='none'))

    @staticmethod
    def calc_recon_loss_raw_mae(sig_recon, sig):
        return torch.mean(F.l1_loss(sig_recon, sig, reduction='none'))

    def forward(self, x):
        aecg_sig, fecg_sig = x['aecg_sig'], x['fecg_sig']
        mean, logvar = self.encoder(aecg_sig)
        std = torch.exp(logvar / 2)

        dist = Normal(mean, std)
        z = dist.rsample()

        x_recon = self.decoder(z)

        recon_loss_raw_mse = self.calc_recon_loss_raw_mse(x_recon, fecg_sig)
        recon_loss_raw_mae = self.calc_recon_loss_raw_mae(x_recon, fecg_sig)
        kl_loss = torch.mean(self.kl_divergence(z, mean, std))

        return {'x_recon': x_recon, 'recon_loss': recon_loss_raw_mse, 'kl_loss': kl_loss,
                'recon_loss_raw_mse': recon_loss_raw_mse, 'recon_loss_raw_mae': recon_loss_raw_mae}

    def training_step(self, batch, batch_idx, device, optimizer_idx=0):
        self.curr_device = device

        results = self.forward(batch)
        results['loss'] = self.loss_function(results)

        self.log_dict({key: val.item() for key, val in results.items() if 'loss' in key}, sync_dist=True)

        return results['loss']

    def validation_step(self, batch, batch_idx, device, optimizer_idx=0):
        self.curr_device = device

        results = self.forward(batch)
        results['loss'] = self.loss_function(results)

        self.log_dict({f"val_{key}": val.item() for key, val in results.items() if 'loss' in key}, sync_dist=True)

    def test_step(self, batch, batch_idx, device, optimizer_idx=0):
        self.curr_device = device

        results = self.forward(batch)
        results['loss'] = self.loss_function(results)

        self.log_dict({f"test_{key}": val.item() for key, val in results.items() if 'loss' in key}, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


MODEL_DICT = {'STFT': STFT_VAE, 'DIRECT': DIRECT_VAE}


class VAE(pl.LightningModule):
    def __init__(self, mode='STFT', learning_rate=.02, z_dim=20):
        super().__init__()
        self.mode = mode
        self.learning_rate, self.z_dim = learning_rate, z_dim
        self.model = MODEL_DICT[mode](learning_rate=learning_rate, z_dim=z_dim, log_dict=self.log_dict)
        self.float()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        curr_device = batch['fecg_sig'].device
        loss, d = self.model.training_step(batch, batch_idx, curr_device, optimizer_idx)

        self.log_dict({k: v for k, v in d.items()}, sync_dist=True)

        return loss.float()

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        curr_device = batch['fecg_sig'].device
        d = self.model.validation_step(batch, batch_idx, curr_device, optimizer_idx)

        self.log_dict({k: v for k, v in d.items()}, sync_dist=True)

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        curr_device = batch['fecg_sig'].device
        d = self.model.test_step(batch, batch_idx, curr_device, optimizer_idx)

        self.log_dict({k: v for k, v in d.items()}, sync_dist=True)

    def configure_optimizers(self):
        return self.model.configure_optimizers()
