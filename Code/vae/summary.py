from vae_backbone import VAE
from torchinfo import summary
import torch
from hyperparams import *

model = VAE(sample_ecg = SAMPLE_ECG)
d = {'mecg_stft' : torch.rand((BATCH_SIZE, 2, 34, 469)).double(), 'fecg_stft' : torch.rand((BATCH_SIZE, 1, 34, 469)).double(),
     'mecg_sig' : torch.rand((BATCH_SIZE, 2, 500)).double(), 'fecg_sig' : torch.rand((BATCH_SIZE, 1, 500)).double(),
    'offset' : torch.rand((BATCH_SIZE, 2, 500)).double()}

print(summary(model, input_data=[d], depth=7))
