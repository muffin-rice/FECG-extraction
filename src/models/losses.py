import torch.nn.functional as F

def calc_mse(x_recon, x):
    return F.mse_loss(x_recon, x, reduction='none')

def calc_mae(x_recon, x):
    return F.l1_loss(x_recon, x, reduction='none')

def calc_peak_loss(mse, mask):
    return mse * mask

def calc_bce_loss(mask_recon, mask):
    return F.binary_cross_entropy(mask_recon, mask, reduction='none')

def apply_pool(mask, pool_kernel, pool_stride):
    return F.max_pool1d(mask, pool_kernel, stride=pool_stride)