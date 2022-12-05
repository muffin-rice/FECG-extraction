# HYPERPARAMS
import os.path
import pickle as pkl

# execution hyperparameters
SEED = 1
LOG_DIR = 'Run/Logging'
MODEL_NAME = 'unet_v1'
DATA_DIR = 'Data/preprocessed_data/paired'
LOG_STEPS = 10
LEARNING_RATE = 1e-3
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
NUM_TRAINER_WORKERS = 1
NUM_DATA_WORKERS = 8
BATCH_SIZE = 128
FIND_UNUSED=False
NUM_EPOCHS = 120
SAMPLE_ECG_PKL = 'sample_ecg.pkl'
SAVE_N_STEPS = 500
TRAIN_PER_VAL_RUN = 1
LAMBDA_LR = 0
LOSS_THRESHOLD=1
TRAIN_PEAKHEAD = False

import os
if os.path.isfile(SAMPLE_ECG_PKL):
    with open(SAMPLE_ECG_PKL, 'rb') as f:
        SAMPLE_ECG = pkl.load(f)

# model hyperparameters
MODEL = 'unet'
Z_DIM = 128
NUM_BLOCKS = (8,8,8,8)
NUM_STRIDES = (3,3,2,1,1)
NUM_KERNELS = (5,5,3,3,3)
DECODER_KERNELS = (3,3,3,6,5)
DECODER_STRIDES = (1,1,2,3,3)
NUM_PLANES_UP = (64,64,64,64,64,1)
NUM_PLANES_DOWN = (1,64,64,64,64,64)
assert len(NUM_PLANES_UP) == len(NUM_PLANES_DOWN)
START_CHANNELS = 1
END_CHANNELS = 3
RECON_SIG = 'gt_fecg' # signal to reconstruct
SKIP = True
# attention hyperparameters
EMBED_DIM = 166
ATTENTION = False

# loss hyperparams (new)
FECG_RATIO = 20 # fecg recon
FECG_BCE_RATIO = 1 # fecg bce peak mask
FECG_BCE_CLASS_RATIO = 1
MECG_RATIO = 1 # mecg recon

# loss hyperparameters
LOG_COSH_FACTOR = 0 # factor for loss_log_cosh
MSE_LOSS_RATIO = 10 # factor for raw MSE
MAE_LOSS_RATIO = 30
KL_FACTOR = 1 # factor for KL loss
PEAK_SCALE = 5 # parameters for peak mask; centered around GT peaks and scaled by PEAK_SCALE
PEAK_SIGMA = 1 # std for peak mask for each peak
FETAL_MASK_LOSS_FACTOR = 300 # factor for the peak loss
MATERNAL_MASK_LOSS_FACTOR = 50 # factor for the peak loss
# peak head learning factors
BCE_LOSS_RATIO = .1
BCE_CLASS_RATIO = 5
# mecg peak head learning factors
BCE_MECG_LOSS_RATIO = .1
BCE_MECG_CLASS_RATIO = 5
MECG_LOSS_RATIO = 1
# contrastive learning factors
SS_LOSS_RATIO = 1 # overall loss factor for the contrastive learning
FIX_FECG_ALPHA = 10 # factor for fixing the fecg (changing mecg)
SWITCH_FECG_WINDOW_ALPHA = 1 # factor for switching fecg window (fixing mecg)
SWITCH_FECG_ALPHA = -8 # factor for switching fecg (fixing mecg)

# data
DROP_LAST = False
MF_RATIO = 4
MF_RATIO_STD = 0.5
LOAD_TYPE = 'new'
LOAD_INTO_MEMORY = False
NUM_MECG_RANDS = 5
NUM_FECG_RANDS = 10
NOISE = 0.0 # ideal noise is 0.001
BINARY_PEAK_WINDOW = 0 # +-2 marked as 1
COMPRESS_RATIO = (0.84,1.16)