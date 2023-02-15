# HYPERPARAMS
import os.path
import pickle as pkl

from argparse import ArgumentParser
parser = ArgumentParser()

# model arguments
parser.add_argument('--model', type=str, default='unet',
                    help='model to use for fecg extraction')
parser.add_argument('--blocks', type=int, default=(8,8,8,8), nargs='+',
                    help='number of blocks')
parser.add_argument('--down_planes', type=int, default=(1,64,64,64,64), nargs='+',
                    help='number of planes in encoder')
parser.add_argument('--down_kernels', type=int, default=(5,3,3,3), nargs='+',
                    help='kernel size in encoder')
parser.add_argument('--down_strides', type=int, default=(3,2,1,1), nargs='+',
                    help='stride length in encoder')
parser.add_argument('--up_planes', type=int, default=(64,64,64,64,1), nargs='+',
                    help='number of planes in decoder')
parser.add_argument('--up_kernels', type=int, default=(3,3,4,6), nargs='+',
                    help='kernel size in decoder')
parser.add_argument('--up_strides', type=int, default=(1,1,2,3), nargs='+',
                    help='stride length in decoder')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--memory_length', type=int, default=20,
                    help='size of memory in model')
parser.add_argument('--pretrained_unet', type=str, default='',
                    help='pretrained unet value encoder/decoder ckpt (blank if train from scratch)')
parser.add_argument('--skips', type=bool, default=False,
                    help='skips to use in decoder')
parser.add_argument('--initial_conv', type=int, default=2,
                    help='number of planes in initial conv for peak index pred')
parser.add_argument('--linear_layers', type=int, default=(108, 128), nargs='+',
                    help='tuple of linear layers for the peak index pred')
parser.add_argument('--key_dim', type=int, default=128,
                    help='key dimension')
parser.add_argument('--val_dim', type=int, default=128,
                    help='value dimension')
parser.add_argument('--embed_dim', type=int, default=128,
                    help='embed dimension')

# loss arguments
parser.add_argument('--fecg_recon_loss', type=int, default=4,
                    help='weight factor for fecg recon')
parser.add_argument('--fecg_peak_loss', type=int, default=1,
                    help='weight factor for fecg peaks')

# sys arguments
parser.add_argument('--model_name', type=str, default='unet_v1',
                    help='model name to save for logging')
parser.add_argument('--model_ver', type=str, default='',
                    help='model version to load checkpoint from (default no ckpt)')
parser.add_argument('--data_dir', type=str, default='Data/preprocessed_data/paired',
                    help='data directory')
parser.add_argument('--log_dir', type=str, default='Run/Logging',
                    help='logging directory')
parser.add_argument('--seed', type=int, default=1,
                    help='seed number')
parser.add_argument('--log_steps', type=int, default=10,
                    help='log every n steps')
parser.add_argument('--device', type=str, default='cpu',
                    help='device to use in pytorch lightning')
parser.add_argument('--sample_ecg_path', type=str, default='sample_ecg.pkl',
                    help='sample ecg for model runs')

# data arguments
parser.add_argument('--load_type', type=str, default='new',
                    help='which data type to load')
parser.add_argument('--window_length', type=int, default=250,
                    help='size of windows to pass as inputs to model')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--data_workers', type=int, default=8,
                    help='number of lightning data workers')
parser.add_argument('--noise', type=float, default=0.0,
                    help='random uniform noise to add to data in noise param')
parser.add_argument('--numtaps', type=int, default=31,
                    help='numtaps for MECG filtering')
parser.add_argument('--drop_last', type=bool, default=True,
                    help='droplast for dataloader')
parser.add_argument('--peak_padding', type=int, default=10,
                    help='length to pad peak array')
parser.add_argument('--fixed_num_windows', type=bool, default=False,
                    help='fix num windows at num_windows')
parser.add_argument('--window_weights', type=float, default=(.2,.2,.2,.2,.2), nargs='+',
                    help='weights of windows for sampling')

# train arguments
parser.add_argument('--trainer_workers', type=int, default=1,
                    help='number of lightning training workers')
parser.add_argument('--num_epochs', type=int, default=120,
                    help='number of epochs')
parser.add_argument('--num_windows', type=int, default=100,
                    help='number of windows to split up a signal')

args, unknown = parser.parse_known_args()

# execution hyperparameters
SEED = args.seed
LOG_DIR = args.log_dir
MODEL_NAME = args.model_name
MODEL_VER = args.model_ver
DATA_DIR = args.data_dir
LOG_STEPS = args.log_steps
LEARNING_RATE = args.learning_rate
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = args.device
NUM_TRAINER_WORKERS = args.trainer_workers
NUM_DATA_WORKERS = args.data_workers
BATCH_SIZE = args.batch_size
FIND_UNUSED=False
NUM_EPOCHS = args.num_epochs
SAMPLE_ECG_PKL = args.sample_ecg_path
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
MODEL = args.model
Z_DIM = 128
NUM_BLOCKS = tuple(args.blocks)
DOWN_PLANES = tuple(args.down_planes)
DOWN_KERNELS = tuple(args.down_kernels)
DOWN_STRIDES = tuple(args.down_strides)
UP_PLANES = tuple(args.up_planes)
UP_KERNELS = tuple(args.up_kernels)
UP_STRIDES = tuple(args.up_strides)
assert len(UP_PLANES) == len(DOWN_PLANES)
START_CHANNELS = 1
END_CHANNELS = 3
RECON_SIG = 'gt_fecg' # signal to reconstruct
SKIP = args.skips
INITIAL_CONV_PLANES = args.initial_conv
LINEAR_LAYERS = tuple(args.linear_layers)
# attention hyperparameters
ATTENTION = False
WINDOW_LENGTH = args.window_length
MEMORY_LENGTH = args.memory_length
# KEY_DIM = args.key_dim
# VAL_DIM = args.val_dim
EMBED_DIM = args.embed_dim
PRETRAINED_UNET_CKPT = args.pretrained_unet

# loss hyperparams (new)
FECG_RATIO = args.fecg_recon_loss # fecg recon
MECG_RATIO = 1 # mecg recon
FECG_PEAK_LOSS_RATIO = args.fecg_peak_loss # fecg peak loss

# data
DROP_LAST = args.drop_last
MF_RATIO = 4
MF_RATIO_STD = 0.5
LOAD_TYPE = args.load_type
NUM_WINDOWS = args.num_windows
NUM_TAPS = args.numtaps
LOAD_INTO_MEMORY = False
NUM_MECG_RANDS = 5
NUM_FECG_RANDS = 10
NOISE = args.noise # ideal noise is 0.001
BINARY_PEAK_WINDOW = 0 # +-2 marked as 1
COMPRESS_RATIO = (0.84,1.16)
PEAK_SCALE = 1
PEAK_SIGMA = 1
WINDOW_WEIGHTS = args.window_weights
FIXED_NUM_WINDOWS = args.fixed_num_windows
PAD_LENGTH = args.peak_padding
import numpy as np
COMPETITION_CHANNELS = np.array([1,1,1,1,1,1,4,1,1,4,2,1,2,1,1,1,3,1,2,3,1,1,2,2,2,2,1,2,1,3,1,1,3,1,3,1,1,1,1,1,1,1,1,1,1,1,2,1,2,4,4,2,1,1,1,1,1,1,1,4,1,1,1,1,1,1,2,1,2,1,4,1,2,1,1]) - 1