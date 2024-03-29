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
                    help='number of planes in (value) encoder')
parser.add_argument('--down_kernels', type=int, default=(5,3,3,3), nargs='+',
                    help='kernel size in (value) encoder')
parser.add_argument('--down_strides', type=int, default=(3,2,1,1), nargs='+',
                    help='stride length in (value) encoder')
parser.add_argument('--down_planes2', type=int, default=(1,64,64,64,64), nargs='+',
                    help='number of planes in (memory) encoder')
parser.add_argument('--down_kernels2', type=int, default=(5,3,3,3), nargs='+',
                    help='kernel size in (memory) encoder')
parser.add_argument('--down_strides2', type=int, default=(3,2,1,1), nargs='+',
                    help='stride length in (memory) encoder')
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
parser.add_argument('--peakhead_downsamples', type=int, default=3,
                    help='number of downsamples to use in peakhead')
parser.add_argument('--key_dim', type=int, default=128,
                    help='key dimension')
parser.add_argument('--val_dim', type=int, default=128,
                    help='value dimension')
parser.add_argument('--embed_dim', type=int, default=128,
                    help='embed dimension')
parser.add_argument('--include_rnn', type=str, default='False',
                    help='have rnn (either fecgmem or unet)')
parser.add_argument('--similarity_type', type=str, default='dot',
                    help='similarity calc between query and memory')
parser.add_argument('--embedding_type', type=str, default='none',
                    help='positional embedding to use, none if no embedding')
parser.add_argument('--embedding_add', type=str, default='False',
                    help='add or concat positional embedding')

# loss arguments
parser.add_argument('--fecg_recon_loss', type=int, default=4,
                    help='weight factor for fecg recon')
parser.add_argument('--fecg_peak_loss', type=float, default=1,
                    help='weight factor for fecg peaks')
parser.add_argument('--peak_loss_ratio', type=float, default=2,
                    help='weight factor for peak indices on fecg')
parser.add_argument('--cancelled_peak_ratio', type=float, default=2,
                    help='weight factor for peaks that are muted')
parser.add_argument('--pooling_kernel', type=int, default=1,
                    help='kernel to pool in final loss')
parser.add_argument('--pooling_stride', type=int, default=1,
                    help='stride to pool in the final loss')

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
# parser.add_argument('')

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
parser.add_argument('--brownian_noise', type=float, default=0.0,
                    help='std of brownian noise to add to data')
parser.add_argument('--numtaps', type=int, default=31,
                    help='numtaps for MECG filtering')
parser.add_argument('--drop_last', type=bool, default=True,
                    help='droplast for dataloader')
parser.add_argument('--peak_padding', type=int, default=10,
                    help='length to pad peak array')
parser.add_argument('--fixed_num_windows', type=bool, default=False,
                    help='fix num windows at num_windows')
parser.add_argument('--window_weights', type=float, default=(.2,.2,.2,.2,.2), nargs='+',
                    help='weights of windows for sampling (if fixednumwin=False)')
parser.add_argument('--suppress_peak', type=float, default=0,
                    help='random chance of peak being suppressed (turn off by = 0)')
parser.add_argument('--max_suppress', type=int, default=2,
                    help='max number of suppressed peaks')
parser.add_argument('--vary_std', type=float, default=0,
                    help='factor to vary the fecg_sig by before combining')
parser.add_argument('--mf_std', type=float, default=0.5,
                    help='factor to vary the mecg:fecg ratio')

# train arguments
parser.add_argument('--trainer_workers', type=int, default=1,
                    help='number of lightning training workers')
parser.add_argument('--num_epochs', type=int, default=120,
                    help='number of epochs')
parser.add_argument('--num_windows', type=int, default=100,
                    help='number of windows to split up a signal')
parser.add_argument('--save_steps', type=int, default=500,
                    help='number of steps to periodically save')
parser.add_argument('--eval_check', type=int, default=1,
                    help='number of train epochs per eval epoch')

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
NUM_EPOCHS = args.num_epochs
SAMPLE_ECG_PKL = args.sample_ecg_path
SAVE_N_STEPS = args.save_steps
TRAIN_PER_VAL_RUN = args.eval_check
LAMBDA_LR = 0
LOSS_THRESHOLD=1
TRAIN_PEAKHEAD = False

import os
if os.path.isfile(SAMPLE_ECG_PKL):
    with open(SAMPLE_ECG_PKL, 'rb') as f:
        SAMPLE_ECG = pkl.load(f)

# model hyperparameters
MODEL = args.model
NUM_BLOCKS = tuple(args.blocks)
VALUE_DOWN_PLANES = tuple(args.down_planes)
VALUE_DOWN_KERNELS = tuple(args.down_kernels)
VALUE_DOWN_STRIDES = tuple(args.down_strides)
MEMORY_DOWN_PLANES = tuple(args.down_planes2)
MEMORY_DOWN_KERNELS = tuple(args.down_kernels2)
MEMORY_DOWN_STRIDES = tuple(args.down_strides2)
UP_PLANES = tuple(args.up_planes)
UP_KERNELS = tuple(args.up_kernels)
UP_STRIDES = tuple(args.up_strides)
assert len(UP_PLANES) == len(VALUE_DOWN_PLANES), f'{UP_PLANES}, {VALUE_DOWN_PLANES}'
SKIP = args.skips
# peak head parameters
INITIAL_CONV_PLANES = args.initial_conv
LINEAR_LAYERS = tuple(args.linear_layers)
PEAK_DOWNSAMPLES = args.peakhead_downsamples
# attention hyperparameters
WINDOW_LENGTH = args.window_length
MEMORY_LENGTH = args.memory_length
# KEY_DIM = args.key_dim
# VAL_DIM = args.val_dim
EMBED_DIM = args.embed_dim
PRETRAINED_UNET_CKPT = args.pretrained_unet
INCLUDE_RNN = args.include_rnn == 'True'
SIMILARITY = args.similarity_type
EMBEDDING_TYPE = args.embedding_type
EMBEDDING_ADD = args.embedding_add == 'True'

# loss hyperparams (new)
FECG_RATIO = args.fecg_recon_loss # fecg recon
FECG_PEAK_LOSS_RATIO = args.fecg_peak_loss # fecg peak loss
FECG_PEAK_CLASS_RATIO = args.peak_loss_ratio # how much to weigh gt peaks on the positive mask
FECG_CANCELLED_RATIO = args.cancelled_peak_ratio
POOLING_KERNEL = args.pooling_kernel
POOLING_STRIDE = args.pooling_stride

# data
DROP_LAST = args.drop_last
MF_RATIO = 4
MF_RATIO_STD = args.mf_std
LOAD_TYPE = args.load_type
NUM_WINDOWS = args.num_windows
NUM_TAPS = args.numtaps
LOAD_INTO_MEMORY = False
NUM_MECG_RANDS = 5
NUM_FECG_RANDS = 10
NOISE = args.noise # ideal noise is 0.001
BROWNIAN_NOISE = args.brownian_noise
SUPPRESS_PEAK_CHANCE = args.suppress_peak
MAX_SUPPRESS = args.max_suppress
BINARY_PEAK_WINDOW = 0 # +-2 marked as 1
COMPRESS_RATIO = (0.84,1.16)
PEAK_SCALE = 1
PEAK_SIGMA = 1
WINDOW_WEIGHTS = tuple(args.window_weights)
FIXED_NUM_WINDOWS = args.fixed_num_windows
PAD_LENGTH = args.peak_padding
VARY_STRENGTH = args.vary_std
import numpy as np
COMPETITION_CHANNELS = np.array([1,1,1,1,1,1,4,1,1,4,2,1,2,1,1,1,3,1,2,3,1,1,2,2,2,2,1,2,1,3,1,1,3,1,3,1,1,1,1,1,1,1,1,1,1,1,2,1,2,4,4,2,1,1,1,1,1,1,1,4,1,1,1,1,1,1,2,1,2,1,4,1,2,1,1]) - 1