# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

from copy import deepcopy
from distutils.command.config import config
import os
from pickle import TRUE
import yaml
from yacs.config import CfgNode as CN
from data import datasets
import argparse
import pdb

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = '/home/zzq/data'
# _C.DATA.NO_SPLIT_PATH = '~/data'
# _C.DATA.SPLIT = True
#Dataset domains,could be overwritten by yaml
_C.DATA.DOMAINS=['photo', 'art_painting', 'cartoon', 'sketch']
# Dataset name could be written by yaml
_C.DATA.DATASET = 'PACS'#PACS
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# # Cache Data in Memory, could be overwritten by command line argument
# _C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.

_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
_C.DATA.SPLIT = 'indomain'
_C.DATA.SPLIT_RATE = 0.9


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'dat'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 7#1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
#one in ['', 'linear', 'cosine'],Do classification using cosine similatity between activations and weights
_C.MODEL.CLASSIFIER='linear'
# Ensemble parameters
_C.MODEL.ENS = CN()
_C.MODEL.ENS.HEAD_DROPOUT = 0.1
_C.MODEL.ENS.HEAD_PROB =0.7
_C.MODEL.ENSEM_NAME = 'Resnet18_shared'
_C.MODEL.ENSEM_TYPE = 'ResNet18_sh'
#load ensemble model for distill
_C.MODEL.ENSEMBLE_TAG = 'default'#'source_idx'
#loade distill model for analysis
_C.MODEL.DISTILL_TAG = 'default'
_C.MODEL.SHARED_KEYS=['conv','fc_inv']


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 150
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.WEIGHT_DECAY = 0.05#0.05
_C.TRAIN.BASE_LR = 5e-4#-4   
_C.TRAIN.WARMUP_LR = 5e-7#-7
_C.TRAIN.MIN_LR = 1e-5#-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.SAMESUB_EPOCH = 0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 40
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
#adamw sgd
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
#If pretrained
_C.TRAIN.PRETRAINED = True

_C.TRAIN.T = 5.0

_C.TRAIN.ENSEM_LAMDA = 0.1

_C.TRAIN.PRE_EPOCH =0

_C.TRAIN.RANDOM_SAMPLER = True  #False:Uniform sampler
_C.TRAIN.MODEL_SELECTION = 'valacc' #valloss

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'auto'#'rand-m9-mstd0.5-inc1' 
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1

# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# overwritten by command line argument
_C.AMP = False
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 100
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# If froze part parameters 
_C.PART_FROZEN = True

_C.DISTILL=CN()


#gamma*cata_loss + (1-gamma)distil_loss
_C.DISTILL.GAMMA = 0.167
_C.DISTILL.IDENTICAL_LOGIT =False

_C.ENS=CN()
_C.ENS.LADA_CTRA = 0.3


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.pretrained:
        config.TRAIN.PRETRAINED = True
    if args.gamma:
        config.DISTILL.GAMMA = args.gamma
    if args.ens_lamda:
        config.TRAIN.ENSEM_LAMDA =args.ens_lamda
    if args.ens_tag:
        config.MODEL.ENSEMBLE_TAG = args.ens_tag      
    if args.dis_tag:
        config.MODEL.DISTILL_TAG = args.dis_tag   
    if args.samesub_epoch:  
        config.TRAIN.SAMESUB_EPOCH =args.samesub_epoch 
    if args.shared_keys:
        config.MODEL.SHARED_KEYS=  args.shared_keys
    if args.classifier:
        config.MODEL.CLASSIFIER = args.classifier
    if args.model_select:
        config.TRAIN.MODEL_SELECTION = args.model_select
    if args.aug:
        config.AUG.AUGMENT = args.aug
    if args.seed:
        config.seed = args.seed
    if args.ens_lada_ctra:
        config.ENS.LADA_CTRA = args.ens_lada_ctra
    if args.dataset:
        if  args.dataset in vars(datasets):
            config.DATA.DATASET= args.dataset
            
        else:
            NotImplementedError(f'datset {args.dataset} not supported')
    else:
        NotImplementedError('Please input dataset')

    hparams={'split':config.DATA.SPLIT,'split_rate':config.DATA.SPLIT_RATE,'aug':'standard'}
    dataset = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,
        [0], hparams)
    config.DATA.DOMAINS= vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,
        [0], hparams).ENVIRONMENTS
    config.MODEL.NUM_CLASSES = vars(datasets)[config.DATA.DATASET](config.DATA.DATA_PATH,
        [0], hparams).NUM_CLASSES
    
    # output folder 
    #for analysis   
    config.DISTILL_PATH=os.path.join(config.OUTPUT,config.DATA.DATASET,config.MODEL.NAME,config.MODEL.DISTILL_TAG)
    #for distill
    config.ENSEM_PATH = os.path.join(config.OUTPUT,config.DATA.DATASET, config.MODEL.ENSEM_NAME, config.MODEL.ENSEMBLE_TAG)
    #for ensemble
    config.OUTPUT = os.path.join(config.OUTPUT,config.DATA.DATASET, config.MODEL.NAME, config.TAG)

    config.freeze()

    


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config




def parse_option():
    parser = argparse.ArgumentParser('Ensemble training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',#有传参 use-checkpoint就是true
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', action='store_true', help='If use pretrained parametres')
    parser.add_argument('--gamma',type=float,help='gamma of loss')
    parser.add_argument('--samesub_epoch',type=int,help='')
    parser.add_argument('--ens_tag',type=str,help='')
    parser.add_argument('--dis_tag',type=str,help='')
    parser.add_argument('--model_select',type=str,help='')
    parser.add_argument('--ens_lamda',type=float,help='gamma of loss')
    parser.add_argument('--shared_keys',nargs='+',help='')
    parser.add_argument('--classifier',type=str,help='')
    parser.add_argument('--aug',type=str,help='')
    parser.add_argument('--seed',type=int,help='')
    parser.add_argument('--ens_lada_ctra',type=float,help='')


    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config
