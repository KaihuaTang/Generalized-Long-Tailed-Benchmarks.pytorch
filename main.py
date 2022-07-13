######################################
#         Kaihua Tang
######################################

import json
import yaml
import os
import argparse
import torch
import torch.nn as nn
import random
import utils.general_utils as utils
from utils.logger_utils import custom_logger
from data.dataloader import get_loader
from utils.checkpoint_utils import Checkpoint
from utils.training_utils import *


from utils.train_loader import train_loader
from utils.test_loader import test_loader

# ============================================================================
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str, help='Indicate the config file used for the training.')
parser.add_argument('--seed', default=25, type=int, help='Fix the random seed for reproduction. Default is 25.')
parser.add_argument('--phase', default='train', type=str, help='Indicate train/val/test phase.')
parser.add_argument('--load_dir', default=None, type=str, help='Load model from this directory for testing')
parser.add_argument('--output_dir', default=None, type=str, help='Output directory that saves everything.')
parser.add_argument('--require_eval', action='store_true', help='Require evaluation on val set during training.')
parser.add_argument('--logger_name', default='logger_eval', type=str, help='Name of TXT output for the logger.')
# update config settings
parser.add_argument('--lr', default=None, type=float, help='Learning Rate')
parser.add_argument('--testset', default=None, type=str, help='Reset the type of test set.')
parser.add_argument('--loss_type', default=None, type=str, help='Reset the type of loss function.')
parser.add_argument('--model_type', default=None, type=str, help='Reset the type of model.')
parser.add_argument('--train_type', default=None, type=str, help='Reset the type of traning strategy.')
parser.add_argument('--sample_type', default=None, type=str, help='Reset the type of sampling strategy.')
parser.add_argument('--rand_aug', action='store_true', help='Apply Random Augmentation During Training.')
parser.add_argument('--save_all', action='store_true', help='Save All Output Information During Testing.')

args = parser.parse_args()

# ============================================================================
# init logger
if args.output_dir is None:
    print('Please specify output directory')
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
if args.phase != 'train':
    logger = custom_logger(args.output_dir, name='{}.txt'.format(args.logger_name))
else:
    logger = custom_logger(args.output_dir)
logger.info('========================= Start Main =========================')


# ============================================================================
# fix random seed
logger.info('=====> Using fixed random seed: ' + str(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ============================================================================
# load config
logger.info('=====> Load config from yaml: ' + str(args.cfg))
with open(args.cfg) as f:
    config = yaml.load(f)

# load detailed settings for each algorithms
logger.info('=====> Load algorithm details from yaml: config/algorithms_config.yaml')
with open('config/algorithms_config.yaml') as f:
    algo_config = yaml.load(f)

# update config
logger.info('=====> Merge arguments from command')
config = utils.update(config, algo_config, args)

# save config
logger.info('=====> Save config as config.json')
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(config, f)
utils.print_config(config, logger)

# ============================================================================
# training
if args.phase == 'train':
    logger.info('========= The Current Training Strategy is {} ========='.format(config['training_opt']['type']))
    train_func = train_loader(config)
    training = train_func(args, config, logger, eval=args.require_eval)
    training.run()


else:
    # ============================================================================
    # create model
    logger.info('=====> Model construction from: ' + str(config['networks']['type']))
    model_type = config['networks']['type']
    model_file = config['networks'][model_type]['def_file']
    model_args = config['networks'][model_type]['params']
    logger.info('=====> Classifier construction from: ' + str(config['classifiers']['type']))
    classifier_type = config['classifiers']['type']
    classifier_file = config['classifiers'][classifier_type]['def_file']
    classifier_args = config['classifiers'][classifier_type]['params']
    model = utils.source_import(model_file).create_model(**model_args)
    classifier = utils.source_import(classifier_file).create_model(**classifier_args)

    model = nn.DataParallel(model).cuda()
    classifier = nn.DataParallel(classifier).cuda()

    # ============================================================================
    # load checkpoint
    checkpoint = Checkpoint(config)
    ckpt = checkpoint.load(model, classifier, args.load_dir, logger)

    # ============================================================================
    # testing
    test_func = test_loader(config)
    if args.phase == 'val':
        # run validation set
        testing = test_func(config, logger, model, classifier, val=True, add_ckpt=ckpt)
        testing.run_val(epoch=-1)
    else:
        assert args.phase == 'test'
        # Run a specific test split
        if args.testset:
            testing = test_func(config, logger, model, classifier, val=False, add_ckpt=ckpt)
            testing.run_test()
        # Run all test splits
        else:
            if 'LT' in config['dataset']['name']:
                config['dataset']['testset'] = 'test_lt'
                testing = test_func(config, logger, model, classifier, val=False, add_ckpt=ckpt)
                testing.run_test()

            config['dataset']['testset'] = 'test_bl'
            testing = test_func(config, logger, model, classifier, val=False, add_ckpt=ckpt)
            testing.run_test()

            config['dataset']['testset'] = 'test_bbl'
            testing = test_func(config, logger, model, classifier, val=False, add_ckpt=ckpt)
            testing.run_test()

logger.info('========================= Complete =========================')
