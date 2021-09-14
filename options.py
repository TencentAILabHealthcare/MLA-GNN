import argparse
import os

import torch


### Parser

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of batches to train/test for. Default: 64')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--model_dir', type=str, default='./pretrained_models', help='models are saved here')
    parser.add_argument('--results_dir', type=str, default='./results', help='models are saved here')

    parser.add_argument('--lambda_cox', type=float, default=1)
    parser.add_argument('--lambda_reg', type=float, default=3e-4)
    parser.add_argument('--lambda_nll', type=float, default=1)

    parser.add_argument('--task', type=str, default='grad', help='surv | grad')
    parser.add_argument('--label_dim', type=int, default=3, help='size of output, grad task: label_dim=2, surv task: label_dim=1')
    parser.add_argument('--input_dim', type=int, default=1, help="input_size for omic vector")
    parser.add_argument('--lin_input_dim', type=int, default=720, help="the feature extracted by GAT layers")
    parser.add_argument('--which_layer', type=str, default='all', help='layer1 | layer2 | layer3, which GAT layer as the input of fc layers.')
    parser.add_argument('--cnv_dim', type=int, default=0, help="if use CNV as input, dim=80, if do not use CNV, dim=0")
    parser.add_argument('--omic_dim', type=int, default=32, help="dimension of the linear layer")
    parser.add_argument('--act_type', type=str, default="none", help='activation function')


    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--lr', default=0.002, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float, help='Used for AdaBound')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Used for Adam. L2 Regularization on weights.')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate')
    parser.add_argument('--adj_thresh', default=0.08, type=float, help='Threshold convert the similarity matrix to adjacency matrix')
    parser.add_argument('--alpha', default=0.2, type=float, help='Used in the leaky relu')
    parser.add_argument('--patience', default=0.005, type=float)
    parser.add_argument('--gpu_ids', type=str, default='3,4,5', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    opt = parser.parse_known_args()[0]
    print_options(parser, opt)

    return opt


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    mkdirs(opt.model_dir)
    file_name = os.path.join(opt.model_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
