# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
# parser.add_argument('dataset', type=str)
parser.add_argument('--dataset', type=str, default="HockeyFights")
# parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--modality', type=str, default="RGB", choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
# ========================= Model Configs ==========================
# parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('--arch', type=str, default="mobilenetv2")
# parser.add_argument('--arch', type=str, default="mobilenetv2x2")
# parser.add_argument('--arch', type=str, default="mobilenetv2x3")
# parser.add_argument('--arch', type=str, default="mobilenetv2x4")
# parser.add_argument('--arch', type=str, default="eca_mobilenet_v2")
# parser.add_argument('--arch', type=str, default="BNInception")
# parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
# parser.add_argument('--dropout', '--do', default=0, type=float,
#                     metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
# parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')
parser.add_argument('--tune_from', type=str, default="./pretrained/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth", help='fine-tune from checkpoint')
# parser.add_argument('--tune_from', type=str, default="./TSM-3/4/HockeyFights/ckpt.best.pth.tar", help='fine-tune from checkpoint')
# parser.add_argument('--tune_from', type=str,
#                     default="./pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth",
#                     help='fine-tune from checkpoint')
# parser.add_argument('--tune_from', type=str,
#                     default="./pretrained/resnext50_32x4d-7cdf4587.pth",
#                     help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('-b', '--batch-size', default=8, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=0, type=float,
#                     metavar='LR', help='initial learning rate')#[1e-6 1e-1]
parser.add_argument('--lr', '--learning-rate', default=0.000375, type=float,
                    metavar='LR', help='initial learning rate')#[1e-6 1e-1]
# parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
#                     metavar='LR', help='initial learning rate')#[1e-6 1e-1]
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')#[0.85,0.95]
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')#[1e-6,1e-2]
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default='/home/yong-group/文档/zys/TSM/temporal-shift-module-master-1.11/TSM-3/4//-1.13/ckpt.best.pth.tar', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
# parser.add_argument('--shift', default=True, action="store_false", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
# parser.add_argument('--dense_sample', default=True, action="store_true", help='use dense sample for video dataset')

#tensorboard --logdir=

