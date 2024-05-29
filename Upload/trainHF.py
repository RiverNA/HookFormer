import argparse
import logging
import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from ViT import HookFormer
from trainerHF import trainer_HookFormer
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='./checkpoints', help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=110, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE",
                    help='path to config file', )
parser.add_argument("--opts",
                    help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None,
                    nargs='+',
                    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default='./swin_tiny_patch4_window7_224.pth', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--local_rank', type=int, default=0)

args = parser.parse_args()
config = get_config(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cudnn.benchmark = True
    cudnn.deterministic = False
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = HookFormer(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_from(config)
    trainer_HookFormer(args, model, args.output_dir)
