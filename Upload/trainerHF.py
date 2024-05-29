import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from Eval_whole import eval_net
from train_dataset import BasicDataset
from valid_dataset import BasDataset

scratch_train = os.path.join(os.environ['TMPDIR'], 'Training')
scratch_valid = os.path.join(os.environ['TMPDIR'], 'Validation')

dir_img_target = os.path.join(scratch_train, 'target_images')
dir_img_context = os.path.join(scratch_train, 'context_images')
dir_mask_target = os.path.join(scratch_train, 'target_masks')
dir_mask_context = os.path.join(scratch_train, 'context_masks')

valid_img_target = os.path.join(scratch_valid, 'target_images')
valid_img_context = os.path.join(scratch_valid, 'context_images')
valid_mask_target = os.path.join(scratch_valid, 'target_masks')
valid_mask_context = os.path.join(scratch_valid, 'context_masks')


def trainer_HookFormer(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train_dataset = BasicDataset(dir_img_target=dir_img_target, dir_mask_target=dir_mask_target, dir_img_context=dir_img_context, dir_mask_context=dir_mask_context)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)
    n_dataset = len(train_dataset)
    valid_dataset = BasDataset(dir_img_target=valid_img_target, dir_mask_target=valid_mask_target, dir_img_context=valid_img_context, dir_mask_context=valid_mask_context)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_path = None
    if load_path:
        checkpoints = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        eepoch = checkpoints['epoch']
        iter_num = checkpoints['ite_num']
        iterator = tqdm(range(max_epoch - eepoch), ncols=70)
        print(f'Model loaded from {load_path}')
    else:
        iterator = tqdm(range(max_epoch), ncols=70)
        iter_num = 0

    for epoch_num in iterator:
        epoch_loss = 0
        print('\n' + 'lr:', optimizer.param_groups[0]['lr'])
        for i_batch, sampled_batch in enumerate(train_loader):
            image_target, label_target, image_context, label_context = sampled_batch['image_target'].cuda(), sampled_batch['mask_target'].cuda(), sampled_batch['image_context'].cuda(), sampled_batch['mask_context'].cuda()
            masks = [label_target]
            for i in range(4):
                big_mask = masks[-1]
                small_mask = F.avg_pool2d(big_mask.cpu(), 2)
                masks.append(small_mask)
            outputs = model(image_target, image_context)
            loss_cet = ce_loss(outputs[0], label_target)
            loss_cec = ce_loss(outputs[1], label_context)
            loss_dicet = dice_loss(outputs[0], label_target, softmax=True)
            loss_dicec = dice_loss(outputs[1], label_context, softmax=True)
            loss_ = 0.5 * loss_cet + 0.5 * loss_cec + 0.5 * loss_dicet + 0.5 * loss_dicec
            loss_ced = ce_loss(outputs[2], masks[-1][:].cuda())
            loss_diced = dice_loss(outputs[2], masks[-1][:].cuda(), softmax=True)
            loss_deep = 0.5 * loss_ced + 0.5 * loss_diced
            loss = loss_ + 0.5 * loss_deep
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
        print('Train average loss:', epoch_loss / n_dataset)
        valid_iou_ratio, iou = eval_net(model, valid_loader, device)
        print('Valid iou ratio: {}, {}'.format(valid_iou_ratio, iou))

        if load_path:
            save_mode_path = os.path.join(snapshot_path, 'HookFormer_epoch{:03d}.pth'.format(epoch_num + eepoch + 1))
            print("save model to {}".format(save_mode_path))
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch_num + eepoch + 1,
                        'ite_num': iter_num,
                        }, save_mode_path)
        else:
            save_mode_path = os.path.join(snapshot_path, 'HookFormer_epoch{:03d}.pth'.format(epoch_num + 1))
            print("save model to {}".format(save_mode_path))
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch_num + 1,
                        'ite_num': iter_num,
                        }, save_mode_path)

    return "Training Finished!"
