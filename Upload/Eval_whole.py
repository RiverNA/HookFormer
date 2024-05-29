import torch
import torch.nn.functional as F
import os
import torchmetrics
import logging
import glob
import sys
import cv2
import numpy as np
import sklearn.metrics
from tqdm import tqdm
from dice_loss import dice_coeff
from loss import make_one_hot
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

test_save = './patch_results'


def whole_preprocess(pil_img):
    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    img_trans = img_nd[:, :, 0]
    H, W = img_trans.shape
    mask = np.ones([H, W]) * 15
    stone = np.where(img_trans == 0)
    na_area = np.where(img_trans == 63)
    na_areas = np.where(img_trans == 64)
    glacier = np.where(img_trans == 127)
    ocean_ice = np.where(img_trans == 254)
    mask[stone] = 0
    mask[na_area] = 1
    mask[na_areas] = 1
    mask[glacier] = 2
    mask[ocean_ice] = 3

    return mask


def mask_to_image(pre_mask, save_path, suffix):
    c, h, w = pre_mask.shape
    out = np.zeros((3, h, w), dtype=np.uint8)
    na_area = np.where(pre_mask.to('cpu') == 0)
    stone = np.where(pre_mask.to('cpu') == 1)
    glacier = np.where(pre_mask.to('cpu') == 2)
    ocean_ice = np.where(pre_mask.to('cpu') == 3)
    out[na_area] = 0
    out[stone] = 63
    out[glacier] = 127
    out[ocean_ice] = 254
    out[1] = out[0]
    out[2] = out[0]
    out = out.transpose((1, 2, 0))
    cv2.imwrite(os.path.join(save_path, suffix + '.png'), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


def eval_net(model, loader, device):
    model.eval()
    mask_type = torch.float32 if model.num_classes <= 2 else torch.int64
    n_val = len(loader)
    iou_ratio = 0
    if not os.path.exists(test_save):
        os.makedirs(test_save)
    with tqdm(total=n_val, desc='Round', unit='img') as pbar:
        for batch in loader:
            imgs_target = batch['image_target']
            imgs_context = batch['image_context']
            true_masks_target = batch['mask_target']
            suffix = batch['suffix'][0]
            imgs_target = imgs_target.to(device=device, dtype=torch.float32)
            imgs_context = imgs_context.to(device=device, dtype=torch.float32)
            true_masks_target = true_masks_target.to(device=device, dtype=mask_type)
            with torch.no_grad():
                masks_pred = model(imgs_target, imgs_context)

            if model.num_classes > 2:
                IOU = torchmetrics.IoU(num_classes=4, absent_score=1)
                prd_target = F.log_softmax(masks_pred[0], dim=1)
                prd_target = torch.argmax(prd_target, dim=1)
                mask_to_image(prd_target, test_save, suffix)
                iou = IOU(prd_target.cpu().detach(), true_masks_target.cpu().detach())
                iou_ratio += iou

            else:
                IOU = torchmetrics.IoU(num_classes=2, absent_score=1)
                pred = torch.sigmoid(masks_pred[0])
                pred = (pred > 0.5).float()
                iou = IOU(pred.cpu().detach(), true_masks_target.type(torch.int64).cpu().detach())
                iou_ratio += iou
            pbar.update()

    groundtruth = './Valid/Validation/ground_truth'
    whole_save = './whole_results'
    masks_ground = sorted(glob.glob(os.path.join(groundtruth, '*.png')))
    totensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    if not os.path.exists(whole_save):
        os.makedirs(whole_save)
    IOU = torchmetrics.IoU(num_classes=4, absent_score=1, reduction='none')
    iou_ratio = 0
    for i in range(len(masks_ground)):
        ground = masks_ground[i]
        suffix = ground.split('/')[-1].split('.')[0][0:-6]
        img = Image.open(ground)
        W, H = img.size
        HH = H // 224 + 1
        WW = W // 224 + 1
        length = 224
        test = sorted(glob.glob(os.path.join(test_save, suffix + '*.png')))
        all = []
        for j in range(len(test)):
            ti = Image.open(test[j])
            all.append(ti)
        whole = Image.new('RGB', (WW * length, HH * length))
        for k in range(len(all)):
            whole.paste(all[k],
                        (length * (k % WW), length * (k // WW), length * (k % WW + 1), length * (k // WW + 1)))
        whole = transforms.CenterCrop((H, W))(whole)
        wholes = totensor(whole)
        save_image(wholes, os.path.join(whole_save, suffix + '.png'))
        img = whole_preprocess(img)
        whole = whole_preprocess(whole)
        iou = IOU(torch.from_numpy(whole).type(torch.int64), torch.from_numpy(img).type(torch.int64))
        iou_ratio += iou
    iou_ratio = iou_ratio / len(masks_ground)
    ave_iou = sum(iou_ratio) / len(iou_ratio)
    model.train()
    
    return ave_iou, iou_ratio
