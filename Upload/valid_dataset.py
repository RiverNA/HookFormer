import os
import torch
import cv2
import logging
import time
import torchvision
import tormentor
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class BasDataset(Dataset):
    def __init__(self, dir_img_target, dir_mask_target, dir_img_context, dir_mask_context, mask_suffix='_zones_NA'):
        self.imgs_target = dir_img_target
        self.imgs_context = dir_img_context
        self.masks_target = dir_mask_target
        self.masks_context = dir_mask_context
        self.mask_suffix = mask_suffix
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(dir_img_target)]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def Fliph(self, target_image, context_image, target_mask, context_mask):
        target = torchvision.transforms.functional.hflip(target_image)
        mask_target = torchvision.transforms.functional.hflip(target_mask)
        context = torchvision.transforms.functional.hflip(context_image)
        mask_context = torchvision.transforms.functional.hflip(context_mask)
        return target, context, mask_target, mask_context

    def Flipv(self, target_image, context_image, target_mask, context_mask):
        target = torchvision.transforms.functional.vflip(target_image)
        mask_target = torchvision.transforms.functional.vflip(target_mask)
        context = torchvision.transforms.functional.vflip(context_image)
        mask_context = torchvision.transforms.functional.vflip(context_mask)
        return target, context, mask_target, mask_context

    def Rotate(self, target_image, context_image, target_mask, context_mask):
        random = np.random.randint(0, 3)
        angle = 90
        if random == 1:
            angle = 180
        elif random == 2:
            angle = 270
        target = torchvision.transforms.functional.rotate(target_image, angle=angle)
        context = torchvision.transforms.functional.rotate(context_image, angle=angle)
        mask_target = torchvision.transforms.functional.rotate(target_mask.unsqueeze(0), angle=angle)
        mask_context = torchvision.transforms.functional.rotate(context_mask.unsqueeze(0), angle=angle)

        return target, context, mask_target.squeeze(0), mask_context.squeeze(0)

    def preprocess(self, img_nd):
        img_nd = img_nd.transpose((2, 0, 1))
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        img_trans = img_nd
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def mask_preprocess(self, img_nd):
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans[0, :, :]

        C, _ = img_trans.shape
        mask = np.ones([C, C]) * 15
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

    def __getitem__(self, i):
        idx = self.ids[i]
        imgs_target = glob(self.imgs_target + '/' + idx + '.*')
        imgs_context = glob(self.imgs_context + '/' + idx + '.*')
        masks_target = glob(self.masks_target + '/' + idx + self.mask_suffix + '.*')
        masks_context = glob(self.masks_context + '/' + idx + self.mask_suffix + '.*')

        suffix = idx + self.mask_suffix
        assert len(masks_target) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_target}'
        assert len(imgs_target) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_target}'

        mask_target = cv2.imread(masks_target[0])
        target = cv2.imread(imgs_target[0])
        mask_context = cv2.imread(masks_context[0])
        context = cv2.imread(imgs_context[0])

        assert target.size == mask_target.size, \
            f'Image and mask {idx} should be the same size, but are {target.size} and {mask_target.size}'

        target = self.preprocess(target)
        context = self.preprocess(context)
        target = torch.from_numpy(target).type(torch.float32)
        context = torch.from_numpy(context).type(torch.float32)

        mask_target = self.mask_preprocess(mask_target)
        mask_context = self.mask_preprocess(mask_context)
        mask_target = torch.from_numpy(mask_target).type(torch.int64)
        mask_context = torch.from_numpy(mask_context).type(torch.int64)

        return {
            'image_target': target,
            'mask_target': mask_target,
            'image_context': context,
            'mask_context': mask_context,
            'suffix': suffix,
        }
