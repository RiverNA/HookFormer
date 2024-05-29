import glob
import os
import PIL
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ip = './data_raw/sar_images/center_crop'
mp = './data_raw/zones/center_crop'

images = sorted(glob.glob(os.path.join(ip, '*.png')))
masks = sorted(glob.glob(os.path.join(mp, '*.png')))

save_path_context = './Train/Transform_images'
save_path_context_mask = './Train/Transform_masks'

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

target_size = 224
interval = target_size * 2

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:03d}.png'.format(id)))
                id += 1

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0][0:-6]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:03d}'.format(id) + '_zones_NA.png'))
                id += 1

ipv = './data_raw/sar_images/center_crop_valid'
mpv = './data_raw/zones/center_crop_valid'

imagesv = sorted(glob.glob(os.path.join(ipv, '*.png')))
masksv = sorted(glob.glob(os.path.join(mpv, '*.png')))

save_path_context = './Valid/Transform_images'
save_path_context_mask = './Valid/Transform_masks'

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(imagesv)):
    image = Image.open(imagesv[i])
    suffix = imagesv[i].split('/')[-1].split('.')[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:03d}.png'.format(id)))
                id += 1

for i in range(len(masksv)):
    mask = Image.open(masksv[i])
    suffix = masksv[i].split('/')[-1].split('.')[0][0:-6]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:03d}'.format(id) + '_zones_NA.png'))
                id += 1

ipt = './data_raw/sar_images/center_crop_test'
mpt = './data_raw/zones/center_crop_test'

imagest = sorted(glob.glob(os.path.join(ipt, '*.png')))
maskst = sorted(glob.glob(os.path.join(mpt, '*.png')))

save_path_context = './Test/Transform_images'
save_path_context_mask = './Test/Transform_masks'

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(imagest)):
    image = Image.open(imagest[i])
    suffix = imagest[i].split('/')[-1].split('.')[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:03d}.png'.format(id)))
                id += 1

for i in range(len(maskst)):
    mask = Image.open(maskst[i])
    suffix = maskst[i].split('/')[-1].split('.')[0][0:-6]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:03d}'.format(id) + '_zones_NA.png'))
                id += 1
