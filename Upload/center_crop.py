import glob
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

im = './data_raw/sar_images/train'
ma = './data_raw/zones/train'

imv = './data_raw/sar_images/valid'
mav = './data_raw/zones/valid'

imt = './data_raw/sar_images/test'
mat = './data_raw/zones/test'

images = './data_raw/sar_images/center_crop'
masks = './data_raw/zones/center_crop'

imagesv = './data_raw/sar_images/center_crop_valid'
masksv = './data_raw/zones/center_crop_valid'

imagest = './data_raw/sar_images/center_crop_test'
maskst = './data_raw/zones/center_crop_test'

if not os.path.exists(images):
    os.makedirs(images)
if not os.path.exists(masks):
    os.makedirs(masks)
if not os.path.exists(imagesv):
    os.makedirs(images)
if not os.path.exists(masksv):
    os.makedirs(masks)
if not os.path.exists(imagest):
    os.makedirs(images)
if not os.path.exists(maskst):
    os.makedirs(masks)
    
ims = sorted(glob.glob(os.path.join(im, '*.png')))
mas = sorted(glob.glob(os.path.join(ma, '*.png')))

imsv = sorted(glob.glob(os.path.join(imv, '*.png')))
masv = sorted(glob.glob(os.path.join(mav, '*.png')))

imst = sorted(glob.glob(os.path.join(imt, '*.png')))
mast = sorted(glob.glob(os.path.join(mat, '*.png')))


target_size = 224
context_size = target_size * 2

for i in range(len(ims)):
    img = Image.open(ims[i])
    suffix = ims[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, padding_mode='symmetric')(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(images, suffix))

for i in range(len(mas)):
    img = Image.open(mas[i])
    suffix = mas[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, padding_mode='symmetric')(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(masks, suffix))

for i in range(len(imsv)):
    img = Image.open(imsv[i])
    suffix = imsv[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, padding_mode='symmetric')(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(imagesv, suffix))

for i in range(len(masv)):
    img = Image.open(masv[i])
    suffix = masv[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, padding_mode='symmetric')(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(masksv, suffix))

for i in range(len(imst)):
    img = Image.open(imst[i])
    suffix = imst[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, padding_mode='symmetric')(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(imagest, suffix))

for i in range(len(mast)):
    img = Image.open(mast[i])
    suffix = mast[i].split('/')[-1]
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, padding_mode='symmetric')(img)

    pil_img = transforms.ToTensor()(pil_img)
    save_image(pil_img, os.path.join(maskst, suffix))
