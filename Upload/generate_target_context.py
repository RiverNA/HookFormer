import glob
import os
import PIL
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ti = './Train/Transform_images'
mi = './Train/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = './Train/Training/target_images'
save_path_context = './Train/Training/context_images'
save_path_target_mask = './Train/Training/target_masks'
save_path_context_mask = './Train/Training/context_masks'

size = 224

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

ti = './Valid/Transform_images'
mi = './Valid/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = './Valid/Validation/target_images'
save_path_context = './Valid/Validation/context_images'
save_path_target_mask = './Valid/Validation/target_masks'
save_path_context_mask = './Valid/Validation/context_masks'

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

ti = './Test/Transform_images'
mi = './Test/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = './Test/Testing/target_images'
save_path_context = './Test/Testing/context_images'
save_path_target_mask = './Test/Testing/target_masks'
save_path_context_mask = './Test/Testing/context_masks'

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))
