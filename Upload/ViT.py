from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from os.path import join as pjoin
from HookFormer_parts import Model


class HookFormer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False):
        super(HookFormer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.HookFormer = Model(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                c_depths=config.MODEL.SWIN.C_DEPTHS,
                                t_depths=config.MODEL.SWIN.T_DEPTHS,
                                CAST_depths=config.MODEL.SWIN.CAST_DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                c_window_size=config.MODEL.SWIN.C_WINDOW_SIZE,
                                t_window_size=config.MODEL.SWIN.T_WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x, y):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size()[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        logits = self.HookFormer(x, y)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.RESUME
        if pretrained_path is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.HookFormer.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']

            model_dict = self.HookFormer.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_k = "c_layers." + k[7:]
                    full_dict.update({current_k: v})

                    current_k = "t_layers." + k[7:]
                    full_dict.update({current_k: v})

                    current_layer_num = 3 - int(k[7:8])
                    current_k = "c_layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})

                    current_layer_num = 3 - int(k[7:8]) - 1
                    current_k = "t_layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})

            for k, v in pretrained_dict.items():
                if "patch_embed." in k:
                    current_k = "c_patch_embed." + k[12:]
                    full_dict.update({current_k: v})

                    current_k = "t_patch_embed." + k[12:]
                    full_dict.update({current_k: v})

            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        del full_dict[k]

            self.HookFormer.load_state_dict(full_dict, strict=False)