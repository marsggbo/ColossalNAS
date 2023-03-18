
from functools import partial

import torch
import torch.nn as nn
from torchvision import models

from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.mutables.spaces import OperationSpace
from hyperbox.mutator import RandomMutator

from hyperbox.networks.ofa import OFAMobileNetV3
from hyperbox.networks.darts import DartsNetwork
from hyperbox.networks.vit import *
from hyperbox.networks.mobilenet.mobile_net import MobileNet


def get_vit(cls=VisionTransformer, **kwargs):
    default_config = {
        'image_size': 224, 'patch_size': 16, 'num_classes': 1000, 'dim': 1024, 
        'depth': 6, 'heads': 16, 'dim_head': 1024, 'mlp_dim': 2048
    }
    default_config.update(kwargs)
    return cls(**default_config)

def get_darts(**kwargs):
    default_config = {
        'in_channels': 3, 'channels': 64, 'n_classes': 1000, 'n_layers': 8, 'auxiliary': False,
        'n_nodes': 4, 'stem_multiplier': 3
    }
    default_config.update(kwargs)
    return DartsNetwork(**default_config)

def get_ofa(**kwargs):
    default_config = {
        'width_mult': 1.0, 'depth_list': [4,5], 'expand_ratio_list': [2,3],
        'base_stage_width': [16, 32, 64, 128, 256, 320, 480, 512, 960],
        'to_search_depth': False
    }
    # base_stage_width=[32, 64, 128, 256, 512, 512, 512, 960, 1024]
    default_config.update(kwargs)
    return OFAMobileNetV3(**default_config)


class ToyNASModel2(BaseNASNetwork):
    def __init__(self, model_init_func=None, is_search_inner=False, rank=None, mask=None):
        super().__init__(mask)
        self.flag = 2
        if self.flag==1:
            self.conv1 = torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False)
        else:
            self.conv1 = OperationSpace(
                [torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.Conv2d(3, 512, kernel_size=5, stride=1, padding=2, bias=False),
                torch.nn.Conv2d(3, 512, kernel_size=7, stride=1, padding=3, bias=False),
                ], key='conv1', mask=self.mask
            )
            self.conv2 = OperationSpace(
                [torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False),
                torch.nn.Conv2d(512, 1024, kernel_size=7, stride=1, padding=3, bias=False),
                ], key='conv2', mask=self.mask
            )
            self.conv3 = OperationSpace(
                [torch.nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),
                 torch.nn.Conv2d(1024, 2048, kernel_size=5, stride=1, padding=2, bias=False),
                #  torch.nn.Conv2d(1024, 2048, kernel_size=7, stride=1, padding=3, bias=False),
                ], key='conv3', mask=self.mask
            )
        self.gavg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1024, 1000, bias=False)
        self.is_search_inner = is_search_inner
        self.rank = rank
        self.count = 0

    def forward(self, x):
        out = self.conv1(x)
        if self.flag==1:
            if self.count % 2 == 0:
                out = self.conv2(out)
            else:
                out = self.conv3(out)
        else:
            out = self.conv2(out)
        self.count += 1
        out = self.gavg(out).view(out.size(0), -1)
        out = self.fc(out)
        return out


class ToyNASModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False)
        self.gavg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1024, 1000, bias=False)
        self.count = 0

    def forward(self, x):
        out = self.conv1(x)
        if self.count % 2 == 0:
            out = self.conv2(out)
        else:
            out = self.conv3(out)
        self.count += 1
        out = self.gavg(out).view(out.size(0), -1)
        out = self.fc(out)
        return out


name2model = {
    'ofa': get_ofa,
    'vit_s': ViT_S,
    'vit_b': ViT_B,
    'vit_h': ViT_H,
    'vit_g': ViT_G,
    'vit_10b': ViT_10B,
    'darts': get_darts,
    'toy': ToyNASModel,
    'toy2': ToyNASModel2,
    'mobilenet': MobileNet,
    'resnet18': models.resnet18,
    'resnet152': models.resnet152,
}

def get_model(name, **kwargs):
    return name2model[name](**kwargs)


if __name__ == '__main__':
    net = get_model('ofa')
