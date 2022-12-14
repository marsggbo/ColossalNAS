# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from hyperbox.networks.darts import DartsNetwork
from hyperbox.networks.ofa import OFAMobileNetV3
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.mutables import spaces, ops, layers
from hyperbox.mutator import RandomMutator
from vit import ViT

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class ToyNASModel(BaseNASNetwork):
    def __init__(self, model_init_func=None, is_search_inner=False, rank=None, mask=None):
        super().__init__(mask)
        ks = spaces.ValueSpace([3, 5, 7])
        self.conv1 = ops.Conv2d(3, 512, kernel_size=ks, stride=1, padding=1, bias=False)
        # self.conv1 = torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv3 = torch.nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv1 = OperationSpace(
        #     [torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #      torch.nn.Conv2d(3, 512, kernel_size=5, stride=1, padding=2, bias=False),
        #     #  torch.nn.Conv2d(3, 512, kernel_size=7, stride=1, padding=3, bias=False),
        #     ], key='conv1', mask=self.mask
        # )
        # self.conv2 = OperationSpace(
        #     [torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        #      torch.nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False),
        #     #  torch.nn.Conv2d(512, 1024, kernel_size=7, stride=1, padding=3, bias=False),
        #     ], key='conv2', mask=self.mask
        # )
        # self.conv3 = OperationSpace(
        #     [torch.nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),
        #      torch.nn.Conv2d(1024, 2048, kernel_size=5, stride=1, padding=2, bias=False),
        #     #  torch.nn.Conv2d(1024, 2048, kernel_size=7, stride=1, padding=3, bias=False),
        #     ], key='conv3', mask=self.mask
        # )
        self.gavg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, 1000, bias=False)
        self.is_search_inner = is_search_inner
        self.rank = rank
        self.count = 0

    def forward(self, x):
        out = self.conv1(x)
        # out = self.conv2(out)
        # out = self.conv3(out)
        out = self.gavg(out).view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    from ipdb import set_trace
    # set_trace()
    setup(0, 1)
    # net = ToyNASModel().cuda()
    # net = DartsNetwork().cuda()
    # net = OFAMobileNetV3(
    #     base_stage_width=[16, 96, 128, 240, 320, 1280],
    #     stride_stages=[1, 1, 2, 1],
    #     act_stages=['relu', 'h_swish', 'h_swish', 'h_swish'],
    #     se_stages=[False, True, False, False],
    #     depth_list=[1, 2]
    # )
    base_stage_width=[16, 24, 32, 40, 80, 112, 160, 160, 960, 1280]
    stride_stages=[1, 2, 2, 2, 1, 1, 2]
    act_stages=['relu', 'relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
    se_stages=[False, False, True, True, False, True, True]
    depth_list=[1, 2, 3]
    remove_indices = [1] # 1~5
    for i in remove_indices:
        base_stage_width.pop(i)
        stride_stages.pop(i)
        act_stages.pop(i)
        se_stages.pop(i)
    net = OFAMobileNetV3(
        base_stage_width=base_stage_width,
        stride_stages=stride_stages,
        act_stages=act_stages,
        se_stages=se_stages,
        depth_list=depth_list,
    )
    # net = OFAMobileNetV3()
    net = net.cuda()
    # op1 = net.stem_layer
    # op2 = net.blocks[0]
    # net = torch.nn.Sequential(op1, op2).cuda()
    # net = ViT(image_size=224, patch_size=16, num_classes=1000, dim=1024,
    # depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1).cuda()
    
    # ks = spaces.ValueSpace([3, 5, 7])
    # net = ops.Conv2d(3, 512, kernel_size=ks, stride=1, padding=1, bias=False)
    
    # cout = spaces.ValueSpace([16, 32, 64, 128, 256, 512])
    # net = layers.MBConvLayer(3, cout, 3, 1, 1, 6, use_se=True, act_func='relu6').cuda()
        
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )
    fsdp_net = FSDP(
        net,
        auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
        mixed_precision=fpSixteen,
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    )
    rm = RandomMutator(net)
    x = torch.randn(1, 3, 224, 224).to(0)
    for i in range(10):
        rm.reset()
        y = fsdp_net(x)
        print(y.shape)