
import colossalai
import psutil
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from transformers import GPT2Config, GPT2LMHeadModel
from time import time
from functools import partial

from ipdb import set_trace
from hyperbox_app.distributed.networks.ofa import OFAMobileNetV3
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.mutables.spaces import OperationSpace
from hyperbox.mutator import RandomMutator


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_ofa(width_mult=4):
    return OFAMobileNetV3(
        width_mult=width_mult, depth_list=[4,5], expand_ratio_list=[2,3],
        base_stage_width=[16, 32, 64, 128, 256, 320, 480, 512, 960]
        # base_stage_width=[32, 64, 128, 256, 512, 512, 512, 960, 1024]
    )

def DeviceInfo(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(name, p.device, p.shape, p.grad.device, p.grad.shape)
        else:
            print(name, p.device, p.shape, None, None)

class NASModel(BaseNASNetwork):
    def __init__(self, model_init_func=None, is_search_inner=False, rank=None, mask=None):
        super().__init__(mask)
        self.conv1 = torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False)
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
        self.fc = torch.nn.Linear(1024, 1000, bias=False)
        self.is_search_inner = is_search_inner
        self.rank = rank
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

def main():
    BATCH_SIZE = 8
    NUM_STEPS = 6
    use_zero = True
    # use_zero = False
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    logger = get_dist_logger()

    logger.info(get_mem_info(), ranks=[0])
    # build GPT model
    if use_zero:
        shard_strategy = TensorShardStrategy()
        # set_trace()
        with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy, shard_param=True) as ctx:
            # model = get_ofa(4)
            model = NASModel()
        numel = ctx.model_numel_tensor.item()
        logger.info(f'Model numel: {numel}', ranks=[0])
        # Set tensor_placement_policy='cpu', which will offload params, grads and os
        model = ShardedModelV2(model, shard_strategy, tensor_placement_policy='cpu', reuse_fp16_shard=True)
        logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    else:
        # model = get_ofa(4).to(torch.cuda.current_device())
        model = NASModel()
        numel = sum([p.numel() for p in model.parameters()])
    rm = RandomMutator(model)
    rm.reset()

    # optimizer
    if use_zero:
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    model.train()
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        x = torch.rand(2,3,64,64).to(torch.cuda.current_device())
        y = torch.rand(2, 1000).to(torch.cuda.current_device())
        rm.reset()
        optimizer.zero_grad()
        start = time()
        outputs = model(x)
        loss = (outputs-y).sum()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Forward '), ranks=[0])
        print('before backward')
        DeviceInfo(model)
        if use_zero:
            optimizer.backward(loss)
        else:
            loss.backward()
        print('after backward')
        DeviceInfo(model)
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Backward '), ranks=[0])
        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        print('after step')
        DeviceInfo(model)
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s', ranks=[0])


if __name__ == '__main__':
    main()
