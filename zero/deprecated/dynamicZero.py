
import colossalai
import psutil
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from transformers import GPT2Config, GPT2LMHeadModel
from time import time
from functools import partial

from ipdb import set_trace


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'

def DeviceInfo(model):
    print('p.name, p.dtype, p.shape, p.grad.dtype, p.grad.shape')
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_device, grad_shape = p.grad.dtype, p.grad.shape
        else:
            grad_device, grad_shape = None, None
        if hasattr(p, 'colo_attr'):
            print(name, p.dtype, p.colo_attr.data_payload.shape, grad_device, grad_shape)
        else:
            print(name, p.dtype, p.shape, grad_device, grad_shape)

class NASModel(nn.Module):
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

def main():
    BATCH_SIZE = 8
    NUM_STEPS = 50
    use_zero = True
    # use_zero = False
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    logger = get_dist_logger()
    grank = gpc.get_global_rank()

    logger.info(get_mem_info(), ranks=[0])
    # build GPT model
    if use_zero:
        shard_strategy = TensorShardStrategy()
        set_trace()
        with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy, shard_param=True) as ctx:
            model = NASModel()
        numel = ctx.model_numel_tensor.item()
        logger.info(f'Model numel: {numel}', ranks=[0])
        # Set tensor_placement_policy='cpu', which will offload params, grads and os
        model = ShardedModelV2(model, shard_strategy, tensor_placement_policy='cuda', reuse_fp16_shard=True)
        logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    else:
        model = NASModel().to(torch.cuda.current_device())
        numel = sum([p.numel() for p in model.parameters()])

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
        optimizer.zero_grad()
        start = time()
        logger.info(f'rank {grank}: before foward')
        DeviceInfo(model)
        outputs = model(x)
        loss = (outputs-y).sum()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Forward '), ranks=[0])
        logger.info(f'rank {grank}: before backward')
        DeviceInfo(model)
        if use_zero:
            optimizer.backward(loss)
        else:
            loss.backward()
        logger.info(f'rank {grank}: before step')
        DeviceInfo(model)
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Backward '), ranks=[0])
        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        logger.info(f'rank {grank}: after step')
        DeviceInfo(model)
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s', ranks=[0])
        logger.info('\n\n')

if __name__ == '__main__':
    main()
    print('='*20)
    print('='*20)
    print('done')
