
import colossalai
import psutil
import torch
import torch.nn as nn
import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.core import global_context as gpc
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
    BS = 64
    NUM_STEPS = 20
    parser = colossalai.get_default_parser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--useZero', type=int, default=1) # 1: True 0: False
    parser.add_argument('--searchspace', type=str, default='ofa_masks.pth') # 1: True 0: False
    parser.add_argument('--debug', type=int, default=0) # 1: True 0: False
    args = parser.parse_args()
    # if args.get('config', None) is None:
    if args.config is None:
        args.config = {}

    # launch from torch
    colossalai.launch_from_torch(config=args.config)
    grank = gpc.get_global_rank()
    use_zero = args.useZero
    gpus = args.gpus
    searchspace_path = args.searchspace
    searchspace = torch.load(searchspace_path)
    debug = args.debug
    # use_zero = False
    if use_zero:
        log_flag = 'zero'
    else:
        log_flag = 'ddp'
    disable_existing_loggers()
    logger = get_dist_logger()
    logger.log_to_file(f'exp_{log_flag}_gpu{gpus}/log_rank{grank}', mode='w')
    logger.info("initialized distributed environment")

    logger.info(get_mem_info())
    if debug:
        set_trace()
    # build NAS model
    if use_zero:
        shard_strategy = TensorShardStrategy()
        with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy, shard_param=True) as ctx:
            model = get_ofa(4)
            # model = NASModel()
        numel = ctx.model_numel_tensor.item()
        logger.info(f'Model numel: {numel}')
        # Set tensor_placement_policy='cpu', which will offload params, grads and os
        model = ShardedModelV2(model, shard_strategy, tensor_placement_policy='cuda', reuse_fp16_shard=True)
        logger.info(get_mem_info(prefix='After init model, '))
    else:
        model = get_ofa(4).to(torch.cuda.current_device())
        # model = NASModel()
        numel = sum([p.numel() for p in model.parameters()])
    rm = RandomMutator(model)
    # rm.reset()

    # optimizer
    if use_zero:
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info(get_mem_info(prefix='After init optim, '))

    model.train()
    infos = '\nrank, batchIdx, batchTime, fw_g, bw_g, update_g, fw_c, bw_c, update_c\n'
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        x = torch.rand(BS, 3, 64, 64).to(torch.cuda.current_device())
        y = torch.rand(BS, 1000).to(torch.cuda.current_device())
        mask = searchspace[n]
        rm.sample_by_mask(mask)
        # rm.reset()
        logger.info(f'rank: {grank}, mask: {mask}')
        optimizer.zero_grad()
        start = time()
        outputs = model(x)
        loss = (outputs-y).sum()
        fw_g, fw_c = get_gpu_mem(), get_cpu_mem()
        # logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Forward '))
        # print('before backward')
        # DeviceInfo(model)
        if use_zero:
            optimizer.backward(loss)
        else:
            loss.backward()
        bw_g, bw_c = get_gpu_mem(), get_cpu_mem()
        # print('after backward')
        # DeviceInfo(model)
        # logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Backward '))
        optimizer.step()
        update_g, update_c = get_gpu_mem(), get_cpu_mem()
        # logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '))
        # print('after step')
        # DeviceInfo(model)
        step_time = time() - start
        infos += f"{grank}, {n}, {step_time:.4f}, {fw_g:.4f}, {bw_g:.4f}, {update_g:.4f}, {fw_c:.4f}, {bw_c:.4f}, {update_c:.4f}\n"
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s')
    logger.info(infos)

if __name__ == '__main__':
    main()
