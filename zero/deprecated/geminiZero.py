'''
# https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/gpt/train_gpt_demo.py
# commit f7e276f 
'''

from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel import ZeroDDP
from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from ipdb import set_trace
# from vit import ViT
from hyperbox.mutator import RandomMutator

def DeviceInfo(model):
    print('p.name, p.device, p.shape, p.grad.device, p.grad.shape')
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_device, grad_shape = p.grad.device, p.grad.shape
        else:
            grad_device, grad_shape = None, None
        if hasattr(p, 'colo_attr'):
            print(name, p.device, p.colo_attr.data_payload.shape, grad_device, grad_shape)
        else:
            print(name, p.device, p.shape, grad_device, grad_shape)

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

def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--distplan",
        type=str,
        default='colossalai',
        help="The distributed plan [colossalai, ddp, zero].",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor Parallelism Degree.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default='cpu',
        help="Placement Policy for Gemini.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Placement Policy for Gemini.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='vit',
        help="model name.",
    )
    parser.add_argument(
        "--shardinit",
        type=bool,
        default=False,
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    args = parser.parse_args()
    return args


## Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    if param.process_group.tp_world_size() == 1:
        param.set_process_group(pg)
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def getVit(**kwargs):
    model_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 1000,
        'dim': 1024,
        'depth': 2, # 6
        'heads': 1, # 16
        'mlp_dim': 2048,
        'dropout': 0.1,
        'emb_dropout': 0.1,
    }
    model_kwargs.update(kwargs)
    model = ViT(**model_kwargs)
    return model

def getModel(name, **kwargs):
    if name == 'vit':
        return getVit(**kwargs)
    elif name == 'nas':
        return NASModel()
    else:
        raise NotImplementedError


# Tensor Parallel
def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    """tensor_parallelize
    Sharding the Model Parameters.

    Args:
        model (torch.nn.Module): a torch module to be sharded
    """
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            # set process group for all parameters
            param.set_process_group(pg)

            if 'mlp.c_fc' in mn:
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)    # colmn slice
                    # keep the shape of the output from c_fc
                    param.compute_spec.set_output_replicate(False)
            elif 'mlp.c_proj' in mn:
                if 'weight' in pn:
                    split_param_row_tp1d(param, pg)    # row slice
            elif 'wte' in mn or 'wpe' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            elif 'c_attn' in mn or 'c_proj' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice


# Gemini + ZeRO DDP
def gemini_zero_dpp(model: torch.nn.Module, pg: ProcessGroup, placement_policy: str = "auto"):
    cai_version = colossalai.__version__
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.nn.parallel import GeminiDDP
        model = GeminiDDP(model,
                          device=get_current_device(),
                          placement_policy=placement_policy,
                          pin_memory=True,
                          search_range_mb=32)
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(cai_version) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
        gemini_manager = GeminiManager(placement_policy, chunk_manager)
        chunk_manager = ChunkManager(chunk_size,
                                     pg,
                                     enable_distributed_storage=True,
                                     init_device=GeminiManager.get_default_device(placement_policy))
        model = ZeroDDP(model, gemini_manager)
    else:
        raise NotImplemented(f"CAI version {cai_version} is not supported")
    return model


def main():
    args = parse_args()
    debug = args.debug
    if debug:
        set_trace()

    BATCH_SIZE = 8
    NUM_STEPS = 10

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    grank = gpc.get_global_rank()

    logger = get_dist_logger()
    logger.info(get_mem_info(), ranks=[0])

    if args.distplan == "colossalai":
        # all param must use the same process group.
        default_pg = ProcessGroup(tp_degree=args.tp_degree)
        default_dist_spec = ShardSpec([-1], [args.tp_degree]) if args.shardinit else None

        # build GPT model
        with ColoInitContext(device='cpu', default_dist_spec=default_dist_spec, default_pg=default_pg):
            model = getModel(args.model)
            # model = getVit()

        pg = default_pg
        # Tensor Parallelism (TP)
        tensor_parallelize(model, pg)
        # Gemini + ZeRO DP, Note it must be used after TP
        model = gemini_zero_dpp(model, pg, args.placement)

        # build optimizer
        optimizer = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=2**5)
        # optimizer = HybridAdam(model.parameters(), lr=1e-3)
        # optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**5)
        logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    elif args.distplan == "ddp":
        model = getModel(args.model).cuda()
        ddp_model = DDP(model)
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)

    elif args.distplan == "zero":
        from torch.distributed.optim import ZeroRedundancyOptimizer
        model = getModel(args.model).cuda()
        ddp_model = DDP(model)
        optimizer = ZeroRedundancyOptimizer(ddp_model.parameters(), optimizer_class=torch.optim.Adam, lr=0.01)
    else:
        raise TypeError(f"{args.distplan} is error")

    
    logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    torch.cuda.synchronize()
    model.train()
    rm = RandomMutator(model)
    for n in range(NUM_STEPS):
        x = torch.rand(2,3,224,224).to(torch.cuda.current_device())
        y = torch.rand(2, 1000).to(torch.cuda.current_device())
        optimizer.zero_grad()
        rm.reset()
        
        logger.info(f'rank {grank}: before foward')
        # DeviceInfo(model)
        start = time()
        outputs = model(x)
        loss = (outputs-y).sum()
        
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Forward '), ranks=[0])
        logger.info(f'rank {grank}: before backward')
        # DeviceInfo(model)
        if args.distplan == "colossalai":
            optimizer.backward(loss)
        elif args.distplan in ["ddp", "zero"]:
            loss.backward()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Backward '), ranks=[0])
        logger.info(f'rank {grank}: before step')
        # DeviceInfo(model)
        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        logger.info(f'rank {grank}: after step')
        # DeviceInfo(model)
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s',
            ranks=[0])

    torch.cuda.synchronize()


if __name__ == '__main__':
    main()