import os

import loguru
import pandas as pd
from time import time
from functools import partial
from prettytable import PrettyTable
from ipdb import set_trace
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
# DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
# FSDP
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

# Colossalai
import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.core import global_context as gpc
from colossalai.zero.shard_utils import (TensorShardStrategy,
                                         BucketTensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2

from hyperbox.mutator import RandomMutator

from models import get_model
from utils import get_peak_gpu_mem, get_gpu_mem, get_cpu_mem, get_mem_info, DeviceInfo


def main():
    batch_size = 64
    parser = colossalai.get_default_parser()
    parser.add_argument('--dist_backend', type=str, default='colossalai') # colossalai, torch_ddp, torch_fsdp
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--model', type=str, default='ofa')
    parser.add_argument('--use_zero', type=int, default=1) # 1: True 0: False
    parser.add_argument('--steps', type=int, default=20) # number of steps
    parser.add_argument('--bs', type=int, default=64) # batch size
    parser.add_argument('--img_size', type=int, default=64) # batch size
    parser.add_argument('--exp_name', type=str, default='') # 1: True 0: False
    parser.add_argument('--debug', type=int, default=0) # 1: True 0: False
    
    ### for colossalai only
    parser.add_argument('--shardstrategy', type=str, default='btss') # tss: TensorShardStrategy, btss: BucketTensorShardStrategy 
    
    ### for ofa only
    parser.add_argument('--searchspace', type=str, default='ofa_masks.pth') 
    
    args = parser.parse_args()
    model = args.model
    dist_backend = args.dist_backend
    use_zero = args.use_zero
    gpus = args.gpus
    debug = args.debug
    num_steps = args.steps
    exp_name = args.exp_name
    batch_size = args.bs
    img_size = args.img_size
    shardstrategy = args.shardstrategy
    if debug:
        set_trace()
    
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    root_dir = f'./logs/{model}/{dist_backend}_gpu{gpus}_{batch_size}x{img_size}x{img_size}_{exp_name}'
    if dist_backend == 'colossalai' and use_zero:
        root_dir += f'_zero_{shardstrategy}'
    root_dir += f'/{date_of_run}'
    # init bacekend
    if dist_backend == 'colossalai':
        if use_zero:
            args.config = {}
        else:
            args.config = {
                'torch_ddp': {
                    'find_unused_parameters': True,
                },
                'parallel': {
                    'data': args.gpus,
                    'pipeline': 1,
                }
            }
            args.config = {}
        colossalai.launch_from_torch(config=args.config)
        grank = gpc.get_global_rank()
        root_dir = f'{root_dir}/rank_{grank}'
        # init logger
        disable_existing_loggers()
        logger = get_dist_logger()
        logger.log_to_file(f'{root_dir}', mode='w')
    elif dist_backend in ['torch_ddp', 'torch_fsdp', 'torch_zero']:
        init_process_group(backend='nccl', init_method='env://')
        grank = torch.distributed.get_rank()
        root_dir = f'{root_dir}/rank_{grank}'
        logger = loguru.logger
        logger.add(f'{root_dir}/rank_{grank}.log', format="{time} {level} {message}", level="INFO", enqueue=True, colorize=True)
    logger.info(f'using backend: {dist_backend}')
    
    ### for ofa only
    searchspace_path = args.searchspace
    if args.model != 'ofa':
        searchspace = None
    elif searchspace_path:
        searchspace = torch.load(searchspace_path)
    else:
        searchspace = None

    # init GPU/CPU memory
    logger.info(get_mem_info())

    # build model
    if dist_backend == 'colossalai':
        if use_zero:
            shard_strategy = TensorShardStrategy() if args.shardstrategy == 'tss' else BucketTensorShardStrategy()
            with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy, shard_param=True) as ctx:
                model = get_model(args.model)
            numel = ctx.model_numel_tensor.item()
            logger.info(f'Model numel: {numel}')
            # Set tensor_placement_policy='cpu', which will offload params, grads and os
            model = ShardedModelV2(model, shard_strategy, tensor_placement_policy='cuda', reuse_fp16_shard=True)
        else:
            model = get_model(args.model).to(grank)
    elif dist_backend in ['torch_ddp', 'torch_zero']:
        model = get_model(args.model).to(grank)
        model = DDP(model, find_unused_parameters=True, device_ids=[grank])
    elif dist_backend == 'torch_fsdp':
        my_auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=20000
        )
        fpSixteen = MixedPrecision(
            param_dtype=torch.float16,
            # Gradient communication precision.
            reduce_dtype=torch.float16,
            # Buffer precision.
            buffer_dtype=torch.float16,
        )
        model = get_model(args.model).to(grank)
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            mixed_precision=fpSixteen,
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
            sync_module_states=True,
            device_id=grank,
        )
        # model = DDP(model, find_unused_parameters=True, device_ids=[grank], output_device=grank)
    logger.info(f"Bulding model: {args.model}")
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {params}")
    logger.info(get_mem_info(prefix='After init model, '))
    rm = RandomMutator(model)
    # rm.reset()

    # build optimizer
    if dist_backend == 'colossalai' and use_zero:
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
    elif dist_backend == 'torch_zero':
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(), optimizer_class=torch.optim.Adam, lr=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info(get_mem_info(prefix='After init optim, '))

    model.train()
    num_samples = 0.
    used_time = 0.
    headers = 'rank, batchIdx, fw_g, bw_g, update_g, fw_p, bw_p, update_p, batch_time, bacth_tp, fw_time, bw_time, update_time'
    headers = headers.split(', ')
    tab_info = PrettyTable(headers)
    size = int(args.img_size)
    x = torch.rand(batch_size, 3, size, size).to(grank)
    y = torch.rand(batch_size, 1000).to(grank)
    
    for n in range(num_steps):
        # we just use randomly generated data here
        if searchspace is not None:
            mask = searchspace[n]
            rm.sample_by_mask(mask)
            logger.info(f'rank: {grank}, mask: {mask}')
        else:
            rm.reset()
        
        optimizer.zero_grad(set_to_none=True)
        logger.info(get_mem_info(prefix=f'[{n+1}/{num_steps}] Pre-Forward '))
        batch_start = time()
        
        ### forward
        fw_start = time()
        outputs = model(x)
        fw_end = time()
        loss = (outputs-y).sum()
        fw_g, fw_c = get_gpu_mem(grank), get_peak_gpu_mem(grank)
        logger.info(get_mem_info(prefix=f'[{n+1}/{num_steps}] Post-Forward ', rank=grank))

        ### backward
        bw_start = time()
        if dist_backend == 'colossalai' and use_zero:
            optimizer.backward(loss)
        else:
            loss.backward()
        bw_end = time()
        bw_g, bw_c = get_gpu_mem(grank), get_peak_gpu_mem(grank)
        logger.info(get_mem_info(prefix=f'[{n+1}/{num_steps}] Post-Backward ', rank=grank))
        
        ### update
        update_start = time()
        optimizer.step()
        update_end = time()
        update_g, update_c = get_gpu_mem(grank), get_peak_gpu_mem(grank)
        logger.info(get_mem_info(prefix=f'[{n+1}/{num_steps}] Post-Step ', rank=grank))

        batch_end = time()
        
        if n <= 2:
            continue
        # time
        batch_time = batch_end - batch_start
        fw_time = fw_end - fw_start
        bw_time = bw_end - bw_start
        update_time = update_end - update_start
        
        used_time += batch_time
        num_samples += (batch_size * gpus)
        batch_tp = gpus * batch_size / (batch_time + 1e-12) # batch throughput
        
        data = [grank, n, fw_g, bw_g, update_g, fw_c, bw_c, update_c, batch_time, batch_tp, fw_time, bw_time, update_time]
        data = [f'{d:.4f}' if isinstance(d, float) else d for d in data ]
        tab_info.add_row(data)
    overall_tp = num_samples / (used_time + 1e-12)
    logger.info(f'rank: {grank}, overall throughput: {overall_tp:.4f} samples/s')
    csv_info = tab_info.get_csv_string()
    columns = csv_info.split('\r\n')[0].split(',')
    values =[x.split(',') for x in csv_info.split('\r\n')[1:-1]]
    pd_csv = pd.DataFrame(values, columns=columns).astype(float)
    mean_csv = pd_csv.groupby('rank').mean().values[0].tolist()
    std_csv = pd_csv.groupby('rank').std().values[0].tolist()
    mean_csv = ['mean'] + [f'{d:.4f}' if isinstance(d, float) else d for d in mean_csv]
    std_csv = ['std'] + [f'{d:.4f}' if isinstance(d, float) else d for d in std_csv]
    tab_info.add_row(mean_csv)
    tab_info.add_row(std_csv)
    logger.info(tab_info)
    csv_info = tab_info.get_csv_string()
    with open(f'{root_dir}/statistic.csv', 'w', newline='') as f:
        f.write(csv_info)
    with open(f'{root_dir}/overall.csv', 'w', newline='') as f:
        overall_info = []
        lines = csv_info.split('\r\n')[:-1]
        for idx, line in enumerate(lines):
            line = line.split(',')
            line.pop(1)
            line = ','.join(line)
            if idx in [0, len(lines) - 2, len(lines) - 1]:
                overall_info.append(line)
        line = ','.join(['TP', f"{overall_tp:.4f}"] + [''] * (len(overall_info[0].split(',')) - 2))
        overall_info.append(line)
        overall_info = '\r\n'.join(overall_info)
        f.write(overall_info)

if __name__ == '__main__':
    main()
