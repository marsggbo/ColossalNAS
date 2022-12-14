
import torch
import torch.nn as nn

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.core import global_context as gpc
from colossalai.zero.shard_utils import (TensorShardStrategy,
                                         BucketTensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from time import time
from functools import partial

from prettytable import PrettyTable
from ipdb import set_trace
from hyperbox.mutator import RandomMutator

from models import get_model
from utils import get_cpu_mem, get_gpu_mem, get_mem_info, DeviceInfo


def main():
    BS = 64
    NUM_STEPS = 10
    parser = colossalai.get_default_parser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--model', type=str, default='ofa')
    parser.add_argument('--shardstrategy', type=str, default='tss') # tss: TensorShardStrategy, btss: BucketTensorShardStrategy 
    parser.add_argument('--useZero', type=int, default=1) # 1: True 0: False
    parser.add_argument('--searchspace', type=str, default='ofa_masks.pth') # 1: True 0: False
    parser.add_argument('--debug', type=int, default=0) # 1: True 0: False
    args = parser.parse_args()
    if args.config is None:
        args.config = {}
    debug = args.debug
    if debug:
        set_trace()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)
    grank = gpc.get_global_rank()
    use_zero = args.useZero
    gpus = args.gpus
    
    ### for ofa only
    searchspace_path = args.searchspace
    if args.model != 'ofa':
        searchspace = None
    elif searchspace_path:
        searchspace = torch.load(searchspace_path)
    else:
        searchspace = None
        
    # use_zero = False
    if use_zero:
        log_flag = 'zero'
    else:
        log_flag = 'ddp'
    
    # init logger
    disable_existing_loggers()
    logger = get_dist_logger()
    logger.log_to_file(f'./logs/exp_{log_flag}_gpu{gpus}/log_rank{grank}', mode='w')
    logger.info("initialized distributed environment")

    # init GPU/CPU memory
    logger.info(get_mem_info())

    # build NAS model
    if use_zero:
        shard_strategy = TensorShardStrategy() if args.shardstrategy == 'tss' else BucketTensorShardStrategy()
        with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy, shard_param=True) as ctx:
            model = get_model(args.model)
        numel = ctx.model_numel_tensor.item()
        logger.info(f'Model numel: {numel}')
        # Set tensor_placement_policy='cpu', which will offload params, grads and os
        model = ShardedModelV2(model, shard_strategy, tensor_placement_policy='cuda', reuse_fp16_shard=True)
    else:
        model = get_model(args.model).to(torch.cuda.current_device())
    logger.info(get_mem_info(prefix='After init model, '))
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
    headers = 'rank, batchIdx, batchTime, fw_g, bw_g, update_g, fw_c, bw_c, update_c, batch_time, fw_time, bw_time, update_time'
    headers = headers.split(', ')
    tab_info = PrettyTable(headers)
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        x = torch.rand(BS, 3, 64, 64).to(torch.cuda.current_device())
        y = torch.rand(BS, 1000).to(torch.cuda.current_device())
        if searchspace is not None:
            mask = searchspace[n]
            rm.sample_by_mask(mask)
            logger.info(f'rank: {grank}, mask: {mask}')
        else:
            rm.reset()
        optimizer.zero_grad()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Pre-Forward '))
        batch_start = time()
        
        ### forward
        fw_start = time()
        outputs = model(x)
        fw_end = time()
        loss = (outputs-y).sum()
        fw_g, fw_c = get_gpu_mem(), get_cpu_mem()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Post-Forward '))
        
        ### backward
        bw_start = time()
        if use_zero:
            optimizer.backward(loss)
        else:
            loss.backward()
        bw_end = time()
        bw_g, bw_c = get_gpu_mem(), get_cpu_mem()
        # DeviceInfo(model)
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Post-Backward '))
        
        ### update
        update_start = time()
        optimizer.step()
        update_end = time()
        update_g, update_c = get_gpu_mem(), get_cpu_mem()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Post-Step '))

        batch_end = time()
        batch_time = batch_end - batch_start
        fw_time = fw_end - fw_start
        bw_time = bw_end - bw_start
        update_time = update_end - update_start
        # infos += f"{grank:6}, {n:9}, {batch_time:10,.4f}, {fw_g:10,.4f}, {bw_g:10,.4f}, {update_g:10,.4f}, {fw_c:10,.4f}, {bw_c:10,.4f}, {update_c:10,.4f}, {batch_time:10,.4f}, {fw_time:10,.4f}, {bw_time:10,.4f}, {update_time:10,.4f}\n"
        data = [grank, n, batch_time, fw_g, bw_g, update_g, fw_c, bw_c, update_c, batch_time, fw_time, bw_time, update_time]
        data = [f'{d:.4f}' if isinstance(d, float) else d for d in data ]
        tab_info.add_row(data)
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {batch_time:.3f}s')
    logger.info(tab_info)
    with open(f'exp_{log_flag}_gpu{gpus}/log_rank{grank}/statistic.csv', 'w', newline='') as f:
        f.write(tab_info.get_csv_string())

if __name__ == '__main__':
    main()
