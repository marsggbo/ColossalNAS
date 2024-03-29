import os
import json
import loguru
import traceback
import atexit
import numpy as np
import pandas as pd
import wandb
from time import time
from functools import partial
from prettytable import PrettyTable
from ipdb import set_trace
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn

# Colossalai
import colossalai
from colossalai.utils import get_dataloader
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.core import global_context as gpc
## Colossalai AMP (fp16)
from colossalai.amp import AMP_TYPE

from hyperbox.mutator import RandomMutator

from dataloader import get_fake_dataloader, get_cifar10_dataloader, get_cifar10_dataset, FakeDataset, RepeatingLoader
from models import get_model
from utils import print_mem_info, get_mem_info, get_memory_usage, ExpLog

logger = loguru.logger


def train_base(args, logger, dataloader):
    exp_log = ExpLog(args)
    grank = int(os.environ['RANK'])
    lrank = int(os.environ['LOCAL_RANK'])
    world_size = torch.distributed.get_world_size()

    model = get_model(args.model, args).to(lrank)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"rank[{grank}] Model params: {params}", ranks=[0])
    model.train()
    rm = RandomMutator(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    engine, _, _, _ = colossalai.initialize(model, optimizer, criterion, None)

    for step, (x, y) in enumerate(dataloader):
        if step >= args.steps:
            break
        rm.reset()
        engine.zero_grad()
        if hasattr(model, 'arch') and step <= 5:
            logger.info(f"step {step}: {model.arch}", ranks=[0])

        x = x.to(lrank)
        y = y.to(lrank)
        BS = x.shape[0]

        torch.cuda.synchronize()
        start = time()
        outputs = engine(x)
        loss = engine.criterion(outputs, y)
        engine.backward(loss)
        engine.step()
        torch.cuda.synchronize()
        end = time()

        # calculate throughput and memory usage
        batch_time = end - start
        throughput = BS* world_size / (end - start)
        calc = f"{BS} (BS) * {world_size}($gpus) / {end - start:.2f}(time)"
        ag, mag, rg, mrg, cm = get_memory_usage(lrank)
        if step>5:
            exp_log.add(batch_time, throughput)
        if lrank == 0:
            logger.info(f'[rank{grank}] step{step}: throughput is {calc}={throughput:.2f} img/sec loss: {loss}')
            logger.info(f'[rank{grank}] step{step}: CPU Mem {cm:.2f} | GPU Mem-A({ag:.2f}) Max-Mem-A({mag:.2f}) Mem-R({rg:.2f}) Max-Mem-R({mrg:.2f})')
    if lrank == 0:
        exp_log.save_as_csv()
        exp_log.stats_and_save()

def train_pipe(args, logger, dataloader):
    ## Colossalai Pipeline
    from colossalai.context import ParallelMode
    from colossalai.pipeline.pipelinable import PipelinableContext
    exp_log = ExpLog(args)
    grank = int(os.environ['RANK'])
    lrank = int(os.environ['LOCAL_RANK'])
    world_size = torch.distributed.get_world_size()

    pipelinable = PipelinableContext()
    with pipelinable:
        model = get_model(args.model, args)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"rank[{grank}] Model params: {params}", ranks=[0])
    pipelinable.to_layer_list()
    # pipelinable.policy = "uniform"
    pipelinable.policy = "balanced"
    model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
    model.train()
    rm = RandomMutator(model)

    data_iter = iter(dataloader)
    (x, y) = data_iter.__next__()
    BS = x.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    engine, _, _, _ = colossalai.initialize(model, optimizer, criterion, None)
    
    for step in range(100000):
        if step >= args.steps:
            break
        rm.reset()
        engine.zero_grad()
        if hasattr(model, 'arch') and step <= 5:
            logger.info(f"step {step}: {model.arch}", ranks=[0])

        torch.cuda.synchronize()
        start = time()
        engine.execute_schedule(data_iter, return_output_label=False)
        torch.cuda.synchronize()
        end = time()

        batch_time = end - start
        throughput = BS / (end - start)
        calc = f"{BS} (BS) / {end - start:.2f}(time)"
        ag, mag, rg, mrg, cm = get_memory_usage(lrank)
        if step>5:
            exp_log.add(batch_time, throughput)
        torch.cuda.synchronize()
        if lrank == 0:
            logger.info(f'[rank{lrank}] step{step}: throughput is {calc}={throughput:.2f} img/sec')
            logger.info(f'[rank{lrank}] step{step}: CPU Mem {cm:.2f} | GPU Mem-A({ag:.2f}) Max-Mem-A{mag:.2f} Mem-R({rg:.2f}) Max-Mem-R({mrg:.2f})')
    if lrank == 0:
        exp_log.save_as_csv()
        exp_log.stats_and_save()

def train_zero(args, logger, dataloader):
    # colossalai 0.2.8
    from colossalai.zero.legacy.init_ctx import ZeroInitContext
    from colossalai.zero.legacy.shard_utils import (TensorShardStrategy,
                                            BucketTensorShardStrategy)
    from colossalai.zero.legacy.sharded_model import ShardedModelV2
    from colossalai.zero.legacy.sharded_optim import ShardedOptimizerV2
    exp_log = ExpLog(args)
    grank = int(os.environ['RANK'])
    lrank = int(os.environ['LOCAL_RANK'])
    world_size = torch.distributed.get_world_size()

    # Init model
    shard_strategy = TensorShardStrategy() if args.shardstrategy == 'tss' else BucketTensorShardStrategy()
    with ZeroInitContext(target_device=torch.device(lrank), shard_strategy=shard_strategy, shard_param=True) as ctx:
        model = get_model(args.model, args)
    numel = ctx.model_numel_tensor.item()
    wandb.config.update({'model_numel': numel})
    logger.info(f'rank[{grank}] Model numel: {numel}', ranks=[0])
    # Set tensor_placement_policy='cpu', which will offload params, grads and os
    model = ShardedModelV2(model, shard_strategy, tensor_placement_policy=args.placement, reuse_use_fp16_shard=True)
    model.train()
    rm = RandomMutator(model)

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"rank[{grank}] Model params: {params}", ranks=[0])

    optimizer = HybridAdam(model.parameters(), lr=1e-3, nvme_offload_fraction=args.nof, nvme_offload_dir='./nvme')
    optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
    criterion = torch.nn.CrossEntropyLoss()

    engine, _, _, _ = colossalai.initialize(model, optimizer, criterion, None)
    for step, (x, y) in enumerate(dataloader):
        if step >= args.steps:
            break
        rm.reset()
        engine.zero_grad()
        if hasattr(model.module, 'arch') and step <= 5:
            logger.info(f"step {step}: {model.module.arch}", ranks=[0])

        x = x.to(lrank)
        y = y.to(lrank)
        BS = x.shape[0]

        torch.cuda.synchronize()
        start = time()
        outputs = engine(x)
        loss = engine.criterion(outputs, y)
        optimizer.backward(loss)
        engine.step()
        torch.cuda.synchronize()
        end = time()

        # calculate throughput and memory usage
        batch_time = end - start
        throughput = BS* world_size / (end - start)
        calc = f"{BS} (BS) * {world_size}($gpus) / {end - start:.2f}(time)"
        ag, mag, rg, mrg, cm = get_memory_usage(lrank)
        if step>5:
            exp_log.add(batch_time, throughput)
        if lrank == 0:
            logger.info(f'[rank{grank}] step{step}: throughput is {calc}={throughput:.2f} img/sec loss: {loss}')
            logger.info(f'[rank{grank}] step{step}: CPU Mem {cm:.2f} | GPU Mem-A({ag:.2f}) Max-Mem-A({mag:.2f}) Mem-R({rg:.2f}) Max-Mem-R({mrg:.2f})')
    if lrank == 0:
        exp_log.save_as_csv()
        exp_log.stats_and_save()

def train_zero_gemini(args, logger, dataloader):
    from colossalai.utils import get_current_device
    from colossalai.zero import ColoInitContext, zero_model_wrapper, zero_optim_wrapper
    from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec

    exp_log = ExpLog(args)
    grank = int(os.environ['RANK'])
    lrank = int(os.environ['LOCAL_RANK'])
    world_size = torch.distributed.get_world_size()        

    # Init model
    rm = None
    default_dist_spec = ShardSpec([-1], [world_size]) if args.tp_degree>1 else None
    shard_pg = ProcessGroup(tp_degree=world_size) if args.tp_degree>1 else None
    with ColoInitContext(device=get_current_device(),
                            dtype=torch.half,
                            default_dist_spec=default_dist_spec,
                            default_pg=shard_pg):
        model = get_model(args.model, args)
        numel = sum(p.numel() for p in model.parameters())
        wandb.config.update({'model_numel': numel})
        logger.info(f'rank[{grank}] Model numel: {numel}', ranks=[0])
        rm = RandomMutator(model)

    # Tensor Parallelism (TP)
    tp_pg = ProcessGroup(tp_degree=args.tp_degree)
    if args.tp_degree > 1:
        # Todo: add support for TP with gemini
        tensor_parallelize(model, tp_pg)

    # asign gemini running configurations
    zero_stage = args.zero_stage
    gemini_config = None
    gemini_config = dict(
        strict_ddp_mode=args.tp_degree == 1,
        device=get_current_device(),
        placement_policy=args.placement,
        pin_memory=True,
        hidden_dim=4096,
        search_range_mb=32
    )
    model = zero_model_wrapper(model, zero_stage, gemini_config)
    if zero_stage == 3:
        optim_config = dict(gpu_margin_mem_ratio=0.)
    else:
        cpu_offload = args.placement == 'cpu'
        optim_config = dict(
            reduce_bucket_size=32 * 1024 * 1024,
            overlap_communication=True,
            cpu_offload=cpu_offload,
        )
    optimizer = HybridAdam(model.parameters(), lr=1e-3, nvme_offload_fraction=args.nof, nvme_offload_dir='./nvme')
    optimizer = zero_optim_wrapper(model, optimizer, optim_config=optim_config)

    criterion = torch.nn.CrossEntropyLoss()
    engine, _, _, _ = colossalai.initialize(model, optimizer, criterion, None)
    for step, (x, y) in enumerate(dataloader):
        if step >= args.steps:
            break
        rm.reset()
        engine.zero_grad()
        if step <= 5:
            try:
                if hasattr(model, 'module') and hasattr(model.module, 'arch'):
                    logger.info(f"step {step}: {model.module.arch}", ranks=[0])
                else:
                    logger.info(f"step {step}: {model.arch}", ranks=[0])
            except:
                pass

        x = x.to(lrank).half()
        y = y.to(lrank)
        BS = x.shape[0]

        torch.cuda.synchronize()
        start = time()
        outputs = engine(x)
        loss = engine.criterion(outputs, y)
        optimizer.backward(loss)
        engine.step()
        torch.cuda.synchronize()
        end = time()

        # calculate throughput and memory usage
        batch_time = end - start
        throughput = BS* world_size / (end - start)
        calc = f"{BS} (BS) * {world_size}($gpus) / {end - start:.2f}(time)"
        ag, mag, rg, mrg, cm = get_memory_usage(lrank)
        if step>5:
            exp_log.add(batch_time, throughput)
        if lrank == 0:
            logger.info(f'[rank{grank}] step{step}: throughput is {calc}={throughput:.2f} img/sec loss: {loss}')
            logger.info(f'[rank{grank}] step{step}: CPU Mem {cm:.2f} | GPU Mem-A({ag:.2f}) Max-Mem-A({mag:.2f}) Mem-R({rg:.2f}) Max-Mem-R({mrg:.2f})')
    if lrank == 0:
        exp_log.save_as_csv()
        exp_log.stats_and_save()

def get_args():
    parser = colossalai.get_default_parser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--model', type=str, default='ofa')
    parser.add_argument('--use_fp16', type=int, default=1) # 1: True 0: False
    parser.add_argument('--use_pipeline', type=int, default=1) # 1: True 0: False
    parser.add_argument('--use_zero', type=int, default=1) # 1: True 0: False
    parser.add_argument('--zero_stage', type=int, default=2, choices=[1,2,3]) # [1,2,3]
    parser.add_argument('--nof', type=float, default=0.) # nvme offload fraction, a value between 0. and 1.
    parser.add_argument('--placement', type=str, default='auto') # 'cpu' or 'nvme'
    parser.add_argument('--use_ac', type=int, default=0, help='use activation checkpointing') # 1: True 0: False
    parser.add_argument('--steps', type=int, default=100) # number of steps
    parser.add_argument('--img_size', type=int, default=224) # img size
    parser.add_argument('--batch_size', type=int, default=64) # batch size
    parser.add_argument('--exp_name', type=str, default='') # 1: True 0: False
    parser.add_argument('--use_gemini', type=int, default=0, help="use gemini") # 1: True 0: False
    parser.add_argument('--tp_degree', type=int, default=1, help="tensor parallel degree") # >=1
    parser.add_argument('--debug', type=int, default=0) # 1: True 0: False
    parser.add_argument('--seed', type=int, default=666)

    ### for colossalai only
    parser.add_argument('--shardstrategy', type=str, default='btss') # tss: TensorShardStrategy, btss: BucketTensorShardStrategy 
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init(args):
    batch_size = args.batch_size
    model = args.model
    use_pipeline = args.use_pipeline
    use_zero = args.use_zero
    zero_stage = args.zero_stage
    placement = args.placement
    shardstrategy = args.shardstrategy
    nof = args.nof
    use_ac = args.use_ac
    use_fp16 = args.use_fp16
    assert not (use_zero and use_pipeline), 'use_zero and use_pipeline cannot be True at the same time'
    gpus = args.gpus
    debug = args.debug
    steps = args.steps
    exp_name = args.exp_name
    img_size = args.img_size
    seed = args.seed
    set_seed(seed)

    # set up log dir
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    root_dir = f'./logs/{model}/gpu{gpus}_{batch_size}x{img_size}x{img_size}'
    if use_zero:
        root_dir += f'_zero{zero_stage}({placement})_{shardstrategy}_nof{nof}'
    if use_pipeline:
        root_dir += f'_pipeline'
    if use_fp16:
        root_dir += f'_fp16'
    if use_ac:
        root_dir += f'_ac'
    if exp_name not in ['', 'null']:
        root_dir += f'_{exp_name}'
    root_dir += f'/{date_of_run}'

    # set up colossal config
    args.config = {}
    if use_zero:
        args.config = {
            'torch_ddp': {
                'find_unused_parameters': True,
            },
            'parallel': {
                'data': args.gpus,
            }
        }
    if use_pipeline:
        args.config.update({
            'torch_ddp': {
                'find_unused_parameters': True,
            },
            'parallel': {
                'pipeline': args.gpus,
            },
            'NUM_MICRO_BATCHES': args.gpus,
        })
    else:
        args.config.update({
            'torch_ddp': {
                'find_unused_parameters': True,
            },
            'parallel': {
                'data': args.gpus,
            }
        })
        # args.config = {}
    if use_fp16:
        args.config['fp16'] = dict(
            mode=AMP_TYPE.NAIVE,
        )
    colossalai.launch_from_torch(config=args.config)

    # set up logger
    grank = int(os.environ['RANK'])
    lrank = int(os.environ['LOCAL_RANK'])
    root_dir = f'{root_dir}/rank_{grank}'
    disable_existing_loggers()
    logger = get_dist_logger()
    logger.log_to_file(f'{root_dir}', mode='w')

    # save args
    args.root_dir = root_dir
    with open(os.path.join(root_dir, 'args.json'), 'w') as f:
        if 'fp16' in args.config:
            args.config['fp16']['mode'] = str(args.config['fp16']['mode'])
        json.dump(args.__dict__, f, indent=4)
    name = '/'.join(root_dir.split('/')[2:4])
    wandb.init(
        project="ColossalNAS",
        config=args.__dict__,
        name=name,
        group=name+f'-{seed}',
        entity='marsggbo',
        mode='offline'
    )
    logger.info(f"args: {args.__dict__}")
    return args, logger


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        from ipdb import set_trace
        set_trace()

    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    args, logger = init(args)
    dataloader = RepeatingLoader(get_cifar10_dataloader(args.batch_size, args.img_size)[0])
    try:
        if args.use_pipeline != 0:
            train_pipe(args, logger, dataloader)
        elif args.use_zero != 0:
            if args.use_gemini == 0:
                train_zero(args, logger, dataloader)
            else:
                train_zero_gemini(args, logger, dataloader)
        else:
            train_base(args, logger, dataloader)
    except BaseException as e:
        logger.info(traceback.format_exc())
