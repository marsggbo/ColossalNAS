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
## Colossalai Zero
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.core import global_context as gpc
from colossalai.zero.shard_utils import (TensorShardStrategy,
                                         BucketTensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
## Colossalai Pipeline
from colossalai.context import ParallelMode
from colossalai.pipeline.pipelinable import PipelinableContext
## Colossalai AMP (fp16)
from colossalai.amp import AMP_TYPE

from hyperbox.mutator import RandomMutator

from dataloader import get_fake_dataloader, get_cifar10_dataloader, get_cifar10_dataset, FakeDataset, RepeatingLoader
from models import get_model
from utils import print_mem_info, get_mem_info, get_memory_usage, ExpLog


logger = loguru.logger
@atexit.register
def exit_handler():
    global logger
    logger.info("exit with no error", ranks=[0])

def get_args():
    parser = colossalai.get_default_parser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--model', type=str, default='ofa')
    parser.add_argument('--use_fp16', type=int, default=1) # 1: True 0: False
    parser.add_argument('--use_pipeline', type=int, default=1) # 1: True 0: False
    parser.add_argument('--use_zero', type=int, default=1) # 1: True 0: False
    parser.add_argument('--nof', type=float, default=0.) # nvme offload fraction, a value between 0. and 1.
    parser.add_argument('--placement', type=str, default='cpu') # 'cpu' or 'nvme'
    parser.add_argument('--use_ac', type=int, default=0, help='use activation checkpointing') # 1: True 0: False
    parser.add_argument('--steps', type=int, default=100) # number of steps
    parser.add_argument('--img_size', type=int, default=224) # img size
    parser.add_argument('--batch_size', type=int, default=64) # batch size
    parser.add_argument('--exp_name', type=str, default='') # 1: True 0: False
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

def infinite_fake_data_generator(batch_size, img_size, num_classes=10):
    while True:
        yield (torch.rand(batch_size, 3, img_size, img_size), torch.randint(low=0, high=num_classes, size=(batch_size,)))

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

def train_pipeline(args, logger, dataloader):
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
        start = time.time()
        engine.execute_schedule(data_iter, return_output_label=False)
        torch.cuda.synchronize()
        end = time.time()

        batch_time = end - start
        throughput = BS / (end - start)
        calc = f"{BS} (BS) / {end - start:.2f}(time)"
        ag, mag, rg, mrg, cm = get_memory_usage(lrank)
        if step>5:
            exp_log.add(batch_time, throughput)
        torch.cuda.synchronize()
        if lrank == 0:
            logger.info(f'[rank{lrank}] step{step}: throughput is {calc}={throughput:.2f} img/sec loss: {loss}')
            logger.info(f'[rank{lrank}] step{step}: CPU Mem {cm:.2f} | GPU Mem-A({ag:.2f}) Max-Mem-A{mag:.2f} Mem-R({rg:.2f}) Max-Mem-R({mrg:.2f})')
    if lrank == 0:
        exp_log.save_as_csv()
        exp_log.stats_and_save()

def train_zero(args, logger, dataloader):
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

def init(args):
    batch_size = args.batch_size
    model = args.model
    use_pipeline = args.use_pipeline
    use_zero = args.use_zero
    use_fp16 = args.use_fp16
    assert not (use_zero and use_pipeline), 'use_zero and use_pipeline cannot be True at the same time'
    gpus = args.gpus
    debug = args.debug
    steps = args.steps
    placement = args.placement
    exp_name = args.exp_name
    img_size = args.img_size
    shardstrategy = args.shardstrategy
    nof = args.nof
    use_ac = args.use_ac
    seed = args.seed
    set_seed(seed)

    # set up log dir
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    root_dir = f'./logs/{model}/gpu{gpus}_{batch_size}x{img_size}x{img_size}'
    if use_zero:
        root_dir += f'_zero_{shardstrategy}_nof{nof}'
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
        args.config = {}
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
            train_zero(args, logger, dataloader)
        else:
            train_base(args, logger, dataloader)
    except BaseException as e:
        logger.info(traceback.format_exc())
