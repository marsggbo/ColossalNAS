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
# DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# FSDP
try:
    from torch.distributed.optim import ZeroRedundancyOptimizer
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
except ImportError:
    print(f"FSDP is not available for torch {torch.__version__}")

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

from dataloader import get_fake_dataloader, get_cifar10_dataloader, get_cifar10_dataset, FakeDataset
from models import get_model
from utils import get_peak_gpu_mem, get_gpu_mem, get_cpu_mem, print_mem_info, DeviceInfo, get_mem_info


logger = loguru.logger
@atexit.register
def exit_handler():
    global logger
    logger.info("exit with no error", ranks=[0])
    
def get_args():
    parser = colossalai.get_default_parser()
    parser.add_argument('--dist_backend', type=str, default='colossalai') # colossalai, torch_ddp, torch_fsdp
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--model', type=str, default='ofa')
    parser.add_argument('--nof', type=float, default=0.) # nvme offload fraction, a value between 0. and 1.
    parser.add_argument('--use_fp16', type=int, default=1) # 1: True 0: False
    parser.add_argument('--use_zero', type=int, default=1) # 1: True 0: False
    parser.add_argument('--use_pipeline', type=int, default=1) # 1: True 0: False
    parser.add_argument('--steps', type=int, default=100) # number of steps
    parser.add_argument('--img_size', type=int, default=224) # img size
    parser.add_argument('--batch_size', type=int, default=64) # batch size
    parser.add_argument('--exp_name', type=str, default='') # 1: True 0: False
    parser.add_argument('--placement', type=str, default='cuda') # 1: True 0: False
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

def main():
    global logger
    args = get_args()
    batch_size = args.batch_size
    model = args.model
    dist_backend = args.dist_backend
    use_pipeline = args.use_pipeline
    use_zero = args.use_zero
    use_fp16 = args.use_fp16
    assert not (use_zero and use_pipeline), 'use_zero and use_pipeline cannot be True at the same time'
    gpus = args.gpus
    debug = args.debug
    num_steps = args.steps
    placement = args.placement
    exp_name = args.exp_name
    img_size = args.img_size
    shardstrategy = args.shardstrategy
    nof = args.nof
    seed = args.seed
    set_seed(seed)    
    
    if debug:
        set_trace()
    
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    root_dir = f'./logs/{model}/{dist_backend}_gpu{gpus}_{batch_size}x{img_size}x{img_size}'
    if dist_backend == 'colossalai':
        if use_zero:
            root_dir += f'_zero_{shardstrategy}_nof{nof}'
        if use_pipeline:
            root_dir += f'_pipeline'
        if use_fp16:
            root_dir += f'_fp16'
    root_dir += f'_{exp_name}/{date_of_run}'

    # init bacekend
    if dist_backend == 'colossalai':
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
        # grank = gpc.get_global_rank()
        grank = int(os.environ['RANK'])
        lrank = int(os.environ['LOCAL_RANK'])
        root_dir = f'{root_dir}/rank_{grank}'
        # init logger
        disable_existing_loggers()
        logger = get_dist_logger()
        logger.log_to_file(f'{root_dir}', mode='w')
    elif dist_backend in ['torch_ddp', 'torch_fsdp', 'torch_zero']:
        init_process_group(backend='nccl', init_method='env://')
        grank = torch.distributed.get_rank()
        lrank = int(os.environ['LOCAL_RANK'])
        root_dir = f'{root_dir}/rank_{grank}'
        logger = loguru.logger
        logger.add(f'{root_dir}/rank_{grank}.log', format="{time} {level} {message}", level="INFO", enqueue=True, colorize=True)
    logger.info(f'rank[{grank}] using backend: {dist_backend}', ranks=[0])

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
    
    # init GPU/CPU memory
    logger.info(f"rank[{grank}] args: {args}", ranks=[0])
    logger.info(print_mem_info(rank=lrank), ranks=[0])

    # build model
    if dist_backend == 'colossalai':
        if use_zero:
            shard_strategy = TensorShardStrategy() if args.shardstrategy == 'tss' else BucketTensorShardStrategy()
            with ZeroInitContext(target_device=torch.device(lrank), shard_strategy=shard_strategy, shard_param=True) as ctx:
                model = get_model(args.model)
            numel = ctx.model_numel_tensor.item()
            wandb.config.update({'model_numel': numel})
            logger.info(f'rank[{grank}] Model numel: {numel}', ranks=[0])
            # Set tensor_placement_policy='cpu', which will offload params, grads and os
            model = ShardedModelV2(model, shard_strategy, tensor_placement_policy=placement, reuse_use_fp16_shard=True)       
            
            # from colossalai.utils.model.colo_init_context import ColoInitContext
            # from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
            # from colossalai.utils import get_current_device
            # world_size = torch.distributed.get_world_size()
            # default_dist_spec = ShardSpec([-1], [world_size]) # or `None``
            # shard_pg = ProcessGroup(tp_degree=world_size) # or `None`
            # with ColoInitContext(device=get_current_device(),
            #                     dtype=torch.half,
            #                     default_dist_spec=default_dist_spec,
            #                     default_pg=shard_pg):
            #     model = get_model(args.model)
            # tp_pg = ProcessGroup(tp_degree=1)
            # gemini_config = dict(strict_ddp_mode=True,
            #                      device=get_current_device(),
            #                      placement_policy='cpu', # 'nvme
            #                      pin_memory=True,
            #                      hidden_dim=1024,
            #                      search_range_mb=128)
            # model = zero_model_wrapper(model, 3, gemini_config)
        elif use_pipeline:
            # Todo: add pipeline
            pipelinable = PipelinableContext()
            with pipelinable:
                model = get_model(args.model)
            # exec_seq = ['conv1', 'conv2', 'gavg', 
            #             (lambda x: torch.flatten(x, 1), "behind"), 'fc']
            # resnet
            # exec_seq = [
            #     'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool',
            #     (lambda x: torch.flatten(x, 1), "behind"), 'fc'
            # ]
            # # pipelinable.to_layer_list(model.exec_seq)
            # pipelinable.to_layer_list(exec_seq)
            pipelinable.to_layer_list()
            # pipelinable.policy = "uniform"
            pipelinable.policy = "balanced"
            model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
            # model = pipelinable.partition(1, gpc.pipeline_parallel_size, torch.distributed.get_rank())
        else:
            model = get_model(args.model).to(lrank)
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
        model = get_model(args.model).to(lrank)
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

    logger.info(f"rank[{grank}] Bulding model: {args.model}", ranks=[0])
    params = sum(p.numel() for p in model.parameters())
    wandb.config.update({'model_params': params})
    logger.info(f"rank[{grank}] Model params: {params}", ranks=[0])
    logger.info(print_mem_info(prefix='After init model, '), ranks=[0])
    rm = RandomMutator(model)
    # rm.reset()

    # build optimizer
    if dist_backend == 'colossalai' and use_zero:
        # optim_config = dict(gpu_margin_mem_ratio=0.)
        # optimizer = HybridAdam(model.parameters(), lr=1e-3, nvme_offload_fraction=nof, nvme_offload_dir='./')
        # optimizer = zero_optim_wrapper(model, optimizer, optim_config=optim_config)
        optimizer = HybridAdam(model.parameters(), lr=1e-3, nvme_offload_fraction=nof, nvme_offload_dir='./nvme')
        optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
    elif dist_backend == 'torch_zero':
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(), optimizer_class=torch.optim.Adam, lr=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info(print_mem_info(prefix='After init optim, '), ranks=[0])

    model.train()
    num_samples = 0.
    used_time = 0.
    if not use_pipeline:
        headers = 'Idx, Forward CPU (MB), Backward CPU (MB), Update CPU (MB), PeakCPU (MB), Forward GPU (MB), Backward GPU (MB), Update GPU (MB), PeakGPU (MB), Batch Time, Batch Throughput, Forward Time, Backward Time, Update Time'
    else:
        headers = 'Idx, For&Back-ward CPU (MB), Update-CPU (MB), PeakCPU (MB), For&Back-ward GPU (MB), Update-GPU (MB), PeakGPU (MB), Batch Time, Batch Throughput, For&Back-ward Time, Update Time'
    headers = headers.split(', ')
    tab_info = PrettyTable(headers)
    size = int(args.img_size)
    
    # loader = get_fake_dataloader(3000000000, size, batch_size)
    # train_loader, test_loader = get_cifar10_dataloader(batch_size)
    # loader = train_loader
    # train_dataset, test_dataset = get_cifar10_dataset()
    # dataset = torch.utils.data.ConcatDataset([train_dataset]*100)
    # loader = get_dataloader(dataset, add_sampler=True, batch_size=args.batch_size, num_workers=4,
    #         pin_memory=True, shuffle=False)
    # data_iter = iter(loader)
    data_iter = infinite_fake_data_generator(args.batch_size, size, 10)

    criterion = torch.nn.CrossEntropyLoss()
    if dist_backend=='colossalai':
        engine, _, _, _ = colossalai.initialize(
            model, optimizer, criterion, None,
        )
    else:
        engine = None

    for n in range(10000000):
        if n >= num_steps:
            break
        rm.reset()
        if hasattr(model, 'arch'):
            logger.info(f"{n}: {model.arch}", ranks=[0])
        
        engine.zero_grad() if engine is not None else optimizer.zero_grad(set_to_none=True)
        logger.info(print_mem_info(prefix=f'[{n+1}/{num_steps}] Pre-Forward ', rank=lrank), ranks=[0])

        gpu_mem_hist = []
        cpu_mem_hist = []
        batch_start = time()
        if not use_pipeline:
            if not use_zero:
                print(f'rank[{grank}] {n}: {model.arch}')
            (x, y) = data_iter.__next__()
            x = x.to(lrank)
            y = y.to(lrank)

            ### forward
            fw_start = time()
            outputs = model(x) if engine is None else engine(x)
            fw_end = time()
            loss = criterion(outputs, y) if engine is None else engine.criterion(outputs, y)
            fw_c, fw_g, fw_gp = get_mem_info(lrank)
            wandb_log = {'step':n, 'fw_cpu_mem': fw_c, 'fw_gpu_mem': fw_g, 'fw_gpu_peak_mem': fw_gp}
            logger.info(print_mem_info(prefix=f'[{n+1}/{num_steps}] Post-Forward ', rank=lrank), ranks=[0])

            ### backward
            bw_start = time()
            if dist_backend == 'colossalai':
                if use_zero:
                    optimizer.backward(loss)
                elif engine is not None:
                    engine.backward(loss)
                else:
                    loss.backward()
            else:
                loss.backward()
            bw_end = time()
            bw_c, bw_g, bw_gp = get_mem_info(lrank)
            wandb_log.update({'bw_cpu_mem': bw_c, 'bw_gpu_mem': bw_g, 'bw_gpu_peak_mem': bw_gp})
            logger.info(print_mem_info(prefix=f'[{n+1}/{num_steps}] Post-Backward ', rank=lrank), ranks=[0])
            fw_time = fw_end - fw_start
            bw_time = bw_end - bw_start
            wandb_log.update({'fw_time': fw_time, 'bw_time': bw_time})
            gpu_mem_hist = [fw_gp, bw_gp]
            cpu_mem_hist = [fw_c, bw_c]
        else:
            fb_start = time()
            # (x, y) = data_iter.__next__()
            # logger.info(f"rank[{grank}] batch size: {x.shape[0]}, {y.view(-1)[:20]}")
            # torch.cuda.synchronize()
            engine.execute_schedule(data_iter, return_output_label=False)
            fb_c, fb_g, fb_gp = get_mem_info(lrank)
            fb_end = time()
            gpu_mem_hist = [fb_gp]
            cpu_mem_hist = [fb_c]
            logger.info(print_mem_info(prefix=f'[{n+1}/{num_steps}] Pipe-Forward-Backward ', rank=lrank), ranks=[0])
            fb_time = fb_end - fb_start
            fw_time = fb_time / 3
            bw_time = fb_time - fw_time
            wandb_log = {'step':n, 'fb_c_mem': fb_c, 'fb_gpu_mem': fb_g, 'fb_gpu_peak_mem': fb_gp,
                         'fw_time': fw_time, 'bw_time': bw_time, 'fb_time': fb_time}
        ### update
        update_start = time()
        optimizer.step() if engine is None else engine.step()
        update_end = time()
        update_time = update_end - update_start
        update_c, update_g, update_gp = get_mem_info(lrank)
        gpu_mem_hist.append(update_gp)
        cpu_mem_hist.append(update_c)
        logger.info(print_mem_info(prefix=f'[{n+1}/{num_steps}] Post-Step ', rank=lrank), ranks=[0])
        wandb_log.update({'update_cpu_mem': update_c, 'update_gpu_mem': update_g, 'update_gpu_peak_mem': update_gp, 'update_time': update_time})

        batch_end = time()
        batch_time = batch_end - batch_start
        wandb_log.update({'batch_time': batch_time})
        cp = max(cpu_mem_hist)
        gp = max(gpu_mem_hist)
        
        if n > 5:
            # time
            used_time += batch_time
            if not use_pipeline:
                batch_samples = batch_size * gpus
            else:
                batch_samples = batch_size
            num_samples += batch_samples
            batch_tp = batch_samples / (batch_time + 1e-12) # batch throughput
            
            if not use_pipeline:
                data = [n, fw_c, bw_c, update_c, cp, fw_g, bw_g, update_g, gp, batch_time, batch_tp, fw_time, bw_time, update_time]
            else:
                data = [n, fb_c, update_c, cp, fb_g, update_g, gp, batch_time, batch_tp, fb_time, update_time]
            data = [f'{d:.4f}' if isinstance(d, float) else d for d in data ]
            tab_info.add_row(data)
            wandb_log.update({'batch_throughput': batch_tp})
        wandb.log(wandb_log)
    overall_tp = num_samples / (used_time + 1e-12)
    wandb.log({'overall_throughput': overall_tp})
    logger.info(f'rank[{grank}], overall throughput: {overall_tp:.4f} samples/s', ranks=[0])
    parse_tab_info(tab_info, overall_tp, logger, root_dir)


def parse_tab_info(tab_info, overall_tp, logger, root_dir):
    csv_info = tab_info.get_csv_string()
    columns = csv_info.split('\r\n')[0].split(',')
    values =[x.split(',') for x in csv_info.split('\r\n')[1:-1]]
    pd_csv = pd.DataFrame(values, columns=columns).astype(float)
    max_csv = pd_csv.max().tolist()[1:]
    mean_csv = pd_csv.mean().tolist()[1:]
    std_csv = pd_csv.std().tolist()[1:]
    max_csv =  ['max'] + [f'{d:.4f}' if isinstance(d, float) else d for d in max_csv]
    mean_csv = ['mean'] + [f'{d:.4f}' if isinstance(d, float) else d for d in mean_csv]
    std_csv = ['std'] + [f'{d:.4f}' if isinstance(d, float) else d for d in std_csv]
    tab_info.add_row(max_csv)
    tab_info.add_row(mean_csv)
    tab_info.add_row(std_csv)
    logger.info(tab_info, ranks=[0])
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
            if idx in [0, len(lines) - 3, len(lines) - 2, len(lines) - 1]:
                overall_info.append(line)
        line = ','.join(['TP', f"{overall_tp:.4f}"] + [''] * (len(overall_info[0].split(',')) - 2))
        overall_info.append(line)
        overall_info = '\r\n'.join(overall_info)
        f.write(overall_info)


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        logger.info(traceback.format_exc(), ranks=[0])
