import os

import colossalai

import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.utils import is_using_pp, get_dataloader
from colossalai.pipeline.pipelinable import PipelinableContext
from titans.model.vit.vit import _create_vit_model
from tqdm import tqdm

from titans.dataloader.cifar10 import build_cifar

import psutil
from ipdb import set_trace
from torchvision.models import resnet18

# from colossalai.builder import build_pipeline_model
import colossalai.nn as col_nn
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader

from hyperbox.mutables.spaces import OperationSpace
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.mutator import RandomMutator


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


class NASModel(torch.nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.conv1 = OperationSpace(
            [torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
             torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
             torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            ], key='conv1'
        )
        self.conv2 = torch.nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(96, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.gavg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = OperationSpace(
            [torch.nn.Linear(512,10),torch.nn.Linear(512,10),torch.nn.Linear(512,10)], key='fc')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.gavg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    grank =gpc.get_global_rank()

    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    use_pipeline = is_using_pp()

    # create model
    model_kwargs = dict(img_size=gpc.config.IMG_SIZE,
                        patch_size=gpc.config.PATCH_SIZE,
                        hidden_size=gpc.config.HIDDEN_SIZE,
                        depth=gpc.config.DEPTH,
                        num_heads=gpc.config.NUM_HEADS,
                        mlp_ratio=gpc.config.MLP_RATIO,
                        num_classes=10,
                        init_method='jax',
                        checkpoint=gpc.config.CHECKPOINT)

    # set_trace()
    if use_pipeline:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = NASModel()
        exec_seq = ['conv1', 'conv2', 'conv3', 'gavg', (lambda x: torch.flatten(x, 1), "behind"), 'fc']
        logger.info(f"rank {grank}: {dict(pipelinable._model.named_modules()).keys()}")
        pipelinable.to_layer_list(exec_seq)
        logger.info(f"Before Pipeline: rank{grank} Pipeline param size={sum(p.numel() for p in model.parameters())/1e6}MB")
        pipelinable.policy = "balanced"
        model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
        rm = RandomMutator(model)
        rm.reset()
        logger.info(f"After Pipeline: rank{grank} arch={rm._cache}\n{model}")
        logger.info(f"After pipeline: rank{grank} Pipeline param size={sum(p.numel() for p in model.parameters())/1e6}MB")
    else:
        model = NASModel()
        rm = RandomMutator(model)
        logger.info(f"rank{grank} param size={sum(p.numel() for p in model.parameters())/1e6}MB")

    # count number of parameters
    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_stage = 0
    else:
        pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    logger.info(f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")

    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    # create dataloaders
    root = os.environ.get('DATA', '~/datasets/cifar10')
    train_dataloader, test_dataloader = build_cifar(gpc.config.BATCH_SIZE, root, pad_if_needed=True)

    # create loss function
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    # initialize
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         criterion=criterion,
                                                                         train_dataloader=train_dataloader,
                                                                         test_dataloader=test_dataloader)

    logger.info("Engine is built", ranks=[0])

    if gpc.is_initialized(ParallelMode.PARALLEL_1D):
        scatter_gather = True
    else:
        scatter_gather = False

    data_iter = iter(train_dataloader)

    for epoch in range(gpc.config.NUM_EPOCHS):
        # training
        engine.train()

        if gpc.get_global_rank() == 0:
            description = 'Epoch {} / {}'.format(epoch, gpc.config.NUM_EPOCHS)
            progress = tqdm(range(len(train_dataloader)), desc=description)
        else:
            progress = range(len(train_dataloader))
        for _ in progress:
            rm.reset()
            logger.info(f"rank{grank} arch={rm._cache}")
            engine.zero_grad()
            engine.execute_schedule(data_iter, return_output_label=False)
            engine.step()
            lr_scheduler.step()


if __name__ == '__main__':
    main()
