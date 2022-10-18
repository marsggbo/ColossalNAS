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

from hyperbox_app.distributed.networks.ofa import OFAMobileNetV3
from hyperbox.mutables.spaces import OperationSpace
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.networks.darts import DartsNetwork
from hyperbox.networks.spos import ShuffleNASNetV2
from hyperbox.mutator import RandomMutator

def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'

def get_ofa(width_mult=2):
    return OFAMobileNetV3(
        width_mult=width_mult, depth_list=[3,4], expand_ratio_list=[3,4],
        base_stage_width=[64, 128, 256, 512, 512, 1024, 1024, 1024, 2048],
        num_classes=10
    )

def get_darts():
    return DartsNetwork(in_channels=3, channels=24, n_classes=10, n_layers=24,
        n_nodes=4,stem_multiplier=5)

class NASModel(BaseNASNetwork):
    def __init__(self, model_init_func=None, is_search_inner=False, rank=None, mask=None):
        super().__init__(mask)
        self.conv1 = OperationSpace(
            [torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
             torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
             torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            ], key='conv1', mask=self.mask
        )
        self.conv2 = OperationSpace(
            [torch.nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False),
             torch.nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=2, bias=False),
             torch.nn.Conv2d(64, 96, kernel_size=7, stride=1, padding=3, bias=False),
            ], key='conv2', mask=self.mask
        )
        self.conv3 = OperationSpace(
            [torch.nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False),
             torch.nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2, bias=False),
             torch.nn.Conv2d(96, 128, kernel_size=7, stride=1, padding=3, bias=False),
            ], key='conv3', mask=self.mask
        )
        self.gavg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(128, 10)
        # self.model = model_init_func()
        # self.mutator = RandomMutator(self.model)
        # self.model = resnet18()
        # self.mutator = torch.nn.Linear(10,10)
        self.is_search_inner = is_search_inner
        self.rank = rank

    def forward(self, x):
        print(x.shape)
        out = self.conv1(x)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        out = self.conv3(out)
        print(out.shape)
        out = self.gavg(out).view(out.size(0), -1)
        print(out.shape)
        out = self.fc(out)
        return out
        # if self.is_search_inner:
        #     self.mutator.reset()
        #     if self.rank is not None:
        #         logger.info(f'Rank {self.rank}: arch={self.model.arch}')
        # return self.model(x)

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
            # model = _create_vit_model(**model_kwargs)
            # model = get_ofa()
            # model = get_darts()
            # model = NASModel()
            model = resnet18()
            # rm = RandomMutator(model)
            # model = NASModel(get_ofa, is_search_inner=True)
            # rm = None
        logger.info(f"Before Pipeline: rank{grank} Pipeline param size={sum(p.numel() for p in model.parameters())/1e6}MB")
        pipelinable.to_layer_list()
        # pipelinable.policy = "uniform"
        pipelinable.policy = "balanced"
        model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
        rm = RandomMutator(model)
        rm.reset()
        logger.info(f"After Pipeline: rank{grank} arch={rm._cache}")
        logger.info(f"After pipeline: rank{grank} Pipeline param size={sum(p.numel() for p in model.parameters())/1e6}MB")
    else:
        # model = _create_vit_model(**model_kwargs)
        model = get_ofa()
        # model = get_darts()
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
    root = os.environ.get('DATA', './data')
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
            # progress = tqdm(range(len(train_dataloader)), desc=description)
            progress = tqdm(train_dataloader, desc=description)
        else:
            # progress = range(len(train_dataloader))
            progress = train_dataloader
        # for _ in progress:
        for idx, (imgs, labels) in enumerate(progress):
            if rm is not None:
                rm.reset()
            
            engine.zero_grad()
            output = engine(imgs.to(gpc.get_global_rank()))
            logger.info(f"rank{grank} imgs.shape={imgs.shape} labels.shape={labels.shape} output size={output.size()}")
            # logger.info(get_mem_info(prefix=f'[{idx+1}/{len(progress)}] Forward '), ranks=[0])
            loss = engine.criterion(output, labels.to(gpc.get_global_rank()))
            engine.backward(loss)
            # logger.info(get_mem_info(prefix=f'[{idx+1}/{len(progress)}] Backward '), ranks=[0])
            engine.step()
            # logger.info(get_mem_info(prefix=f'[{idx+1}/{len(progress)}] Optimizer step '), ranks=[0])
            
            # engine.zero_grad()
            # rm.reset()
            # engine.execute_schedule(data_iter, return_output_label=False)
            # engine.step()
            # lr_scheduler.step()


if __name__ == '__main__':
    main()
