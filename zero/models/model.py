
import types
from functools import partial

import torch
import torch.nn as nn
from torchvision import models

from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.mutables import ops, spaces
from hyperbox.mutables.spaces import OperationSpace
from hyperbox.mutator import RandomMutator

from hyperbox.networks.ofa import OFAMobileNetV3
from hyperbox.networks.darts import DartsNetwork
from hyperbox.networks.vit import *
from hyperbox.networks.mobilenet.mobile_net import MobileNet


def get_vit(cls=VisionTransformer, **kwargs):
    use_ac = False
    if 'use_ac' in kwargs:
        use_ac = kwargs.pop('use_ac')
    if cls is VisionTransformer:
        default_config = {
            'image_size': 224, 'patch_size': 16, 'num_classes': 1000, 'dim': 512, 
            'depth': 6, 'heads': 8, 'dim_head': 512, 'mlp_dim': 512, 'search_ratio': [1]
        }
        default_config.update(kwargs)
        net = cls(**default_config)
    else:
        net = cls()
        # net = cls(search_ratio=[1])
    if use_ac:
        def forward(self, x):
            out = self.vit_embed(x)
            if self.to_search_path:
                vit_blocks = list(self.vit_blocks.children())
                runtime_depth = self.run_depth.value
                vit_blocks = vit_blocks[:runtime_depth]
                vit_blocks = nn.Sequential(*vit_blocks)
                out = vit_blocks(out)
            else:
                for i, block in enumerate(self.vit_blocks):
                    try:
                        from colossalai.utils.activation_checkpoint import checkpoint
                        out = checkpoint(block, True, out, True)
                        print(f'block {i} is finished')
                    except ImportError:
                        out = torch.utils.checkpoint.checkpoint(block, out) 
            out = self.vit_cls_head(out)
            return out
    return net

def get_darts(**kwargs):
    default_config = {
        'in_channels': 3, 'channels': 64, 'n_classes': 1000, 'n_layers': 8, 'auxiliary': False,
        'n_nodes': 4, 'stem_multiplier': 3
    }
    default_config.update(kwargs)
    use_ac = False
    if 'use_ac' in default_config:
        use_ac = default_config.pop('use_ac')
    net = DartsNetwork(**default_config)
    if use_ac:
        print('using activation checkpointing for darts')
        def forward(self, pprev, prev):
            tensors = [self.preproc0(pprev), self.preproc1(prev)]
            for idx, node in enumerate(self.mutable_ops):
                try:
                    from colossalai.utils.activation_checkpoint import checkpoint
                    cur_tensor = checkpoint(node, True, tensors, use_reentrant=True)
                except ImportError:
                    cur_tensor = torch.utils.checkpoint.checkpoint(node, tensors)
                tensors.append(cur_tensor)

            output = torch.cat(tensors[2:], dim=1)
            return output
        from hyperbox.networks.darts.darts_network import DartsCell
        for module in net.modules():
            if isinstance(module, DartsCell):
                module.forward = types.MethodType(forward, module)
    return net

def get_ofa(**kwargs):
    default_config = {
        'width_mult': 1.0, 'depth_list': [4,5], 'expand_ratio_list': [2,3],
        'base_stage_width': [16, 32, 64, 128, 256, 320, 480, 512, 960],
        'to_search_depth': False
    }
    # base_stage_width=[32, 64, 128, 256, 512, 512, 512, 960, 1024]
    default_config.update(kwargs)
    use_ac = False
    if 'use_ac' in default_config:
        use_ac = default_config.pop('use_ac')
    net = OFAMobileNetV3(**default_config)
    if use_ac:
        print('using activation checkpointing for ofa')
        def forward(self, x):
            x = self.stem_layer(x)
            if self.to_search_depth:
                x = self.blocks[0](x)
                # inverted residual blocks
                for stage_id, block_group in enumerate(self.block_group_info):
                    depth = self.runtime_depth[stage_id].value
                    active_idx = block_group[:depth]
                    for idx in active_idx:
                        x = self.blocks[idx](x)
            else:
                for block in self.blocks:
                    try:
                        from colossalai.utils.activation_checkpoint import checkpoint
                        x = checkpoint(block, True, x, use_reentrant=True)
                        # print('using deepspeed checkpointing')
                    except ImportError:
                        from torch.utils.checkpoint import checkpoint
                        # print('using torch checkpointing')
                        x = checkpoint(block, x)
            x = self.final_expand_layer(x)
            x = self.avg_pool(x)
            x = self.feature_mix_layer(x)
            x = self.classifier(x)
            return x
        net.forward = types.MethodType(forward, net)
    return net

def get_mobilenet(**kwargs):
    default_config = {
        'c_in': 3, 'first_stride': 1, 'width_stages': [24,40,80,96,192,320],
        'n_cell_stages': [4,4,4,4,4,1], 'stride_stages': [2,2,2,1,2,1],
        'op_list': None, 'width_mult': 1, 'classes': 1000, 'dropout_rate': 0,
        'bn_param': (0.1, 1e-3), 'mask': None
    }
    default_config.update(kwargs)
    use_ac = False
    if 'use_ac' in default_config:
        use_ac = default_config.pop('use_ac')
    net = MobileNet(**default_config)
    if use_ac:
        print('using activation checkpointing for mobilenet')
        def forward(self, x):
            x = self.first_conv(x)
            for block in self.blocks:
                try:
                    from colossalai.utils.activation_checkpoint import checkpoint
                    x = checkpoint(block, True, x, use_reentrant=True)
                    # print('using deepspeed checkpointing')
                except ImportError:
                    from torch.utils.checkpoint import checkpoint
                    # print('using torch checkpointing')
                    x = checkpoint(block, x)
            x = self.feature_mix_layer(x)
            x = self.global_avg_pooling(x)
            x = self.classifier(x)
            return x
        net.forward = types.MethodType(forward, net)
    return net


class ToyNASModel2(BaseNASNetwork):
    def __init__(self, model_init_func=None, is_search_inner=False, rank=None, mask=None):
        super().__init__(mask)
        self.flag = 2
        if self.flag==1:
            self.conv1 = torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False)
        else:
            self.conv1 = OperationSpace(
                [torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.Conv2d(3, 512, kernel_size=5, stride=1, padding=2, bias=False),
                torch.nn.Conv2d(3, 512, kernel_size=7, stride=1, padding=3, bias=False),
                ], key='conv1', mask=self.mask
            )
            self.conv2 = OperationSpace(
                [torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False),
                torch.nn.Conv2d(512, 1024, kernel_size=7, stride=1, padding=3, bias=False),
                ], key='conv2', mask=self.mask
            )
            self.conv3 = OperationSpace(
                [torch.nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),
                 torch.nn.Conv2d(1024, 2048, kernel_size=5, stride=1, padding=2, bias=False),
                #  torch.nn.Conv2d(1024, 2048, kernel_size=7, stride=1, padding=3, bias=False),
                ], key='conv3', mask=self.mask
            )
        self.gavg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1024, 1000, bias=False)
        self.is_search_inner = is_search_inner
        self.rank = rank
        self.count = 0

    def forward(self, x):
        out = self.conv1(x)
        if self.flag==1:
            if self.count % 2 == 0:
                out = self.conv2(out)
            else:
                out = self.conv3(out)
        else:
            from colossalai.utils.activation_checkpoint import checkpoint
            out = checkpoint(self.conv2, True, out, use_reentrant=True)
            # out = self.conv2(out)
        self.count += 1
        out = self.gavg(out).view(out.size(0), -1)
        out = self.fc(out)
        return out


class ToyNASModel(BaseNASNetwork):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=True)
        channels = spaces.ValueSpace([512, 1024])
        self.conv2 = ops.Conv2d(512, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = ops.Conv2d(512, channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn = ops.BatchNorm2d(channels)
        self.gavg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ops.Linear(channels, 1000, bias=True)
        self.count = 0

    def forward(self, x):
        out = self.conv1(x)
        # from colossalai.utils.activation_checkpoint import checkpoint
        if self.count % 2 == 0:
            # out = checkpoint(self.conv2, True, out, use_reentrant=True)
            print('running conv2')
            out = self.conv2(out)
        else:
            # out = checkpoint(self.conv3, True, out, use_reentrant=True)
            print('running conv3')
            out = self.conv3(out)
        # # out = checkpoint(self.conv2, True, out, use_reentrant=True)
        # out = self.conv2(out)
        self.count += 1
        # out = self.bn(out)
        out = self.gavg(out).view(out.size(0), -1)
        out = self.fc(out)
        return out


name2model = {
    'ofa': get_ofa,
    'vit_vanilla': get_vit,
    'vit_s': partial(get_vit, cls=ViT_S),
    'vit_b': partial(get_vit, cls=ViT_B),
    'vit_l': partial(get_vit, cls=ViT_L),
    'vit_h': partial(get_vit, cls=ViT_H),
    'vit_g': partial(get_vit, cls=ViT_G),
    'vit_10b': partial(get_vit, cls=ViT_10B),
    'darts': get_darts,
    'toy': ToyNASModel,
    'toy2': ToyNASModel2,
    'mobilenet': get_mobilenet,
    'resnet18': models.resnet18,
    'resnet152': models.resnet152,
}

def get_model(name, args=None, **kwargs):
    model_class = name2model[name]
    if name == 'mobilenet':
        kwargs['classes'] = 10
    elif name == 'ofa':
        kwargs['num_classes'] = 10
    elif name.startswith('vit'):
        kwargs['num_classes'] = 10
    if name in ['darts', 'mobilenet'] or name.startswith('vit'):
        kwargs['use_ac'] = args.use_ac
        print(f"Use ac: {args.use_ac}")
    ins = model_class(**kwargs)
    return ins


if __name__ == '__main__':
    from hyperbox.mutator import RandomMutator
    x = torch.rand(2,3,224,224)
    net = get_model('vit_s')
    rm = RandomMutator(net)
    for i in range(10):
        print(rm.reset())
        y = (net(x))