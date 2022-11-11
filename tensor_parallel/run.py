import colossalai
import colossalai.nn as col_nn
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device, print_rank_0
from colossalai.global_variables import tensor_parallel_env as tp_env
import torch
import torch.nn as nn

class MLP(torch.nn.Module):

    def __init__(self, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = col_nn.Linear(dim, intermediate_dim)
        print_rank_0(f'Weight of the first linear layer: {self.dense_1.weight.shape}')
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)
        print_rank_0(f'Weight of the second linear layer: {self.dense_2.weight.shape}')
        self.dropout = col_nn.Dropout(0.1)

    def forward(self, x):
        x = self.dense_1(x)
        print_rank_0(f'Output of the first linear layer: {x.shape}')
        x = self.activation(x)
        x = self.dense_2(x)
        print_rank_0(f'Output of the second linear layer: {x.shape}')
        x = self.dropout(x)
        return x

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
        print(f"rank{self.rank} out1 {out.shape}")
        if self.count % 2 == 0:
            out = self.conv2(out)
        else:
            out = self.conv3(out)
        self.count += 1
        print(f"rank{self.rank} out2 {out.shape}")
        out = self.gavg(out).view(out.size(0), -1)
        out = self.fc(out)
        return out

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


def main():
    colossalai.logging.disable_existing_loggers()
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(config=args.config)
    grank = gpc.get_global_rank()

    # m = MLP()

    # x = torch.randn((16, 256), device=get_current_device())
    
    m = NASModel().to(get_current_device())
    m.rank = grank
    x = torch.rand(16, 3, 32, 32, device=get_current_device())
    torch.distributed.broadcast(x, src=0)

    # partition input
    if tp_env.mode == '1d':
        pass
    elif tp_env.mode == '2d':
        print(f"ParallelMode.PARALLEL_2D_COL={gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)}")
        print(f"ParallelMode.PARALLEL_2D_ROW={gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)}")
        x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)]
        x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)]
    elif tp_env.mode == '2.5d':
        print(f"ParallelMode.PARALLEL_2P5D_DEP={gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)}")
        print(f"ParallelMode.PARALLEL_2P5D_COL={gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)}")
        print(f"ParallelMode.PARALLEL_2P5D_ROW={gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)}")
        x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)]
        x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)]
        x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)]
    elif tp_env.mode == '3d':
        print(f"ParallelMode.PARALLEL_3D_WEIGHT={gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)}")
        print(f"ParallelMode.PARALLEL_3D_INPUT={gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)}")
        print(f"ParallelMode.PARALLEL_3D_OUTPUT={gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)}")
        x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)]
        x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)]
        x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)]
    print_rank_0(f'Input: {x.shape}')
    print(f"rank{grank} input: {x.shape}")
    DeviceInfo(m)

    y = m(x)
    print(f"rank{grank} output: {y.shape}")
    gpc.destroy()


if __name__ == '__main__':
    main()
