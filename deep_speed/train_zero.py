import argparse
from time import time

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from hyperbox.mutator import RandomMutator
from hyperbox.networks.darts import DartsNetwork
from hyperbox.networks.vit import ViT_10B, ViT_B, ViT_G, ViT_H, ViT_S
from model_zoo import get_model


def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=8,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")

    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        type=int,
                        nargs='+',
                        default=[
                            1,
                        ],
                        help='number of experts list, MoE related.')
    parser.add_argument(
        '--mlp-type',
        type=str,
        default='standard',
        help=
        'Only applicable when num-experts > 1, accepts [standard, residual]')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )
    parser.add_argument('--model',
                        type=str,
                        default='vit_b',
                        help='model name')
    parser.add_argument('--steps',
                        type=int,
                        default=50,
                        help='steps')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def get_datasets(DATA_PATH='/home/xihe/datasets/cifar10'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()

    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH,
                                            train=True,
                                            download=True,
                                            transform=transform)

    testset = torchvision.datasets.CIFAR10(root=DATA_PATH,
                                        train=False,
                                        download=True,
                                        transform=transform)

    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
            'ship', 'truck')
    return trainset, testset, trainloader, testloader, classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if args.moe:
            fc3 = nn.Linear(84, 84)
            self.moe_layer_list = []
            for n_e in args.num_experts:
                # create moe layers based on the number of experts
                self.moe_layer_list.append(
                    deepspeed.moe.layer.MoE(
                        hidden_size=84,
                        expert=fc3,
                        num_experts=n_e,
                        ep_size=args.ep_world_size,
                        use_residual=args.mlp_type == 'residual',
                        k=args.top_k,
                        min_capacity=args.min_capacity,
                        noisy_gate_policy=args.noisy_gate_policy))
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
            self.fc4 = nn.Linear(84, 10)
        else:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if args.moe:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x

 
args = add_argument()
deepspeed.init_distributed()
trainset, testset, trainloader, testloader, classes = get_datasets()

# net = ViT_10B(num_classes=10)
# net = ViT_S(num_classes=10)
# net = DartsNetwork(n_layers=2, n_nodes=4)
net = get_model(args.model)
rm = RandomMutator(net)

def create_moe_param_groups(model):
    from deepspeed.moe.utils import \
        split_params_into_different_moe_groups_for_optimizer

    parameters = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }

    return split_params_into_different_moe_groups_for_optimizer(parameters)

parameters = filter(lambda p: p.requires_grad, net.parameters())
if args.moe_param_group:
    parameters = create_moe_param_groups(net)

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset)

fp16 = model_engine.fp16_enabled()
print(f'fp16={fp16}')

criterion = nn.CrossEntropyLoss()
rank = model_engine.local_rank
world_size = torch.distributed.get_world_size()
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    epoch_start = time()
    BS = 0
    for i, data in enumerate(trainloader):
        if i+1 > args.steps:
            break
        rm.reset()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
        start = time()
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()
        end = time()
        BS = inputs.shape[0]
        throughput = BS * world_size / (end - start)
        calc = f"{BS} (BS) * {world_size}($gpus) / {end - start:.2f}(time)"
        torch.cuda.synchronize()
        if rank == 0:
            print(f'[rank{rank}] OMG~~~ throughput is {calc}={throughput:.2f} img/sec loss: {loss}')

        # print statistics
        running_loss += loss.item()
        # if i % args.log_interval == (
        #         args.log_interval -
        #         1):  # print every log_interval mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / args.log_interval))
        #     running_loss = 0.0
    epoch_end = time()
    throughput = BS * args.steps * world_size / (epoch_end - epoch_start)
    calc = f"{BS} (BS) * {args.steps} (steps) * {world_size}($gpus) / {epoch_end - epoch_start:.2f}(time)"
    print(f'[rank{rank}] OMG~~~ all throughput is {calc}={throughput:.2f} img/sec loss: {loss}')

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.


########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         if fp16:
#             images = images.half()
#         outputs = net(images.to(model_engine.local_rank))
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels.to(
#             model_engine.local_rank)).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' %
#       (100 * correct / total))

# ########################################################################
# # That looks way better than chance, which is 10% accuracy (randomly picking
# # a class out of 10 classes).
# # Seems like the network learnt something.
# #
# # Hmmm, what are the classes that performed well, and the classes that did
# # not perform well:

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         if fp16:
#             images = images.half()
#         outputs = net(images.to(model_engine.local_rank))
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels.to(model_engine.local_rank)).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

# for i in range(10):
#     print('Accuracy of %5s : %2d %%' %
#           (classes[i], 100 * class_correct[i] / class_total[i]))