import torch
import torchvision
from torchvision.models import resnet50
import deepspeed
import argparse

def add_argument():
    parser = argparse.ArgumentParser(description='Activation Checkpointing')
    
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


args = add_argument()
deepspeed.init_distributed()
model = resnet50(pretrained=False)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Load the training dataset.
train_dataset = torchvision.datasets.CIFAR10(root='/home/xihe/datasets/cifar10', train=True, download=True, transform=torchvision.transforms.ToTensor())

# Wrap the dataset with DeepSpeed to enable efficient data loading.
train_loader = deepspeed.ops.dataset.parallel.DistributedLoader(train_dataset, num_workers=4, batch_size=32)

# Wrap the model and optimizer with DeepSpeed.
model, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_dataset)

# Enable activation checkpointing with a maximum number of layers per partition.
model = deepspeed.ops.checkpointing.checkpoint_wrapper(model, num_checkpoints=2)

# Train the model.
for epoch in range(10):
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
