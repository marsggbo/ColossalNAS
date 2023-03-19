import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_cifar10_dataset(path='/datasets/cifar10'):
    # Create a CIFAR-10 train dataset and dataloader with 4 workers
    trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    return trainset, testset

def get_cifar10_dataloader(batch_size, path='/datasets/cifar10'):
    # Create a CIFAR-10 train dataset and dataloader with 4 workers
    trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
