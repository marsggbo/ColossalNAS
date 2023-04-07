import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_dataset(img_size=32, path='/datasets/cifar10'):
    # Create a CIFAR-10 train dataset and dataloader with 4 workers
    transform = transforms.Compose([
        transforms.Resize(int(img_size/0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    return trainset, testset

def get_cifar10_dataloader(batch_size=32, img_size=32, path='/datasets/cifar10'):
    # Create a CIFAR-10 train dataset and dataloader with 4 workers
    trainset, testset = get_cifar10_dataset(img_size, path)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader
