import torch
from torch.utils.data import Dataset, DataLoader

class FakeDataset(Dataset):
    def __init__(self, data_size=100, img_size=224):
        self.data_size = data_size
        self.img_size = img_size
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # Generate a random input and output for the fake data
        input_data = torch.randn(3, self.img_size, self.img_size)
        output_data = torch.randint(low=0, high=1000, size=(1,))
        return input_data, output_data

def get_fake_dataloader(data_size=30, img_size=224, batch_size=10):
    fake_data_loader = DataLoader(
        dataset=FakeDataset(data_size, img_size), batch_size=batch_size, shuffle=True)
    return fake_data_loader

if __name__ == '__main__':
    dataloader = get_fake_dataloader(100)
    dataiter = iter(dataloader)
    for i in range(10):
        inputs, labels = dataiter.next()
        print(inputs.shape, labels.shape)