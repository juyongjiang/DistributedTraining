import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"

# 1) Initialize the backend of computation
torch.distributed.init_process_group(backend="nccl")

# 2） Configure the gpu of each process
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class RandomDataset(Dataset):
    def __init__(self, length, size):
        self.len = length
        self.data = torch.randn(length, size)
        self.labels = torch.cat([torch.zeros((length//2,)), torch.ones((length - length//2,))], 0)

    def __getitem__(self, index):
        return self.data[index], self.labels[index].long()

    def __len__(self):
        return self.len

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output

if __name__ == "__main__":
    data_size, batch_size, input_size, output_size = 900, 32, 5, 2
    dataset = RandomDataset(data_size, input_size)
    # 3）Use DistributedSampler to distribute data to each gpu 
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              sampler=DistributedSampler(dataset))

    model = Model(input_size, output_size)
    # 4）Move the model to each gpu
    model.to(device)
    # 5）Wrap up model
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    for epoch in range(100):
        for batch_idx, (data, label) in enumerate(train_loader):
            if torch.cuda.is_available():
                data = data.to('cuda')
                label = label.to('cuda')

            output = model(data)
            loss = loss_function(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} Loss: {}'.format(epoch, loss.item()))