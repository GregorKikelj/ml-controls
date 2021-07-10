import numpy as np
from random import random

def sample_generator():
    while True:
        x = random()
        y = int(10*x)/10
        yield x, y

import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

class SimDataset(Dataset):
    def __init__(self, length, i, o):
        self.len = length
        self.i = i
        self.o = o

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.i[idx], self.o[idx]

generator = sample_generator()

def random_data_dataset(length):
    raw_in = [next(generator) for _ in range(length)]
    #print(len(raw_in))

    i_data = []
    o_data = []
    for i, data in enumerate(raw_in):
        x = data[0]
        y = data[1]
        data_input = torch.zeros((1,))
        data_input[0]=x
        data_output = torch.zeros((1,))
        data_output[0]=y
        i_data.append(data_input)
        o_data.append(data_output)
    return SimDataset(length, i_data, o_data)


in_set = random_data_dataset(1000)
out_set = random_data_dataset(1000)
train_dataloader = DataLoader(in_set, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(out_set, batch_size=64, shuffle=False)

def train_loop(dataloader, model, loss_fn, optimizer, debug):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if debug:
        loss = loss.item()
        print(f"loss: {loss:>10f}")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: {test_loss:>10f} \n")

model = NeuralNetwork()
model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 101
print(len(train_dataloader))
print(len(test_dataloader))
for t in range(epochs):
    if t%5 == 0:
        print(f"Epoch {t}")
    train_loop(train_dataloader, model, loss_fn, optimizer, t%10==0)
    if t%5 == 0:
        test_loop(test_dataloader, model, loss_fn)
print("Done!")
