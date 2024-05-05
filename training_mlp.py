# %%
from utils import get_train_data
import numpy as np
import torch
import os

input_dir = "./data/"
train_data = get_train_data(input_dir, "all")

# %%

aa = torch.zeros(4)
bb = torch.zeros(4)
ma, mi = 0, 0
for data in train_data:
    a, b = torch.std_mean(data[1])
    ma += torch.max(data[1])
    mi += torch.min(data[1])
    aa += a
    bb += torch.square(b)
print(aa/len(train_data))
print(torch.sqrt(bb)/len(train_data))
print(ma/len(train_data))
print(mi/len(train_data))

# %%
EPOCHS = 100
BATCH_SIZE = 64
batches = []

for i in range(len(train_data)):
    input_data, target = [], []
    for j in range(BATCH_SIZE):
        if i*BATCH_SIZE+j >= len(train_data):
            break
        x, y = train_data[i*BATCH_SIZE+j]
        input_data.append(x)
        target.append(y)
    if not input_data: break
    input_data = torch.stack(input_data)
    target = torch.stack(target)
    batches.append((input_data, target))

print(len(batches))


# %%
from models import MyMLP
import torch.nn as nn
import torch.optim as optim

# Initialize the model, loss function, and optimizer
model = MyMLP(d_input=4, d_hidden=64, d_out=1)
criterion1 = nn.L1Loss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

# %%

for ii in range(EPOCHS):
    losses = []
    for input_data, target in batches:

        # Forward pass
        output = model(input_data)
        loss = criterion1(output.squeeze(), target) + criterion2(output.squeeze(), target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Step: {ii}  Loss: {np.mean(losses)}")
    
    torch.save(model.state_dict(), "./models/mlp.pt")