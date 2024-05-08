# %%
from utils import *

COST_THRESHOLD = 400
BATCH_SIZE = 512

filenames = []
filedirs = [f"./data/SYNTHETIC_V{i}" for i in range(1,2+1)]
# filedirs = [f"./data/SYNTHETIC_V1"]
for filedir in filedirs:
    temp = get_filenames(filedir)

    with open(f"{filedir}/cost.txt", "r") as f: 
        cost = f.readlines()
    cost = [float(c) for c in cost]

    # Cost filtering
    for ii, c in enumerate(cost):
        if c < COST_THRESHOLD:
            filenames.append(temp[ii])

print(len(filenames))
trajectories = get_train_data(filenames, split="all")
train_data = get_batches(trajectories, batch_size=BATCH_SIZE, batch_first=False, combined=True)
train_data, test_data = train_data[:-10], train_data[-10:]

print(f"Percentage of data used: {len(filenames)/(200*len(filedirs))} %")
print(f"Number of trajectories: {len(trajectories)}")
print(f"Number of training batches : {len(train_data)}")
print(f"Number of evaluation batches : {len(test_data)}")
print(f"Data shape  : {train_data[0].shape}")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import Decoder
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.cuda.empty_cache()

context_length = 100
# 64-2 has much potential
model = Decoder(d_input=5, d_model=128, num_layers=6, seq_len=context_length, dropout=0.05).cuda()
model.train()
criterion = nn.MSELoss()

EPOCHS = 100
lr = 1e-3
lr_min = 1e-6
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler1 = optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, len(train_data))
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_data)*(EPOCHS-1),lr_min)
scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
offset = torch.randint(0, 600-context_length, (1,))

for ii in range(EPOCHS):
    losses = []
    model.train()
    for data in tqdm(train_data):
        data = data[offset:offset+context_length]
        data = data.cuda()
    
        # Forward pass
        data = data + torch.randn_like(data) * 0.001
        output = model(data)
        loss = criterion(output[:-1,:], data[1:,:,-1:]) + 0.01*torch.mean(torch.diff(output, dim=0)**2)

        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        scheduler.step()

    torch.save(model.state_dict(), "./models/decoder.pt")

    # Evaluation
    eval_losses = []
    model.eval()
    for data in tqdm(test_data):
        data = data[offset:offset+context_length]
        data = data.cuda()
        # Forward pass
        with torch.no_grad():
            data = data + torch.randn_like(data) * 0.001
            output = model(data)
            loss = criterion(output[:-1,:], data[1:,:,-1:]) + 0.01*torch.mean(torch.diff(output, dim=0)**2)
        eval_losses.append(loss.item())
    print(f"Epoch: {ii+1}  Train Loss: {np.mean(losses)}  Eval Loss: {np.mean(eval_losses)}")


# %%
import matplotlib.pyplot as plt

model.eval()
# offset = torch.randint(0, 500, (1,)).item()
# data = train_data[torch.randint(0,len(train_data),(1,)).item()][offset:offset+100,:1].cuda()
# data[:,:,1] = nn.functional.normalize(data[:,:,1], dim=0)
# data[:,:,1] = data[:,:,1]/40
print(data.shape)
print(torch.mean(data[:,0], dim=0))
print(model(data).shape)
plt.plot(data.detach().cpu()[:,1,-1])
plt.plot(model(data).detach().cpu()[:,1])
plt.legend(["gt", "pred"])
plt.show()