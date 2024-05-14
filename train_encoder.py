# %% Preparing dataset
from utils import get_filenames, get_train_data, get_batches
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

BATCH_SIZE = 512
COST_THRESHOLD = 800 # This threshold can be more lenient for encoder

filenames = []
filedirs = [f"./data/SYNTHETIC_V{i}" for i in range(1,2+1)]

for filedir in filedirs:

    temp = get_filenames(filedir)

    with open(f"{filedir}/cost.txt", "r") as f: 
        cost = f.readlines()
    cost = [float(c) for c in cost]

    # Cost filtering
    for ii, c in enumerate(cost):
        if c < COST_THRESHOLD:
            filenames.append(temp[ii])

trajectories = get_train_data(filenames, split="all")
train_data = get_batches(trajectories, batch_size=BATCH_SIZE, batch_first=False, combined=False)
train_data, test_data = train_data[:-5], train_data[-5:]

print(f"Percentage of data used: {len(filenames)/(200*len(filedirs))} %")
print(f"Number of trajectories: {len(trajectories)}")
print(f"Number of training batches : {len(train_data)}")
print(f"Number of evaluation batches : {len(test_data)}")
print(f"Input shape  : {test_data[0][0].shape}")
print(f"Target shape : {test_data[0][1].shape}")

# %% Train seed model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import Encoder
import random

CONTEXT_LENGTH = 100
EPOCHS = 40
lr = 1e-4
lr_min = 1e-6
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Encoder(
    d_input=4,
    d_model=64,
    num_layers=4,
    seq_len=CONTEXT_LENGTH,
    dropout=0.05
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,lr_min)
scheduler1 = optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, len(train_data))
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_data)*(EPOCHS-1),lr_min)
scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
offset = torch.randint(0, 600-CONTEXT_LENGTH, (1,))

for ii in range(EPOCHS):
    losses = []
    model.train()
    for input_data, steer in tqdm(train_data):
        input_data = input_data[offset:offset+CONTEXT_LENGTH].to(device)
        input_data = input_data + 0.001*torch.randn_like(input_data, device=device)
        steer = steer[offset:offset+CONTEXT_LENGTH].to(device)
    
        # Forward pass
        output = model(input_data)
        loss = criterion(output.squeeze(), steer)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        scheduler.step()

    torch.save(model.state_dict(), f"./models/encoder-128-6-seed-{seed}.pt")

    # Evaluation
    eval_losses = []
    model.eval()
    for input_data, steer in tqdm(test_data):
        input_data = input_data[offset:offset+CONTEXT_LENGTH].to(device)
        steer = steer[offset:offset+CONTEXT_LENGTH].to(device)
    
        # Forward pass
        with torch.no_grad():
            output = model(input_data)
            loss = criterion(output.squeeze(), steer)

        # Backward pass
        eval_losses.append(loss.item())

    print(f"Epoch: {ii+1}  Train Loss: {np.mean(losses)}  Eval Loss: {np.mean(eval_losses)}")
