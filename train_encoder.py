# %% Preparing dataset
from utils import get_filenames, get_train_data, get_batches

BATCH_SIZE = 512

filedir = "./data/SYNTHETIC_V0/"
filenames = get_filenames(filedir)
trajectories = get_train_data(filenames, split="labelled", num_files=20000)
train_data = get_batches(trajectories, batch_size=BATCH_SIZE)

print(f"Number of training trajectories: {len(trajectories)}")
print(f"Number of batches : {len(train_data)}")
print(f"Input shape  : {train_data[0][0].shape}")
print(f"Target shape : {train_data[0][0].shape}")

# %% Train seed model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import Encoder
import random

seed = 42
random.seed(seed)
np.random.seed(seed)

model = Encoder(d_input=4, d_model=128, num_layers=6).cuda()
model.train()
criterion = nn.MSELoss()
# criterion = nn.HuberLoss()

EPOCHS = 20
lr = 1e-4
lr_min = 1e-6
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,lr_min)

for ii in range(EPOCHS):
    losses = []
    for input_data, target in tqdm(train_data[:20]):

        input_data = input_data.cuda() + 0.005*torch.randn_like(input_data, device="cuda")
        target = target.cuda()
    
        # Forward pass
        # with torch.autocast(device_type="cuda"):
        output = model(input_data)
        loss = criterion(output.squeeze(), target)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    scheduler.step()

    print(f"Epoch: {ii+1}  Loss: {np.mean(losses)}")
    
    torch.save(model.state_dict(), "./models/encoder.pt")
