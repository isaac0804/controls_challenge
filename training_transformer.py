"""
The seed model will be trained on first 100 steps of trajectory with bidirectional attention to predict steer command.
The idea is that with future info, the model will learn to predict the steer command more accurately.
Then all the trajectories will be labelled by this seed model.

The controller model will be trained on ground truth and generated data by the seed model.
The controller model will predict the steer command and latAccel of next step autoregressively.

The seed model and controller model can be retrained iteratively.
Question 1: Do we need to filter on generated data? (based on the loss?)
Question 2: During testing, seed model or controller model will perform better?
Question 3: Do we train next model using data generated by seed OR controller data?
"""

# %% Preparing dataset
from utils import get_train_data, create_batches

trajectories = get_train_data("./data/", split="labelled", num=20000)
print(f"Number of training trajectories: {len(trajectories)}")
print(f"Input shape  : {trajectories[0][0].shape}")
print(f"Target shape : {trajectories[0][0].shape}")

train_data = create_batches(trajectories, batch_size=256)
print(f"Number of batches : {len(train_data)}")
print(f"Input shape  : {train_data[0][0].shape}")
print(f"Target shape : {train_data[0][0].shape}")

del trajectories

# %% Train seed model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import Encoder

model = Encoder(d_input=4, d_model=128, num_layers=4).cuda()
model.train()
# criterion = nn.L1Loss()
criterion = nn.MSELoss()

EPOCHS = 10
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

for ii in range(EPOCHS):
    losses = []
    for input_data, target in tqdm(train_data):

        input_data = input_data.cuda()
        target = target.cuda()

        # Forward pass
        output = model(input_data)
        # loss = criterion1(output.squeeze(), target) + criterion2(output.squeeze(), target)
        loss = criterion(output.squeeze(), target) + 0.01*torch.mean(output[:99,:]-output[1:,:])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch: {ii+1}  Loss: {np.mean(losses)}")
    
    torch.save(model.state_dict(), "./models/encoder.pt")

# %% Inference / Generate Data
import torch
from utils import * 
import matplotlib.pyplot as plt

batch_size = 32
trajectories = get_train_data("./data/", split="all", num=200)
inference_data = create_batches(trajectories, batch_size)
context_length = 100
overlap_width = 50
mask = torch.concat([torch.linspace(0,1,overlap_width), torch.ones(context_length-2*overlap_width), torch.linspace(1,0,overlap_width)]).unsqueeze(1).cuda()
first_mask = torch.concat([torch.ones(context_length-overlap_width), torch.linspace(1,0,overlap_width)]).unsqueeze(1).cuda()
generated_data = []

with torch.no_grad():
    for input_data, target in tqdm(inference_data):

        # TODO: Verify len of target == 100 for every file
        # TODO: Pad len with to 600
        input_data = input_data.cuda()
        target = target[:100].cuda()
        outputs = torch.concat([first_mask*target, torch.zeros(500, batch_size).cuda()])

        for index in range(50,500+1,50):
            data = input_data[index:min(index+100,input_data.shape[0])]
            output = model(data)
            outputs[index:min(index+100,input_data.shape[0])] += mask*output.squeeze()
        break
print(outputs.shape)
plt.plot(outputs[:,2].cpu())
plt.show()

# %%

outputs = outputs.cpu()
print(input_data.shape)
print(outputs.shape)
plt.plot(outputs[:,0])
plt.show()


# %% Train controller model
