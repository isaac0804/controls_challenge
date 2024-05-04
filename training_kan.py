# %%
from tinyphysics import ACC_G
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
from kan import KAN

input_dir = "./data/"
filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

dataset = {}
train_input = []
train_label = []
test_input = []
test_label = []

for ii, filename in tqdm(enumerate(filenames)):
    df = pd.read_csv(filename)[:100]

    roll = torch.tensor(np.sin(df['roll'].values) * ACC_G)
    v_ego = torch.tensor(df['vEgo'].values)
    # a_ego = torch.tensor(df['aEgo'].values)
    # target_lataccel = torch.tensor(df['targetLateralAcceleration'].values)
    steer_command = torch.tensor(df['steerCommand'].values).float()
    # input_data = torch.stack([roll, v_ego, a_ego, target_lataccel]).float().T
    input_data = torch.stack([roll, v_ego]).float().T


    if ii < 1000:
        test_input.append(input_data)
        test_label.append(steer_command.unsqueeze(-1))
    else:
        train_input.append(input_data)
        train_label.append(steer_command.unsqueeze(-1))

dataset["train_input"] = torch.cat(train_input)
dataset["train_label"] = torch.cat(train_label)
dataset["test_input"] = torch.cat(test_input)
dataset["test_label"] = torch.cat(test_label)

# %%
import random

selections = np.random.randint(0, len(dataset["train_input"])-1, 10000)
dataset["train_input"] = torch.cat(train_input)[selections]
dataset["train_label"] = torch.cat(train_label)[selections]
print(selections)

# %%

# Initialize the model, loss function, and optimizer
model = KAN(width=[2,1,1,1], grid=10, k=3, seed=1)
model(train_input[0])
model.plot(beta=10)

# %%
results = model.train(dataset, opt="LBFGS", steps=50, lamb=0.001, lamb_entropy=2.);

# %%

grids = [3,5,10,20,50]

train_rmse = []
test_rmse = []

for i in range(len(grids)):
    model = KAN(width=[4,2,2,1], grid=grids[i], k=3, seed=0).initialize_from_another_model(model, dataset['train_input'])
    results = model.train(dataset, opt="LBFGS", steps=50, stop_grid_update_step=30);
    train_rmse.append(results['train_loss'][-1].item())
    test_rmse.append(results['test_loss'][-1].item())

# %%

# model.plot(in_vars=["roll", "vEgo", "aEgo", "targetLat"])
model.plot(in_vars=["roll", "vEgo"])
