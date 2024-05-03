# %%
from tinyphysics import ACC_G
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os

input_dir = "./data/"
filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

train_data = []
test_data = []

for ii, filename in tqdm(enumerate(filenames)):
    df = pd.read_csv(filename)[:100]

    roll = torch.tensor(np.sin(df['roll'].values) * ACC_G)
    v_ego = torch.tensor(df['vEgo'].values)
    a_ego = torch.tensor(df['aEgo'].values)
    target_lataccel = torch.tensor(df['targetLateralAcceleration'].values)
    steer_command = torch.tensor(df['steerCommand'].values).float()
    input_data = torch.stack([roll, v_ego, a_ego, target_lataccel]).float().T

    if ii < 1000:
        test_data.append((input_data, steer_command))
    else:
        train_data.append((input_data, steer_command))

def create_dataset(data):
    x, y = train_data[ii%len(train_data)]
    input_data.append(x)
    target.append(y)

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
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc3(torch.tanh(self.fc2(x))) + x
        x = self.fc4(x)
        return x

# Initialize the model, loss function, and optimizer
model = MLP(input_dim=4, hidden_dim=16, output_dim=1)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


# %%
training_steps = 40000
batch_size = 32
losses = []

for ii in range(training_steps+1):
    input_data, target = [], []

    for _ in range(batch_size):
        x, y = train_data[ii%len(train_data)]
        input_data.append(x)
        target.append(y)

    input_data = torch.stack(input_data)
    target = torch.stack(target)

    # Forward pass
    output = model(input_data)
    loss = criterion(output.squeeze(), target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if ii % 1000 == 0:
        print(f"Step: {ii}  Loss: {np.mean(losses[-1000:])}")

torch.save(model.state_dict(), "./models/mlp.pt")