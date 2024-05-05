import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from tinyphysics import ACC_G


def get_train_data(input_dir, split="labelled", num=None):
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    if not num: num = len(filenames)
    train_data = []

    for ii, filename in tqdm(enumerate(filenames[:num])):
        df = pd.read_csv(filename)

        roll = torch.tensor(np.sin(df['roll'].values) * ACC_G)
        v_ego = torch.tensor(df['vEgo'].values)
        a_ego = torch.tensor(df['aEgo'].values)
        target_lataccel = torch.tensor(df['targetLateralAcceleration'].values)
        target_lataccel = target_lataccel[1:] - target_lataccel[:-1]
        target_lataccel = torch.concat([target_lataccel, torch.zeros(1)])
        # delta_lataccel = target_lataccel[1:] - target_lataccel[:-1]
        # delta_lataccel = torch.concat([delta_lataccel, torch.zeros(1)])
        steer_command = torch.tensor(df['steerCommand'].values).float()
        input_data = torch.stack([roll, v_ego, a_ego, target_lataccel]).float().T
        # input_data = torch.stack([roll, v_ego, a_ego, target_lataccel, delta_lataccel]).float().T

        if split=="labelled":
            train_data.append((input_data[:100], steer_command[:100]))
        else:
            if (t:=input_data.shape[0])<600: 
                input_data = torch.cat([input_data,torch.zeros(600-t,4)])
                steer_command = torch.cat([steer_command,torch.zeros(600-t)])
            if input_data.shape[0]>600: 
                input_data = input_data[:600]
                steer_command = steer_command[:600]

            if split=="unlabelled":
                train_data.append((input_data[100:], steer_command[:100]))
            elif split=="all": 
                train_data.append((input_data, steer_command))
            else: 
                return NotImplementedError

    return train_data

def create_batches(train_data, batch_size):
    batches = []
    for i in range(len(train_data)):
        input_data, target = [], []
        for j in range(batch_size):
            if i*batch_size+j >= len(train_data):
                break
            x, y = train_data[i*batch_size+j]
            input_data.append(x)
            target.append(y)
        if not input_data: break
        input_data = torch.stack(input_data, dim=1)
        target = torch.stack(target, dim=1)
        batches.append((input_data, target))
    return batches