import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from tinyphysics import ACC_G

def get_filenames(input_dir, ext=".csv"):
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(ext)]
    return filenames

def get_train_data(filenames, split="labelled"):
    train_data = []
    for ii, filename in tqdm(enumerate(filenames)):
        df = pd.read_csv(filename)
        _is_synthetic = True if 'actualLateralAcceleration' in df.columns else False


        roll = torch.tensor(np.sin(df['roll'].values) * ACC_G)
        v_ego = torch.tensor(df['vEgo'].values)
        a_ego = torch.tensor(df['aEgo'].values)

        if _is_synthetic:
            target_lataccel = torch.tensor(df['actualLateralAcceleration'].values)
        else: 
            target_lataccel = torch.tensor(df['targetLateralAcceleration'].values)
        target_lataccel = torch.diff(target_lataccel,dim=0) 
        target_lataccel = torch.concat([target_lataccel, target_lataccel[-1:]])

        steer_command = torch.tensor(df['steerCommand'].values).float()
        # steer_command = steer_command[1:] - steer_command[:-1]
        # steer_command = torch.concat([steer_command, steer_command[-1:]])

        input_data = torch.stack([roll, v_ego, a_ego, target_lataccel]).float().T
        # input_data = torch.stack([roll, v_ego, a_ego, target_lataccel, delta_lataccel]).float().T

        if split=="labelled":
            train_data.append((input_data[:100], steer_command[:100]))
        else:
            if (t:=input_data.shape[0])<600: 
                input_data = torch.cat([input_data,torch.zeros(600-t,4)])
                steer_command = torch.cat([steer_command,torch.zeros(600-t)])

            input_data = input_data[:600]
            steer_command = steer_command[:600]

            if split=="unlabelled":
                train_data.append((input_data[100:], steer_command[100:]))
            elif split=="all": 
                train_data.append((input_data, steer_command))
            else: 
                return NotImplementedError

    return train_data

def get_batches(train_data, batch_size, batch_first=False, combined=False):
    batches = []
    batch_dim = 0 if batch_first else 1
    for i in range(len(train_data)):
        input_data, target = [], []
        for j in range(batch_size):
            if i*batch_size+j >= len(train_data):
                break
            x, y = train_data[i*batch_size+j]
            input_data.append(x)
            target.append(y)
        if not input_data: break

        input_data = torch.stack(input_data, dim=batch_dim)
        target = torch.stack(target, dim=batch_dim)

        if not combined:
            batches.append((input_data, target))
        else: 
            batches.append(torch.concat([input_data, target.unsqueeze(-1)], dim=-1))

    return batches