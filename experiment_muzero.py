import torch
import pickle
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random

from data_collector import ParallelDataCollector
from mcts import select_action
from muzeronet import MuZeroNet
from tinyphysics_env import TinyPhysicsEnv

if __name__ =="__main__":
    VISUALIZE = True

    model = MuZeroNet(state_dim=5, action_dim=21)
    model.load_state_dict(torch.load("models/muzero_checkpoints/60.pt"))

    env = TinyPhysicsEnv(data_dir='data/', model_path='models/tinyphysics.onnx')

    total_reward = 0
    for _ in range(200):
        state = env.reset(debug=VISUALIZE)
        done = False
        episode_reward = 0
        while not done:
            action = select_action(model, torch.FloatTensor(state).unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            print(action, reward, state)
        total_reward += episode_reward