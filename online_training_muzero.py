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
from replay_buffer import ReplayBuffer

def train_muzero(model, data_dir, model_path, num_iterations=100, batch_size=64, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer()
    collector = ParallelDataCollector(
        env_class=TinyPhysicsEnv,
        env_args={'data_dir': data_dir, 'model_path': model_path},
        num_processes=11,  # Adjust based on your system's capabilities
        episodes_per_process=1  # Adjust as needed
    )

    for iteration in tqdm(range(num_iterations)):
        # Collect data
        trajectories = collector.collect_data(model)
        for trajectory in trajectories:
            replay_buffer.add(trajectory)
        
        with open("replay_buffer.pickle", "wb") as f:
            pickle.dump(replay_buffer, f)
        
        # Evaluate the model periodically
        if iteration % 10 == 0:
            eval_env = TinyPhysicsEnv(data_dir=data_dir, model_path=model_path)
            eval_reward = evaluate_model(model, eval_env)
            print(f"Iteration {iteration}, Eval Reward: {eval_reward}")

        # Training loop
        for _ in tqdm(range(500)):  # Adjust the number of training steps per iteration
            if len(replay_buffer) < batch_size:
                break

            batch = random.sample(replay_buffer.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # MuZero training step
            optimizer.zero_grad()

            # Initial inference
            hidden_states, policy_logits, values = model.initial_inference(states)

            # Recurrent inference
            next_hidden_states, next_reward, next_policy_logits, next_values = model.recurrent_inference(hidden_states, actions)

            # Compute losses
            value_loss = F.mse_loss(values, rewards + (1 - dones) * next_values)
            policy_loss = F.cross_entropy(policy_logits, actions)
            reward_loss = F.mse_loss(model.dynamics(torch.cat([hidden_states, F.one_hot(actions, num_classes=model.action_dim).float()], dim=1))[:, 0], rewards)

            total_loss = value_loss + policy_loss + reward_loss
            total_loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"models/muzero_checkpoints/{iteration}.pt")

    return model

def evaluate_model(model, env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = select_action(model, torch.FloatTensor(state).unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    return total_reward / num_episodes

# Usage example:
if __name__ == "__main__":
    muzero_model = MuZeroNet(state_dim=5, action_dim=21)
    trained_model = train_muzero(
        model=muzero_model,
        data_dir='data/',
        model_path='models/tinyphysics.onnx',
        num_iterations=1000,
        batch_size=64,
        learning_rate=1e-4
    )