import torch
import pickle
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random

from trajectory_loader import TrajectoryDataLoader
from mcts import select_action
from muzeronet import MuZeroNet
from tinyphysics_env import TinyPhysicsEnv
from replay_buffer import ReplayBuffer

def train_muzero(model, data_dir, model_path, num_iterations=100, batch_size=64, learning_rate=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    gt_buffer = ReplayBuffer()
    replay_buffer = ReplayBuffer()
    random_buffer = ReplayBuffer()

    with open("gt_buffer.pickle", "rb") as f:
        gt_buffer = pickle.load(f)

    with open("replay_buffer.pickle", "rb") as f:
        replay_buffer = pickle.load(f)
    
    with open("random_replay_buffer.pickle", "rb") as f:
        random_buffer = pickle.load(f)

    gt_buffer.buffer.extend(replay_buffer.buffer)
    gt_buffer.buffer.extend(random_buffer.buffer)

    # replay_buffer = ReplayBuffer()
    # dataloader = TrajectoryDataLoader('data/')
    # for file in tqdm(dataloader.csv_files):
    #     trajectories = []
    #     df = dataloader.load_single_file(file)[:100]
    #     states = np.array([
    #         df['roll_lataccel'],
    #         df['v_ego'],
    #         df['a_ego'],
    #         df['target_lataccel'],
    #         df['target_lataccel'],
    #     ]).T
    #     for i in range(100-1):
    #         trajectories.append((states[i], df['steer_command'][i], 0, states[i+1], 0))
    #     replay_buffer.add(trajectories)
    # with open("gt_buffer.pickle", "wb") as f:
    #     pickle.dump(replay_buffer, f)

    running_loss = 0

    # Training loop
    for iteration in tqdm(range(num_iterations)):
        # Collect data
        # trajectories = collector.collect_data(model)
        # for trajectory in trajectories:

        #     replay_buffer.add(trajectory)
        
        # with open("replay_buffer.pickle", "wb") as f:
        #     pickle.dump(replay_buffer, f)
        
        # Evaluate the model periodically
        # if iteration % 10 == 0:
        #     eval_env = TinyPhysicsEnv(data_dir=data_dir, model_path=model_path)
        #     eval_reward = evaluate_model(model, eval_env)
        #     print(f"Iteration {iteration}, Eval Reward: {eval_reward}")

        batch = random.sample(replay_buffer.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # MuZero training step
        optimizer.zero_grad()

        # Initial inference
        hidden_states, policy_logits, values = model.initial_inference(states)

        # Recurrent inference
        next_hidden_states, next_reward, next_policy_logits, next_values = model.recurrent_inference(hidden_states, actions)

        # Consistency inference
        target_next_hidden_states, policy_logits, values = model.initial_inference(next_states)

        # Compute losses
        value_loss = F.mse_loss(values, (torch.log(-rewards+1e-9) + (1 - dones) * next_values)/10)
        policy_loss = F.cross_entropy(policy_logits, actions)
        reward_loss = F.mse_loss(next_reward, torch.log(-rewards+1e-9)/10)

        consistency_loss = F.mse_loss(model.predictor(model.projection(next_hidden_states)), model.projection(target_next_hidden_states).detach())

        total_loss = value_loss + policy_loss + reward_loss
        running_loss += total_loss.item()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        if iteration % 1000 == 0:
            # print(hidden_states)

            # print(next_reward)
            # print(rewards/-100)
            # print(reward_loss)

            # print(policy_logits)
            # print(actions)
            # print(policy_loss)

            # print(values)
            # print(rewards + (1-dones)*next_values)
            # print(value_loss)
            print(value_loss, policy_loss, reward_loss, consistency_loss)

            print(f"\nIteration: {iteration}  Loss: {running_loss/1000}")
            running_loss = 0

        if iteration % 10000 == 0:
            torch.save(model.state_dict(), f"models/muzero_gt_checkpoints/{iteration}.pt")

    torch.save(model.state_dict(), f"models/muzero_gt_checkpoints/{iteration+1}.pt")
    return model

# def evaluate_model(model, env, num_episodes=10):
#     total_reward = 0
#     for _ in range(num_episodes):
#         state = env.reset()
#         done = False
#         episode_reward = 0
#         while not done:
#             action = select_action(model, torch.FloatTensor(state).unsqueeze(0))
#             next_state, reward, done, _ = env.step(action)
#             episode_reward += reward
#             state = next_state
#         total_reward += episode_reward
#     return total_reward / num_episodes

# Usage example:
if __name__ == "__main__":
    muzero_model = MuZeroNet(state_dim=5, action_dim=21)
    trained_model = train_muzero(
        model=muzero_model,
        data_dir='data/',
        model_path='models/tinyphysics.onnx',
        num_iterations=100000,
        batch_size=1024,
        learning_rate=1e-3
    )