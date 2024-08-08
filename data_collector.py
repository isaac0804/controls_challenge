import multiprocessing as mp
from tqdm import tqdm
from mcts import select_action
from muzeronet import MuZeroNet
from tinyphysics_env import TinyPhysicsEnv
import torch
import random

class ParallelDataCollector:
    def __init__(self, env_class, env_args, num_processes, episodes_per_process):
        self.env_class = env_class
        self.env_args = env_args
        self.num_processes = num_processes
        self.episodes_per_process = episodes_per_process

    def collect_data(self, model):
        with mp.Pool(self.num_processes) as pool:
            results = list(tqdm(pool.imap(
                self._collect_episodes,
                [(model, self.env_class, self.env_args, self.episodes_per_process) for _ in range(self.num_processes)]
            ), total=self.num_processes))

        all_trajectories = []
        for process_trajectories in results:
            all_trajectories.extend(process_trajectories)

        return all_trajectories

    @staticmethod
    def _collect_episodes(args):
        model, env_class, env_args, num_episodes = args
        env = env_class(**env_args)
        trajectories = []

        for _ in range(num_episodes):
            state = env.reset()
            trajectory = []
            done = False
            # i = 0

            while not done:
                action = ParallelDataCollector._select_action(model, torch.FloatTensor(state).unsqueeze(0))
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward, next_state, done))
                state = next_state
                # print(i)
                # i += 1

            trajectories.append(trajectory)

        return trajectories

    @staticmethod
    def _select_action(model, state):
        # print(state)
        # return select_action(model, state)
        return random.randint(0,20)

if __name__ == "__main__":
    model = MuZeroNet(5, 21)
    collector = ParallelDataCollector(
        env_class=TinyPhysicsEnv,
        env_args={'data_dir': 'data/', 'model_path': 'models/tinyphysics.onnx'},
        num_processes=10,
        episodes_per_process=1
    )
    trajectories = collector.collect_data(model)