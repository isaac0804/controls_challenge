import pickle
from tqdm import tqdm

from data_collector import ParallelDataCollector
from groundtruth_training_muzero import ReplayBuffer
from muzeronet import MuZeroNet
from tinyphysics_env import TinyPhysicsEnv

if __name__ == "__main__":
    model = MuZeroNet(5, 21)
    collector = ParallelDataCollector(
        env_class=TinyPhysicsEnv,
        env_args={'data_dir': 'data/', 'model_path': 'models/tinyphysics.onnx'},
        num_processes=10,
        episodes_per_process=1
    )
    replay_buffer = ReplayBuffer()
    with open("random_replay_buffer.pickle", "rb") as f:
        replay_buffer = pickle.load(f)

    for i in tqdm(range(1000)):
        trajectories = collector.collect_data(model)

        for trajectory in trajectories:
            replay_buffer.add(trajectory)

        if i % 10 == 0:
            with open("random_replay_buffer.pickle", "wb") as f:
                pickle.dump(replay_buffer, f)
            