import pandas as pd
import numpy as np
from pathlib import Path
import random

class TrajectoryDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.csv_files = list(self.data_dir.glob('*.csv'))
        # print(f"Found {len(self.csv_files)} CSV files in {self.data_dir}")

    def load_single_file(self, file_path):
        df = pd.read_csv(file_path)
        processed_df = pd.DataFrame({
            'roll_lataccel': np.sin(df['roll'].values) * 9.81,  # ACC_G = 9.81
            'v_ego': df['vEgo'].values,
            'a_ego': df['aEgo'].values,
            'target_lataccel': df['targetLateralAcceleration'].values,
            'steer_command': -df['steerCommand'].values  # Negative due to convention change
        })
        return processed_df

    def get_random_file(self):
        return random.choice(self.csv_files)

# Usage example:
# loader = TrajectoryDataLoader('data/')
# random_file = loader.get_random_file()
# data = loader.load_single_file(random_file)