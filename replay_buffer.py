from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self, max_size=10_000_000):
        self.max_size = max_size
        self.buffer = []

    def add(self, trajectory):
        if len(self.buffer) + len(trajectory) > self.max_size:
            self.buffer = self.buffer[len(trajectory):]
        self.buffer.extend(trajectory)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

class TrajectoryBuffer(Dataset):
    def __init__(self, max_size=10_000_000):
        self.max_size = max_size
        self.buffer = []

    def add(self, trajectory):
        if len(self.buffer) + len(trajectory) > self.max_size:
            self.buffer = self.buffer[len(trajectory):]
        self.buffer.extend(trajectory)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]