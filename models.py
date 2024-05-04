import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc3(torch.tanh(self.fc2(x))) + x
        x = self.dp1(x)
        x = self.fc5(torch.tanh(self.fc4(x))) + x
        x = self.dp2(x)
        x = self.fc6(x)
        return x