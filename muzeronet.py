import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZeroNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.discount = 0.99
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Representation function (s_t => h_t)
        self.representation = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Dynamics function (h_t, a_t => h_{t+1}, r_{t+1})
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim+1),
            nn.GELU()
        )

        # Prediction function (h_t => p_t, v_t)
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim + 1)  # Action logits + value
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def initial_inference(self, state):
        hidden_state = self.representation(state)
        policy_logits, value = self.prediction(hidden_state).split([self.action_dim, 1], dim=1)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action):
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        # print(hidden_state.shape, action_onehot.shape)
        next_hidden_state, next_reward = self.dynamics(torch.cat([hidden_state, action_onehot], dim=1)).split([self.hidden_dim, 1], dim=1)
        policy_logits, value = self.prediction(next_hidden_state).split([self.action_dim, 1], dim=1)
        return next_hidden_state, next_reward, policy_logits, value

# Usage example:
# state_dim = 5  # roll_lataccel, v_ego, a_ego, target_lataccel, current_lataccel
# action_dim = 21  # Discretized steering actions from -2 to 2
# model = MuZeroNet(state_dim, action_dim)
# print(model)