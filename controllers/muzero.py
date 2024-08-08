from . import BaseController
import numpy as np
import torch

from muzeronet import MuZeroNet
from mcts import select_action

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.model = MuZeroNet(5, 21)
    self.model.load_state_dict(torch.load("models/muzero_gt_checkpoints/100000.pt"))
    self.action_space = np.linspace(-2, 2, 21)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    input_state = torch.tensor([*state, target_lataccel, current_lataccel]).float().unsqueeze(0)

    hidden_state, policy_logits, value = self.model.initial_inference(input_state)
    return self.action_space[torch.argmax(policy_logits)]

    # action = select_action(self.model, input_state, 100)
    # return self.action_space[action]