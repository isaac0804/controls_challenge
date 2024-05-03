import numpy as np
import torch
from models import MLP

class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError

class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel

class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3

class PIDController(BaseController):
  def __init__(self, c_p=0.3, c_i=0.05, c_d=1.0) -> None:
    super().__init__()
    # self.moving_window_target_lataccel = []
    # self.moving_window_current_lataccel = []
    self.errors = []
    self.c_p = c_p
    self.c_i = c_i
    self.c_d = c_d

  def update(self, target_lataccel, current_lataccel, state):
    # Update target lat accel
    # if len(self.moving_window_target_lataccel) == 10:
    #   self.moving_window_target_lataccel.pop(0)
    # self.moving_window_target_lataccel.append(target_lataccel)
    # Update current lat accel
    # if len(self.moving_window_current_lataccel) == 10:
    #   self.moving_window_current_lataccel.pop(0)
    # self.moving_window_current_lataccel.append(current_lataccel)
    # Update errors
    if len(self.errors) == 10:
      self.errors.pop(0)
    self.errors.append(target_lataccel-current_lataccel)

    # Calculate I
    I = np.sum(self.errors)
    if len(self.errors) < 3:
      D = 0
    else:
      D = np.mean(np.subtract(self.errors[:-1],self.errors[1:]))

    P = target_lataccel - current_lataccel
    # D = np.mean(self.moving_window_target_lataccel) - np.mean(self.moving_window_current_lataccel)

    return P * self.c_p + I * self.c_i + D * self.c_d
  
class MLPController(BaseController):
  def __init__(self) -> None:
    super().__init__()
    self.model = MLP(4, 16, 1)
    self.model.load_state_dict(torch.load("./models/mlp.pt"))

  def update(self, target_lataccel, current_lataccel, state):
    roll = torch.tensor(state.roll_lataccel)
    v_ego = torch.tensor(state.roll_lataccel)
    a_ego = torch.tensor(state.roll_lataccel)
    target_lataccel = torch.tensor(target_lataccel)
    input_data = torch.stack([roll, v_ego, a_ego, target_lataccel]).float()
    with torch.no_grad(): output_data = self.model(input_data)
    return output_data.numpy()[0]*-1


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
  'mlp': MLPController,
}
