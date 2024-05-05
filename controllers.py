import numpy as np
import torch
from models import MyMLP, Encoder

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
    self.errors = []
    self.c_p = c_p
    self.c_i = c_i
    self.c_d = c_d

  def update(self, target_lataccel, current_lataccel, state):
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

    return P * self.c_p + I * self.c_i + D * self.c_d
  
class MLPController(BaseController):
  def __init__(self) -> None:
    super().__init__()
    self.model = MyMLP(4, 64, 1)
    self.model.load_state_dict(torch.load("./models/mlp.pt"))
    self.model.eval()

  def update(self, target_lataccel, current_lataccel, state):
    roll = torch.tensor(state.roll_lataccel)
    v_ego = torch.tensor(state.roll_lataccel)
    a_ego = torch.tensor(state.roll_lataccel)
    target_lataccel = torch.tensor(target_lataccel)
    input_data = torch.stack([roll, v_ego, a_ego, target_lataccel]).float()
    with torch.no_grad(): output_data = self.model(input_data)
    return output_data.numpy()[0]

class EncoderController(BaseController):
  def __init__(self) -> None:
    super().__init__()
    self.model = Encoder(d_input=4, d_model=128, num_layers=4)
    self.model.load_state_dict(torch.load("./models/encoder.pt"))
    self.model.eval()
    self.input_history = []

  def update(self, target_lataccel, current_lataccel, state):
    roll = torch.tensor(state.roll_lataccel)
    v_ego = torch.tensor(state.roll_lataccel)
    a_ego = torch.tensor(state.roll_lataccel)
    target_lataccel = torch.tensor(target_lataccel)
    # delta_lataccel = torch.tensor(target_lataccel-current_lataccel)

    self.input_history.append(torch.Tensor([roll, v_ego, a_ego, target_lataccel]))
    # self.input_history.append(torch.Tensor([roll, v_ego, a_ego, target_lataccel, delta_lataccel]))
    if len(self.input_history) > 100: self.input_history.pop(0)
    input_data = torch.stack(self.input_history).float().unsqueeze(1)

    with torch.no_grad(): output_data = self.model(input_data)
    return output_data[-1].item() # *-1


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
  'mlp': MLPController,
  'enc': EncoderController,
}
