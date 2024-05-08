import numpy as np
import torch
from models import MyMLP, Encoder, Decoder

class BaseController:
  def update(self, target_lataccel, current_lataccel, state, steer):
    raise NotImplementedError

class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, steer):
    return target_lataccel[-1]

class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, steer):
    return (target_lataccel[-1] - current_lataccel[-1]) * 0.3

class PIDController(BaseController):
  def __init__(self, c_p=0.3, c_i=0.05, c_d=1.0) -> None:
    super().__init__()
    self.errors = []
    self.c_p = c_p
    self.c_i = c_i
    self.c_d = c_d

  def update(self, target_lataccel, current_lataccel, state, steer):
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

  def update(self, target_lataccel, current_lataccel, state, steer):
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
    self.model = Encoder(d_input=4, d_model=128, num_layers=6)
    self.model.load_state_dict(torch.load("./models/encoder-128-6.pt"))
    self.model.eval()
    self.input_history = []

  def update(self, target_lataccel, current_lataccel, state, steer):
    roll = torch.tensor(state.roll_lataccel)
    v_ego = torch.tensor(state.v_ego)
    a_ego = torch.tensor(state.a_ego)
    target_lataccel = torch.tensor(target_lataccel)
    # delta_lataccel = torch.tensor(target_lataccel-current_lataccel)

    self.input_history.append(torch.Tensor([roll, v_ego, a_ego, target_lataccel]))
    # self.input_history.append(torch.Tensor([roll, v_ego, a_ego, target_lataccel, delta_lataccel]))
    if len(self.input_history) > 100: self.input_history.pop(0)
    input_data = torch.stack(self.input_history).float().unsqueeze(1)

    with torch.no_grad(): output_data = self.model(input_data)
    return output_data[-1].item() # *-1
  
class DecoderController(BaseController):
  def __init__(self) -> None:
    super().__init__()
    self.model = Decoder(d_input=5, d_model=128, num_layers=6, seq_len=100).cuda()
    self.model.load_state_dict(torch.load("./models/decoder-128-6.pt"))
    self.model.eval()

  def to_data(self, target_lataccel, current_lataccel, state, steer):

    state = torch.tensor(state, dtype=torch.float32)[-self.model.seq_len:] # [100, 1, 3]

    target_lataccel = torch.tensor(target_lataccel, dtype=torch.float32).view(-1,1)
    target_lataccel = target_lataccel[-self.model.seq_len:]
    current_lataccel = torch.tensor(current_lataccel, dtype=torch.float32).view(-1,1)
    current_lataccel = current_lataccel[-self.model.seq_len:]
    delta_lataccel = target_lataccel - current_lataccel # [100, 1, 1]

    steer = torch.tensor(steer, dtype=torch.float32)[-self.model.seq_len:].view(-1,1) # [100, 1, 1]

    data = torch.concat([state, delta_lataccel, steer], dim=-1).unsqueeze(1)

    return data

  def update(self, target_lataccel, current_lataccel, state, steer):
    data = self.to_data(target_lataccel, current_lataccel, state, steer)
    with torch.no_grad():
      output_data = self.model(data.cuda())
    action = torch.clip(output_data[-1,0].cpu(),steer[-1]-0.1,steer[-1]+0.1).item()
    return action

class CopyController(BaseController):
  def __init__(self):
    super().__init__()
    import pandas as pd
    df = pd.read_csv("./data/SYNTHETIC_V1/00000.csv")
    self.steer = df["steerCommand"]
    self.index = 75-1
  def update(self, target_lataccel, current_lataccel, state):
    self.index += 1
    return self.steer[self.index]
  
CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
  'mlp': MLPController,
  'enc': EncoderController,
  'dec': DecoderController,
  'copy': CopyController,
}
