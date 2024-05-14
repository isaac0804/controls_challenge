import numpy as np
import torch
from models import Encoder, Decoder

class BaseController:
  def update(self, target_lataccel, current_lataccel, state, steer):
    raise NotImplementedError

class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, steer):
    return target_lataccel[-1]

class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, steer):
    return (target_lataccel[-1] - current_lataccel[-1]) * 0.3

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

class DecoderPIDController(BaseController):
  def __init__(self) -> None:
    super().__init__()
    self.model = Decoder(d_input=5, d_model=128, num_layers=6, seq_len=100).cuda()
    self.model.load_state_dict(torch.load("./models/decoder-128-6.pt"))
    self.model.eval()
    self.error_history = []

  def to_data(self, target_lataccel, current_lataccel, state, steer):

    K_p = 0.03
    K_i = 0.08
    K_d = 0.01
    self.error_history.append(target_lataccel[-1]-current_lataccel[-1])
    if len(self.error_history) > 10: self.error_history = self.error_history[-10:]
    P = self.error_history[-1]
    I = np.mean(self.error_history)
    D = self.error_history[-1] - self.error_history[-2] if len(self.error_history)>1 else 0
    self.correction = K_p * P + K_i * I + K_d * D

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
    action = torch.clip(output_data[-1,0].cpu(),steer[-1]-0.1,steer[-1]+0.1).item() + self.correction
    return action

CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'enc': EncoderController,
  'dec': DecoderController,
  'decpid': DecoderPIDController
}
