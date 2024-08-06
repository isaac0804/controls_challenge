# %%
import torch.autograd
from tinyphysics import *
from utils import *
from models import Decoder

CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
filedir = "./data/SYNTHETIC_V0"

# Get synthetic data
filenames = get_filenames(filedir)[:4]
trajectories = get_train_data(filenames, split="all")
batches = get_batches(trajectories, 2, batch_first=True, combined=True)
print(batches[0].shape)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Simulation model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = LataccelTokenizer()
sim_model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

model = Decoder(d_input=6).to(device)
ref_model = Decoder(d_input=6).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), 1e-5)

def simulate_step(state, target_lataccel, current_lataccel_history, steer_history, step_idx):
    # This is where the data are concat to feed in simulation model
    state = torch.concat([
        steer_history[:,step_idx-CONTEXT_LENGTH:step_idx],
        state[:,step_idx-CONTEXT_LENGTH:step_idx]
    ], dim=-1)
    past_preds = current_lataccel_history[:,-CONTEXT_LENGTH:,-1].cpu()

    tokenized_actions = tokenizer.encode(past_preds)
    input_data = {
        'states': state.cpu().numpy(),
        'tokens': tokenized_actions
    }
    next_lataccel = tokenizer.decode(sim_model.predict(input_data, temperature=1.))
    current_lataccel = current_lataccel_history[:,-1,-1].cpu() # used for clipping
    next_lataccel = torch.clip(torch.tensor(next_lataccel, dtype=torch.float32), current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA).to(device)
    return next_lataccel.view(-1,1,1)

def predict_steer(state, target_lataccel, current_lataccel_history, steer_history, step_idx):
    input_data = torch.concat([
        state[:,step_idx-CONTEXT_LENGTH:step_idx],
        target_lataccel[:,step_idx-CONTEXT_LENGTH:step_idx],
        current_lataccel_history[:,step_idx-CONTEXT_LENGTH:step_idx],
        steer_history[:,step_idx-CONTEXT_LENGTH:step_idx]
    ], dim=-1)
    steer_pred = model(input_data.transpose(0,1).cuda()).transpose(0,1)
    return steer_pred
    # with torch.no_grad():
    #     steer_ref = ref_model(input_data.transpose(0,1).cuda()).transpose(0,1)
    # return steer_pred, steer_ref
torch.autograd.set_detect_anomaly(True)

for data in batches:
    # Process a batch at once
    data = data.cuda()
    state = data[:,:,:3]
    target_lataccel = data[:,:,3:4]
    steer_history = data[:,:CONTROL_START_IDX,-1:]

    current_lataccel_history = target_lataccel[:,:CONTROL_START_IDX]

    for step_idx in tqdm(range(CONTROL_START_IDX,600)):

        optimizer.zero_grad()
        # Prediction
        steer_pred, value = predict_steer(state, target_lataccel, current_lataccel_history, steer_history, step_idx)
        steer_history = torch.concat([steer_history, steer_pred[:,-1:]], dim=1)

        # Simulation
        with torch.no_grad():
            next_lataccel = simulate_step(state, target_lataccel, current_lataccel_history, steer_history, step_idx)
        current_lataccel_history = torch.concat([current_lataccel_history, next_lataccel], dim=1)

        # cost = F.mse_loss(next_lataccel, target_lataccel[:,step_idx]) # + F.mse_loss(steer_pred, torch.diff(steer_history[:,-2:-1]))
        # ref_cost = F.mse_loss(ref_lataccel, target_lataccel[:,step_idx]) # + F.mse_loss(steer_ref, torch.diff(steer_history[:,-2:-1]))
        # print(cost, ref_cost)
        # loss = -torch.log(F.sigmoid(0.1*torch.log(cost/ref_cost)))
        cost = torch.sum(torch.square(next_lataccel[:,-1]-current_lataccel_history[:,-2]))
        loss = cost * torch.sum(torch.square(steer_pred[:,-1:]-steer_history[:,-2]))
        print(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()
