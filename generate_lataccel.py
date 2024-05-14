# %%
from tinyphysics import *
from utils import *

CONTROL_START_IDX = 75
filedir = "./data/SYNTHETIC_V3"

# Get synthetic data
filenames = get_filenames(filedir)[:20000]
trajectories = get_train_data(filenames, split="all")

# %%

batches = get_batches(trajectories, 2000, batch_first=True)

# Simulation model
tokenizer = LataccelTokenizer()
sim_model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

cost = []
generated = []

for data, steer in batches:
    # Process a batch at once
    current_lataccel_history = data[:,:CONTROL_START_IDX,-1:]
    for step_idx in tqdm(range(CONTROL_START_IDX,600)):
        # For each batch, run through all the time steps autoregressively

        # Prepare data for each sample in batch
        # Shape should be [B, N=CONTEXT_LENGTH, D]
        state = torch.concat([
            steer[:,step_idx-CONTEXT_LENGTH:step_idx,None],
            data[:,step_idx-CONTEXT_LENGTH:step_idx,:-1]
        ], dim=-1)

        current_lataccel = current_lataccel_history[:,-1,-1]
        past_preds = current_lataccel_history[:,-CONTEXT_LENGTH:,-1]
        tokenized_actions = tokenizer.encode(past_preds)

        # Log
        # print(f"input_data.states.shape: {state.shape}")
        # print(f"past_preds.shape: {past_preds.shape}")
        # print(f"input_data.tokens.shape: {tokenized_actions.shape}")

        input_data = {
            'states': state.numpy(),
            'tokens': tokenized_actions
        }
        pred = tokenizer.decode(sim_model.predict(input_data, temperature=1.))
        pred = torch.clip(torch.tensor(pred), current_lataccel - MAX_ACC_DELTA, current_lataccel+ MAX_ACC_DELTA)

        current_lataccel_history = torch.concat([current_lataccel_history, pred.view(-1,1,1)], dim=1)
    
    target = data[:,CONTROL_START_IDX:,-1].numpy()
    predicted = current_lataccel_history[:,CONTROL_START_IDX:,-1].numpy()

    lat_accel_cost = np.mean((target - predicted)**2, axis=1) * 100
    jerk_cost = np.mean((np.diff(predicted) / DEL_T)**2, axis=1) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    cost.extend(total_cost.tolist())

    generated.append(current_lataccel_history)

with open(f"{filedir}/cost.txt", "w") as f:
    f.writelines([str(c)+"\n" for c in cost])

generated = torch.concat(generated, dim=0)

for ii, filename in enumerate(filenames):
    df = pd.read_csv(filename)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if len(df)>600: df = df.iloc[:600]
    df['actualLateralAcceleration'] = generated[ii][:len(df)]
    df.to_csv(filename, index=False)

