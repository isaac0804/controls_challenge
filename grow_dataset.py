# %% Inference / Generate Data
import torch
from utils import * 
import matplotlib.pyplot as plt

BATCH_SIZE = 512
CONTEXT_LENGTH = 100
OVERLAP_WIDTH = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

filenames = get_filenames("./data/SYNTHETIC_V0/")[:100]
trajectories = get_train_data(filenames, split="all")
inference_data = get_batches(trajectories, BATCH_SIZE)
mask = torch.concat([torch.linspace(0,1,OVERLAP_WIDTH), torch.ones(CONTEXT_LENGTH-2*OVERLAP_WIDTH), torch.linspace(1,0,OVERLAP_WIDTH)]).unsqueeze(1).cuda()
first_mask = torch.concat([torch.ones(CONTEXT_LENGTH-OVERLAP_WIDTH), torch.linspace(1,0,OVERLAP_WIDTH)]).unsqueeze(1).cuda()
end_mask = torch.concat([torch.linspace(0,1,OVERLAP_WIDTH),torch.ones(CONTEXT_LENGTH-OVERLAP_WIDTH)]).unsqueeze(1).cuda()

# %%
from models import Encoder

model = Encoder(d_input=4, d_model=64, num_layers=4).cuda()

# TODO: there should be smarter masking method

model.load_state_dict(torch.load(f"./models/encoder-64-4-seed-{k}.pt"))
model.eval()
generated_data = []
steer_temp = []
with torch.no_grad():
    for input_data, target in tqdm(inference_data):
        input_data = input_data.cuda()
        target = target[:100].cuda()
        outputs = torch.concat([first_mask*target, torch.zeros(500, target.shape[1]).cuda()])

        for index in range(50,500+1,50):
            data = input_data[index:min(index+100,input_data.shape[0])]
            output = model(data)
            if index != 500:
                outputs[index:min(index+100,input_data.shape[0])] += mask*output.squeeze()
            else:
                outputs[index:min(index+100,input_data.shape[0])] += end_mask*output.squeeze()
        generated_data.append(outputs)

generated_data = torch.concat(generated_data, axis=1)
generated_data_ = generated_data.cpu().T
filenames = [os.path.join("./data/SYNTHETIC_V0", f) for f in os.listdir("./data/SYNTHETIC_V0")]

for ii, filename in tqdm(enumerate(filenames)):
    df = pd.read_csv(filename)
    m = min(len(df), 600)
    df["steerCommand"][:m] = generated_data_[ii][:m].numpy()
    os.makedirs(f"./data/SYNTHETIC_V{k+4}", exist_ok=True)
    df.to_csv(filename.replace("SYNTHETIC_V0",f"SYNTHETIC_V{k+4}"))

# %%
from models import Decoder

model = Decoder(d_input=5, d_model=128, num_layers=6).cuda()
model.load_state_dict(torch.load(f"./models/decoder-128-6-seed-{k}.pt"))
model.eval()
generated_data = []

with torch.no_grad():
    for data, steer in inference_data:
        # Process a batch at once
        steer_history = []

        for step_idx in tqdm(range(75,600)):
            # For each batch, run through all the time steps autoregressively

            # Prepare data for each sample in batch
            # Shape : [B, N=CONTEXT_LENGTH, D]
            state = torch.concat([
                data[step_idx-CONTEXT_LENGTH:step_idx,:,:-1],
                steer[step_idx-CONTEXT_LENGTH:step_idx,:,None],
            ], dim=-1).to(device)

            output = model(state)
            generated_data.append(outputs.permute())
