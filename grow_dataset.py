# %% Inference / Generate Data
import torch
from utils import * 
import matplotlib.pyplot as plt

batch_size = 32
trajectories = get_train_data("./data/SYNTHETIC_V0", split="all", num_files=20000)
inference_data = get_batches(trajectories, batch_size)
context_length = 100
overlap_width = 50
mask = torch.concat([torch.linspace(0,1,overlap_width), torch.ones(context_length-2*overlap_width), torch.linspace(1,0,overlap_width)]).unsqueeze(1).cuda()
first_mask = torch.concat([torch.ones(context_length-overlap_width), torch.linspace(1,0,overlap_width)]).unsqueeze(1).cuda()
end_mask = torch.concat([torch.linspace(0,1,overlap_width),torch.ones(context_length-overlap_width)]).unsqueeze(1).cuda()
generated_data = []

# %%
from models import Encoder

model = Encoder(d_input=4, d_model=128, num_layers=6).cuda()
model.load_state_dict(torch.load("./models/encoder-128-6.pt"))
model.eval()

# TODO: there should be smarter masking method

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
print(len(generated_data))
generated_data = torch.concat(generated_data, axis=1)
print(generated_data.shape)

# %%

generated_data_ = generated_data.cpu().T
filenames = [os.path.join("./data/SYNTHETIC_V0", f) for f in os.listdir("./data/SYNTHETIC_V0")]

for ii, filename in tqdm(enumerate(filenames)):
    df = pd.read_csv(filename)
    k = min(len(df), 600)
    df["steerCommand"][:k] = generated_data_[ii][:k].numpy()
    df.to_csv(filename.replace("SYNTHETIC_V0","SYNTHETIC_V2"))


# %% Train controller model

plt.plot(generated_data_[0]+torch.randn_like(generated_data_[0])*0.005)
plt.show()