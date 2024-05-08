import torch
import torch.nn as nn
import torch.optim as optim

a = torch.linspace(0,1,100).view(-1,100,1).cuda()
b = (torch.sin(7*a)+torch.sin(10*a)+torch.sin(13*a)).cuda()

from models import Decoder

model = Decoder(d_input=1, d_model=64, num_layers=2, seq_len=100).cuda()
model.train()
criterion = nn.MSELoss()

EPOCHS = 100
lr = 1e-3
lr_min = 1e-6
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,lr_min)

for _ in range(EPOCHS):
    b_ = model(a) #+0.001*torch.randn_like(a)
    loss = criterion(b_, b)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    print(loss.item())

import matplotlib.pyplot as plt

model.eval()
print(a.shape)
print(model(a).shape)
print(b.shape)
# plt.plot(a.detach().cpu()[0,:])
plt.plot(model(a).detach().cpu()[0,:])
plt.plot(b.detach().cpu()[0,:])
plt.legend(["predicted", "gt"])
plt.show()