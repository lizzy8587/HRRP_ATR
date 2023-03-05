# %% [markdown]
# # HRRP-ATR-VAE

# %% [markdown]
# ## import

# %%
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.dataloader import *
from models import VanillaVAE,E2EVAE

# %%
train_path = r'datasets\Train_hrrp.mat'
test_path = r'datasets\Test_hrrp.mat'
batch_size = 32
epochs = 30
epochs_log = epochs/5
learning_rate = 0.001

# %% [markdown]
# ## load data

# %% [markdown]
# 一定要幅度归一化！！！loss明显降低！！！

# %%
from torchvision.transforms import Compose
train_ds = hrrpDataset(train_path, train=True, transform=Compose([AmplitudeNormalize(), To4DTensor()]))
train_dl = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True)
for labels in train_dl:
    print('labels:',labels[1])
    break

# %% [markdown]
# ## model

# %%
# model = VanillaVAE(1,256)
model = E2EVAE(1,256)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    # --------------------- train ---------------------
    model.train()
    for x,y in train_dl:
        output = model(x)
        loss = model.loss_function(*output,M_N=0.1,labels=y)['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("[{}/{}] loss: {}".format(epoch,epochs,loss.item()))

# %%
x1 = model.generate(x)
                                                                                        
plt.figure(figsize=(8,4))
plt.tight_layout(pad=2)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.plot(x[i].view(-1),label='raw')
    plt.plot(x1[i].view(-1).detach(),label='new')
    plt.legend()
    plt.title('{}'.format(y[i]))


