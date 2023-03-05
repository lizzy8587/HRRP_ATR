#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   run.py
@Version      :   1.0
@Time         :   2023/01/27 20:20:52
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   None
'''
import os  
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose
from utils.dataloader import hrrpDataset3, To3DTensor, AmplitudeNormalize,padZeros
from utils.metrics import accurate_labels
from models import *
import matplotlib.pyplot as plt

# ===================== config params =====================
parser = argparse.ArgumentParser(description='Generic runner for HRRP datasets')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/e2eVAE.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# ===================== dataloader =====================
dataset = hrrpDataset3(config['data']['data_path'], train=True, length=config['data']['length'], transform=Compose([AmplitudeNormalize(), To3DTensor()]))
train_num = int(len(dataset)*config['data']['train_ratio'])
test_num = len(dataset)-train_num
train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[train_num,test_num])
# train_dataset = hrrpDataset3(config['data']['train_path'], train=True, transform=Compose([padZeros(256), AmplitudeNormalize(), To3DTensor()]))
print(train_dataset[0][0].shape)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=config['data']['shuffle']
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=config['data']['shuffle']
)
# ===================== model train =====================
model = hrrp_models[config['model']['name']](
                                            config['model']['in_channels'],
                                            config['model']['latent_dim'],
                                            config['model']['cls_num'],
                                            length=config['data']['length'],
                                            hidden_dims=[32, 64, 128]
                                        )
if config['model']['weights']:  # load pretrained model
    model.load_state_dict(torch.load(config['model']['weights']))

optimizer = torch.optim.Adam(model.parameters(),lr=config['train']['learning_rate'])

best_acc = 0    # a metric to decide wether to save the model
save_dict = model.state_dict()
for epoch in range(config['train']['epochs']):
    # --------------------- train ---------------------
    model.train()
    right_,losses_ = [],[]
    for x,y in train_dataloader:
        output = model(x)
        loss = model.loss_function(
            *output,labels=y,
            kld_weight=config['train']['kld_weight'],
            cls_weight=config['train']['cls_weight'],
            rec_weight=config['train']['rec_weight']
        )
        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        
        right_.append(accurate_labels(output[4],y)) # [(rights,len(labels)),(),(),...]
        losses_.append([k.item() for k in loss.values()])

    losses_ = np.array(losses_)
    right_,num = zip(*right_)
    train_acc = np.sum(right_)/np.sum(num)
    train_loss = dict(zip(
        loss.keys(),
        np.sum(np.multiply(losses_.transpose(),num),axis=1)/np.sum(num)  # [loss,rec_loss,kld,cls_loss]
    ))


    if epoch%config['log']['epochs_print']==0:  
        print('[epoch {}/{}] loss:{:.4f}, rec: {:.4f}, kld: {:.4f}, cls: {:.4f}, train_acc:{:.4f}'.format(epoch,config['train']['epochs'],train_loss['loss'],train_loss['rec_loss'],train_loss['kld'],train_loss['cls_loss'],train_acc))

x1 = model.generate(x)

plt.figure(figsize=(8, 4))
plt.tight_layout(pad=2)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.plot(x[i].view(-1), label='raw')
    plt.plot(x1[i].view(-1).detach(), label='new')
    plt.legend()
    plt.title('{}'.format(y[i]))
plt.show()
