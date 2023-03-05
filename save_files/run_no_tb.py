#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   run_no_tb.py
@Version      :   1.0
@Time         :   2023/01/27 20:21:27
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   None
'''

import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.dataloader import hrrpDataset
from utils.metrics import accurate_labels
from models import *

# ===================== config params =====================
parser = argparse.ArgumentParser(description='Generic runner for HRRP datasets')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/oneD_CNN.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# ===================== dataloader =====================
train_dataset = hrrpDataset(config['data']['train_path'], train=True)
test_dataset = hrrpDataset(config['data']['test_path'], train=False)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=config['data']['shuffle']
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=config['data']['shuffle']
)

# ===================== model train =====================
model = hrrp_models[config['model']['name']]()
loss_func = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(),lr=config['train']['learning_rate'])

train_acc = []
val_acc = []
losses = []
for epoch in range(config['train']['epochs']):
    # --------------------- train ---------------------
    model.train()
    right_,losses_ = [],[]
    for x,y in train_dataloader:
        predict_y = model(x)
        loss = loss_func(predict_y,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        right_.append(accurate_labels(predict_y,y)) # [(rights,len(labels)),(),(),...]
        losses_.append((loss.item()))

    right_,num = zip(*right_)
    train_acc.append(np.sum(right_)/np.sum(num))
    losses.append(np.sum(np.multiply(losses_,num))/np.sum(num))

    # --------------------- valid ---------------------
    model.eval()
    with torch.no_grad():
        right_,num = zip(*[accurate_labels(model(x),y) for x,y in test_dataloader])
        val_acc.append(np.sum(right_)/np.sum(num))
    
    if epoch%config['log']['epochs_log']==0:  
        print('[epoch {}/{}] loss:{}, train_acc:{}, val_acc:{}'.format(epoch,config['train']['epochs'],losses[epoch],train_acc[epoch],val_acc[epoch]))