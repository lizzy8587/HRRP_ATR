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
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from utils.dataloader import hrrpDataset2, To4DTensor, AmplitudeNormalize
from utils.metrics import accurate_labels
from models import *

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

# ===================== tensorboard =====================
writer = SummaryWriter(log_dir=config['log']['log_dir'])

# ===================== dataloader =====================
train_dataset = hrrpDataset2(config['data']['train_path'], train=True, transform=Compose([AmplitudeNormalize(), To4DTensor()]))
valid_dataset = hrrpDataset2(config['data']['valid_path'], train=False, transform=Compose([AmplitudeNormalize(), To4DTensor()]))

x = train_dataset[0]

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
                                            config['model']['cls_num']
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
    writer.add_scalar('Accuracy/train',train_acc,epoch)
    writer.add_scalars('Loss/train',train_loss,epoch)


    # --------------------- valid ---------------------
    model.eval()
    with torch.no_grad():
        right_,losses_ = [],[]
        for x,y in valid_dataloader:
            output = model(x)
            loss = model.loss_function(
                *output,labels=y,
                kld_weight=config['train']['kld_weight'],
                cls_weight=config['train']['cls_weight'],
                rec_weight=config['train']['rec_weight']
            )
            right_.append(accurate_labels(output[4],y)) # [(rights,len(labels)),(),(),...]
            losses_.append((loss['loss'].item()))
        right_,num = zip(*right_)
        valid_acc = np.sum(right_)/np.sum(num)
        valid_loss = np.sum(np.multiply(losses_,num))/np.sum(num)
        writer.add_scalar('Accuracy/valid',valid_acc,epoch)
        writer.add_scalar('Loss/valid',valid_loss,epoch)
        
        
        if (config['log']['save_pkl']) & (valid_acc>best_acc):
            best_acc = valid_acc
            save_dict = model.state_dict()

    if epoch%config['log']['epochs_print']==0:  
        print('[epoch {}/{}] loss:{:.4f}, rec: {:.4f}, kld: {:.4f}, cls: {:.4f}, train_acc:{:.4f}, val_acc:{:.4f}'.format(epoch,config['train']['epochs'],train_loss['loss'],train_loss['rec_loss'],train_loss['kld'],train_loss['cls_loss'],train_acc,valid_acc))

# ===================== save model =====================
if config['log']['save_pkl']:
    save_path = os.path.join(config['log']['log_dir'], config['log']['name']+'.pkl')
    torch.save(save_dict,save_path)
    print('save: {}'.format(save_path))

# run tensorboard
# tensorboard --logdir runs