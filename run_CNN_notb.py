#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   run.py
@Version      :   1.0
@Time         :   2023/01/27 20:20:52
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   1D-CNN
'''
import os
import sys
import datetime
import yaml
import argparse
import numpy as np
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from utils.dataloader import hrrpDataset2, hrrpDataset3, To3DTensor, AmplitudeNormalize, padZeros
from utils.metrics import accurate_labels
from utils.tools import Logger
from models import *

# ===================== config params =====================
parser = argparse.ArgumentParser(description='Generic runner for HRRP datasets')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/oneD_CNN.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# ===================== logger =====================
dateTime = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = '{}_{}/'.format(config['log']['log_dir'], dateTime)  # 按时间生成目录，存放log文件和训练模型文件
sys.stdout = Logger(fpath=os.path.join(log_dir, 'log.txt'))

print(config)

# ===================== dataloader =====================
dataset = hrrpDataset3(config['data']['data_path'], length=config['data']['length'], transform=Compose([AmplitudeNormalize(), To3DTensor()]))
train_num = int(len(dataset)*config['data']['train_ratio'])
test_num = len(dataset)-train_num
train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[train_num,test_num])
# train_dataset = hrrpDataset2(config['data']['train_path'], train=True, transform=Compose([padZeros(256) ,AmplitudeNormalize(), To4DTensor()]))
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
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = hrrp_models[config['model']['name']](
#                                             config['model']['in_channels'],
#                                             config['data']['cls_num'],
#                                             length=config['data']['length'],
#                                             hidden_dims=[32, 64, 128], #[4, 8, 16]
#                                         dropout_ratio=config['train']['dropout_ratio']
#                                         )

model = hrrp_models[config['model']['name']](
                                            ResidualBlock,
                                            [3,4,6,3],
                                            config['data']['cls_num']
                                        )


# model = hrrp_models[config['model']['name']]()
# model.to(device)
if config['model']['weights']:  # load pretrained model
    model.load_state_dict(torch.load(config['model']['weights']))

# loss_func = F.cross_entropy
loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=config['train']['learning_rate'])
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config['train']['learning_rate'],
    momentum=config['train']['momentum'],
    weight_decay=config['train']['weight_decay']
)

lr_scheduler = None
if config['train']['lr_decay']:
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config['train']['lr_step'],
        gamma=config['train']['lr_decay']
    )

best_acc = 0    # a metric to decide whether to save the model
save_dict = model.state_dict()
for epoch in range(config['train']['epochs']):
    # --------------------- train ---------------------
    model.train()
    right_,losses_ = [],[]
    for x,y in train_dataloader:
        # x=x.to(device)
        # y=y.to(device)
        
        optimizer.zero_grad()
        predict_y = model(x)
        loss = loss_func(predict_y,y)
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        right_.append(accurate_labels(predict_y,y)) # [(rights,len(labels)),(),(),...]
        losses_.append((loss.item()))

    right_,num = zip(*right_)
    train_acc = np.sum(right_)/np.sum(num)
    train_loss = np.sum(np.multiply(losses_,num))/np.sum(num)

    # --------------------- valid ---------------------
    model.eval()
    with torch.no_grad():
        right_,losses_ = [],[]
        for x,y in valid_dataloader:
            # x = x.to(device)
            # y = y.to(device)
            predict_y = model(x)
            loss = loss_func(predict_y,y)
            right_.append(accurate_labels(predict_y,y)) # [(rights,len(labels)),(),(),...]
            losses_.append((loss.item()))
        right_,num = zip(*right_)
        valid_acc = np.sum(right_)/np.sum(num)
        valid_loss = np.sum(np.multiply(losses_,num))/np.sum(num)

        if (config['log']['save_pkl']) & (valid_acc>best_acc):
            best_acc = valid_acc
            save_dict = model.state_dict()

    if epoch%config['log']['epochs_print']==0:
        print('[epoch {}/{}] loss:{:.4f}, train_acc:{:.4f}, val_acc:{:.4f}'.format(epoch,config['train']['epochs'],train_loss,train_acc,valid_acc))

# ===================== save model =====================
if config['log']['save_pkl']:
    save_path = os.path.join(log_dir, config['log']['name']+'.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print('save: {}'.format(save_path))
