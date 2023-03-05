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
from torchvision.transforms import Compose, Normalize
from utils.dataloader import hrrpDataset3, To4DTensor, AmplitudeNormalize, padZeros
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
dataset = hrrpDataset3(config['data']['data_path'], train=True)

for x in range(3020,3030):
    plt.plot(dataset[x][0])
    plt.show()
    plt.close()
