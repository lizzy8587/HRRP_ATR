#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   __init__.py
@Version      :   1.0
@Time         :   2023/01/27 20:21:12
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   None
'''

from .oneD_CNN import *
from .VAE import *
from .E2EVAE import *
from .ResNet import *

hrrp_models = {
    'oneD_CNN': oneD_CNN,
    'ResNet': ResNet,
    'VAE': VanillaVAE,
    'E2EVAE': E2EVAE
}
