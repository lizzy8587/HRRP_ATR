#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   oneD_CNN.py
@Version      :   1.0
@Time         :   2023/01/27 20:21:01
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   1D卷积神经网络模型
'''

import torch.nn as nn
from typing import List, Callable, Union, Any, TypeVar, Tuple


class oneD_CNN(nn.Module):
    '''
    description: 1D卷积神经网络模型
    '''
    def __init__(self,in_channels: int,
                 cls_num: int = 3,
                 length: int = 256,
                 hidden_dims: List = None,
                 dropout_ratio: float = 0.3,
                 **kwargs) -> None:
        super().__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU()),
                    nn.MaxPool1d(2),
                    nn.BatchNorm1d(h_dim)
            )
            in_channels = h_dim

        self.layers = nn.Sequential(*modules)
        self.fc = nn.Linear(int(length * hidden_dims[0] / 2), cls_num)
        self.dropout = nn.Dropout1d(dropout_ratio)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        return self.fc(x)

# class oneD_CNN(nn.Module):
#     '''
#     description: 1D卷积神经网络模型
#     '''
#     def __init__(self) -> None:
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(1,16,5,1,2),
#             nn.LeakyReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(16,32,5,1,2),
#             nn.LeakyReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         self.out = nn.Sequential(
#             nn.Linear(32*50,9),
#             # nn.Softmax(dim=1)
#         )
#
#     def forward(self,x,out_layer=False):
#         '''
#         output_layer: 当该参数为True时，返回各中间层的参数结果(dict)；否则直接返回预测结果
#         '''
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = x2.view(x2.shape[0],-1)
#         x3 = self.out(x3)
#         return {'layer1':x1,'layer2':x2,'output':x3} if out_layer else x3
