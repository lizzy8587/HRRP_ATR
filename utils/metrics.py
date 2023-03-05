#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   metrics.py
@Version      :   1.0
@Time         :   2023/01/27 20:21:42
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   None
'''

import torch

def accurate_labels(predictions, labels):
    '''
    description: 计算准确率
    param: 预测概率向量shape(bs,classes)，和标签shape(bs)
    return : 预测正确个数，总个数
    '''
    pred = torch.max(predictions.data, 1)[1] 
    rights = pred.eq(labels.data.view_as(pred)).sum().item()
    return rights, len(labels) 