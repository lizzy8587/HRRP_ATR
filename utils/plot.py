#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   plot.py
@Version      :   1.0
@Time         :   2023/03/05 22:38:31
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   可视化相关函数
'''


import torch

def confusion_matrix(preds, labels, cls_num):
    conf_matrix = torch.zeros(cls_num, cls_num)
    if len(preds.shape) > 1:
        preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix