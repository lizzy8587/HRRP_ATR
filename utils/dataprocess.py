#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   dataprocess.py
@Version      :   1.0
@Time         :   2023/03/05 21:48:27
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   烟台数据切片、可视化
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

class YantaiData():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def read_hrrp_mat(mat_path, complex=True):
        '''
        description : 读取mat_path对应mat中的hrrp数据
        param       : complex(True返回复数/False返回幅度)
        return      : 2D-array，每一行代表一条hrrp
        '''
        z = list(scio.loadmat(mat_path).values())[-1]
        if not complex: z = abs(z)
        return z

    @staticmethod
    def hrrp2d_filter(hrrp2d):
        '''
        description : 过滤hrrp2d中可能不含目标的hrrp序列
        param       : 2d-array
        return      : 2d-array，每一行代表经过滤的hrrp
        '''
        targets_index = np.where(np.mean(hrrp2d ,axis=1)>np.mean(hrrp2d))[0]    # 条件：局部均值>全局均值
        return hrrp2d[targets_index]

    @staticmethod
    def plot_hrrp2d(hrrp2d, save_path=None, title=None, colorbar=None):
        '''
        description : 画目标hrrp对应的热力图
        param       : 2d-array
        return      : 显示/保存图片
        '''
        x,y = np.meshgrid(
            np.linspace(0,hrrp2d.shape[1]-1,hrrp2d.shape[1]),
            np.linspace(0,hrrp2d.shape[0]-1,hrrp2d.shape[0])
        )
        plt.figure()
        plt.contourf(x, y, hrrp2d)
        if title: plt.title(title)
        if colorbar: plt.colorbar()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_hrrp1d(hrrp1d, save_path=None, title=None):
        '''
        description : 画hrrp1d图
        param       : 1d-array
        return      : 显示/保存图片
        '''
        plt.figure()
        plt.plot(hrrp1d)
        if title: plt.title(title)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
                