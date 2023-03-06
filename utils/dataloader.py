#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   dataloader.py
@Version      :   1.0
@Time         :   2023/01/27 20:21:36
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   数据加载模块
'''

import scipy.io as scio
from scipy.signal import find_peaks
import numpy as np
import torch
import os
import pandas as pd
from .dataprocess import YantaiData

class FeatureExtraction(object):
  '''
  description: extract hrrp basic features
  param: sample(numpy(w),num)
  return: sample(numpy(features),num)
  '''
  def __init__(self,window_size=10,mu=1,peak_height=0.08) -> None:
    self.window_size = window_size
    self.mu = mu
    self.peak_height = peak_height

  def __call__(self, sample, method='peak'):
    '''
    param: method: 'peak'|'threshold'
    '''
    # --------------------- find the first and the last bin of the traget ---------------------
    if method=='peak':
      start,end = self.get_target_by_peaks(sample[0])
    elif method=='threshold':
      start,end = self.get_target(sample[0])
    else:
      raise ValueError('the method you input({}) doesn\'t exist!'.format(method))

    # --------------------- extract hrrp featrues ---------------------
    target = sample[0][start:end]
    features = []
    features.append(np.mean(target))  # mean
    features.append(np.var(target)) # variance
    features.append(self.get_length(start,end))  # length
    features.append(self.get_center_of_mass(target))  # center of mass
    features.append(self.get_energy(target))  # energy
    features.extend(self.get_peaks(target)) # NP,DPK,DEP,EA,EP

    return np.array(features), sample[1]

  def get_energy(self,hrrp):
    return np.sum(hrrp**2)

  def sliding_average(self,x,window_size):
    weights = np.ones(window_size)/window_size
    return np.convolve(weights,x,'same')

  def get_target_by_threshold(self,raw_hrrp):
    '''
    description: find the target's first bin and last bin
    param: raw_hrrp:numpy(w)
    return: index,index
    '''
    eta = self.mu*np.sqrt(np.mean(raw_hrrp**2))  # threshold
    slide = self.sliding_average(raw_hrrp,self.window_size)    # sliding average
    # tmp = np.nonzero(slide>eta)[0]
    tmp = np.where(slide>eta)[0]
    start,end = tmp[0],tmp[-1]      # choose the first and last bin
    return start,end

  def get_target_by_peaks(self,raw_hrrp):
    peaks, _ = find_peaks(raw_hrrp,height=self.peak_height)
    peaks0, _ = find_peaks(raw_hrrp)
    start_idx = np.where(peaks0==peaks[0])[0].item()-1
    end_idx = np.where(peaks0==peaks[-1])[0].item()+1
    start = peaks0[start_idx if start_idx>0 else 0]  # 前一个小峰
    end = peaks0[end_idx if end_idx<len(peaks0) else len(peaks0)-1]    # 后一个小峰
    return start,end

  def get_length(self,start,end):
    return end-start

  def get_center_of_mass(self,hrrp):
    return np.sum(np.multiply(hrrp,np.arange(1,len(hrrp)+1)))/np.sum(hrrp)

  def get_peaks(self,hrrp):
      peak_index,_ = find_peaks(hrrp,height=self.peak_height)
      NP = len(peak_index)
      peak_value = hrrp[peak_index]
      tmp = np.argsort(peak_value)    # the index of sorted peak_value (ascending)
      DPK = 0 if NP<2 else peak_index[tmp[-1]]-peak_index[tmp[-2]]
      DEP = len(hrrp)- peak_index[tmp[-1]]
      EA = -np.sum(np.multiply(peak_value,np.log2(peak_value)))
      peak_index_normal = peak_index/len(hrrp)
      EP = -np.sum(np.multiply(peak_index_normal,np.log2(peak_index_normal)))

      return NP,DPK,DEP,EA,EP

class AmplitudeNormalize(object):
  def __call__(self, sample):
    tmp = (sample[0]-sample[0].min())/(sample[0].max()-sample[0].min()) # sample[0]/np.max(sample[0])
    tmp[np.where(np.isinf(tmp))] = 0
    return tmp, sample[1]
    # return (sample[0] -np.mean(sample[0]))/ np.std(sample[0]), sample[1]

class padZeros(object):
  def __init__(self, length) -> None:
    self.length = length

  def __call__(self, sample):
    if self.length <= len(sample):
      return sample
    else:
      return np.concatenate([sample[0], np.zeros(self.length-len(sample[0]))], axis=0), sample[1]

class To4DTensor(object):
    '''
    description: convert numpy to tensor ---> used by oneD_CNN model
    param: sample(numpy(w),num)
    return: sample(tensor(c,h,w),tensor(1))
    '''
    def __call__(self, sample):
        return torch.tensor(sample[0],dtype=torch.float)[None,None,:],torch.tensor(sample[1],dtype=torch.int64)


class To3DTensor(object):
  def __call__(self, sample):
    return torch.tensor(sample[0], dtype=torch.float)[None, :], torch.tensor(sample[1], dtype=torch.int64)


class hrrpDataset(torch.utils.data.Dataset):
  '''
  description: generate a hrrp dataset format for Train_hrrp.mat (download from github)
  '''
  def __init__(self, filepath, train=True, transform=None):
    # read data
    arg = 'aa' if train else 'bb'
    data_base = scio.loadmat(filepath)[arg]
    self.hrrp = data_base[:,3:] # numpy shape(bs,w)
    labels_ = data_base[:,0:3]  # onehot
    self.labels = np.array([np.argmax(k) for k in labels_]) # numpy shape(bs)
    # # numpy to tensor
    # self.hrrp = torch.tensor(self.hrrp,dtype=torch.float)[:,None,None,:]    # [bs,c,h,w]
    # self.labels = torch.tensor(self.labels,dtype=torch.int64)

    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    sample = self.hrrp[index], self.labels[index]
    if self.transform is not None:
      sample = self.transform(sample)
    return sample


class hrrpDataset2(torch.utils.data.Dataset):
  '''
  description: generate a hrrp dataset format (四类民船mat格式)
  '''
  def __init__(self, filepath, transform=None, length=None):
    # read data
    targets = os.listdir(filepath)
    targets_mapping = dict(zip(targets, [k for k in range(len(targets))]))

    hrrp = []
    labels = []
    for target in targets:
      target_dir = os.path.join(filepath, target)
      for dir in os.listdir(target_dir):
        mat_dir = os.path.join(target_dir, dir)
        for file in os.listdir(mat_dir):
          mat_path = os.path.join(mat_dir, file)
          if file.split('.')[-1] != 'mat': continue
          x = scio.loadmat(mat_path)['dataR'][1]
          # length = length if length else len(x)
          if length:
            if length>len(x):
              # x = np.concatenate([x, np.zeros(length-len(x))], axis=0)
              left = int((length-len(x))/2)
              right = length-len(x)-left
              x = np.pad(x,(left,right),'constant',constant_values=0)
            else:
              x = x[0:length]
          hrrp.append(x[0:length])
          labels.append(targets_mapping[target])

    self.hrrp = np.array(hrrp)
    self.labels = np.array(labels)
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    sample = self.hrrp[index], self.labels[index]
    if self.transform is not None:
      sample = self.transform(sample)

    if np.isinf(sample[0]).any():
      print('inf data in index:',index)
    # print(sample[0])
    return sample


class hrrpDataset3(torch.utils.data.Dataset):
  '''
  description: generate a hrrp dataset format(9类npy格式-mt)
  '''
  def __init__(self, filepath, transform=None, length=None):
    # read data
    targets = os.listdir(filepath)
    targets_mapping = dict(zip(targets, [k for k in range(len(targets))]))

    hrrp = []
    labels = []
    for target in targets:
      for file in os.listdir(os.path.join(filepath,target)):
          npy_path = os.path.join(filepath,target, file)
          if file.split('.')[-1] != 'npy': continue
          x = np.load(npy_path)
          if length:
            if length>len(x):
              # x = np.concatenate([x, np.zeros(length-len(x))], axis=0)
              left = int((length-len(x))/2)
              right = length-len(x)-left
              x = np.pad(x,(left,right),'constant',constant_values=0)
            else:
              x = x[0:length]
          hrrp.append(x)
          labels.append(targets_mapping[target])

    self.hrrp = np.array(hrrp)
    self.labels = np.array(labels)
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    sample = self.hrrp[index], self.labels[index]
    if self.transform is not None:
      sample = self.transform(sample)

    if np.isinf(sample[0]).any():
      print('inf data in index:',index)
    # print(sample[0])
    return sample


class hrrpDataset4(torch.utils.data.Dataset):
  '''
  description: generate a hrrp dataset format (Yantai dataset,mat格式)
  '''
  def __init__(self, filepath, transform=None, length=None):
    # read data
    targetsMMSI = os.listdir(filepath)
    df = pd.read_csv(os.path.join(filepath, 'info.csv'))
    df.MMSI = df.MMSI.astype('str')
    mmsi2targets = dict(zip(df.MMSI, df.targetClass))
    targets = set(df.targetClass)
    targets2id = dict(zip(targets, [k for k in range(len(targets))]))

    hrrp = []
    labels = []
    for targetMMSI in targetsMMSI:
      target_dir = os.path.join(filepath, targetMMSI)
      if not os.path.isdir(target_dir): continue
      mat_path = os.path.join(target_dir, 'data.mat')
      try:
        x = YantaiData.read_hrrp_mat(mat_path, complex=False)
        x = YantaiData.hrrp2d_filter(x)
      except:
        print('Load Error: ',mat_path)
        continue
      if length:
        if length>x.shape[1]:
          # x = np.concatenate([x, np.zeros(length-len(x))], axis=0)
          left = int((length-x.shape[1])/2)
          right = length-x.shape[1]-left
          x = np.pad(x,(left,right),'constant',constant_values=0)
        else:
          x = x[:, 0:length]
      hrrp.extend(x)
      labels.extend([targets2id[mmsi2targets[targetMMSI]]]*len(x))

    self.hrrp = np.array(hrrp)
    self.labels = np.array(labels)
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    sample = self.hrrp[index], self.labels[index]
    if self.transform is not None:
      sample = self.transform(sample)

    if np.isinf(sample[0]).any():
      print('inf data in index:',index)
    # print(sample[0])
    return sample
