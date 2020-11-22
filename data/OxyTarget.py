from __future__ import print_function
from __future__ import division

import copy
import glob
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import pickle
import platform
from PIL import Image
import time

import cv2
import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import binary_dilation,binary_erosion
from skimage import exposure

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets, models, transforms,utils
from torchvision.transforms import functional as func

from .transforms import *
from utils.tools import *


def rescale(img):
    new_img = img - img.min()
    new_img /= (new_img.max() + 1e-6)
    return new_img

def preproc_imgs(img,seg,resize=256):
    img = img.astype(np.float32)
    if resize < img.shape[1] and resize < img.shape[2]:
        _,m,n = img.shape
        iis,jjs = np.where(seg > 0)
        if np.random.rand() > 0.5 and len(iis) > 0:
          ind = np.random.choice(len(iis))
          ii = int(np.clip(iis[ind] - resize / 2, 0, m-resize))
          jj = int(np.clip(jjs[ind] - resize / 2, 0, n-resize))
        else:
          ii = np.random.choice(m-resize+1)
          jj = np.random.choice(n-resize+1)
        img = img[:,ii:(ii+resize),jj:(jj+resize)]
        seg = seg[ii:(ii+resize),jj:(jj+resize)]
    elif resize != img.shape[1] and resize != img.shape[2]:
        new_img = np.zeros((img.shape[0], resize, resize))
        for ii in range(img.shape[0]):
            new_img[ii,:,:] = cv2.resize(img[ii,:,:].astype(np.float32), (resize,resize))
        img = new_img
        seg = cv2.resize(seg.astype(np.float32), (resize,resize), interpolation=cv2.INTER_NEAREST)
    img -= img.min()
    img /= (img.max()+1e-6)
    
    return img,seg
    
def sample_extend_data(images,labels,frames,dataset_size=1):
    images_tr = []
    labels_tr = []
    frames_tr = []
    for ii in range(int(np.ceil(dataset_size / len(images)))):
        images_tr += images
        labels_tr += labels
        frames_tr += frames
    return images_tr,labels_tr,frames_tr

class OxyTargetDataset(Dataset):
    def __init__(self, path_data, resize=256, train=True, dataset_size=1, transform=None):
        self.path_data = path_data
        
        num_test = [43,153,24,103,106,150,68,73,46,165]
        num_val = [57,96,134,64,145]
        
        self.path_imgs = []
        self.frame_num = []
        self.labels    = []
        
        for name_img in listdir(join(path_data, "MRI_OxyTarget")):
            if len(name_img) > 13:
                num = int(name_img[10:-14])
            else:
                num = int(name_img[10:])
            if num in num_test:
                continue
            elif num in num_val and train:
                continue
            elif num not in num_val and not train:
                continue
            img = nib.load(join(path_data, "MRI_OxyTarget", name_img, 'T2.nii')).get_fdata()
            self.path_imgs += [join(path_data, "MRI_OxyTarget", name_img) for ii in range(1,img.shape[2]-1)]
            self.labels    += [join(path_data, "GT_OxyTarget", str(num) + '.nii') for ii in range(1,img.shape[2]-1)]
            self.frame_num += range(1,img.shape[2]-1)
        self.train = train
        self.path_imgs,self.labels,self.frame_num = sample_extend_data(self.path_imgs,self.labels,self.frame_num,dataset_size)
        self.resize = resize
        self.transform = transform
    def __len__(self):
        return len(self.path_imgs)
    def __getitem__(self,idx):
        im1 = rescale(nib.load(join(self.path_imgs[idx], 'T2.nii')).get_fdata())
        im2 = rescale(nib.load(join(self.path_imgs[idx], 'b1000.nii')).get_fdata())
        seg = (nib.load(join(self.labels[idx])).get_fdata() > 0) + 0
        #if self.train:
        #    resize = self.resize
        #else:
        #    resize = max(img.shape[0],img.shape[1])
        ii  = self.frame_num[idx]
        im1 = im1[:,:,(ii-1):(ii+2)].transpose([2,0,1])
        im2 = im2[:,:,(ii-1):(ii+2)].transpose([2,0,1])
        img = np.concatenate([im1,im2], axis=0)
        seg = seg[:,:,ii]
        if self.train:
            resize = self.resize
        else:
            resize = max(im1.shape[1], self.resize)
        img,seg = preproc_imgs(img,seg,resize)
        #if resize != im1.shape[0] or resize != im1.shape[1]:
        #    for jj in range(3):
        #        im1_new[:,:,jj] = cv2.resize(im1[:,:,jj].astype(np.float32), (resize,resize))
        #        im2_new[:,:,jj] = cv2.resize(im2[:,:,jj].astype(np.float32), (resize,resize))
        #    seg = cv2.resize(seg.astype(np.float32), (resize,resize), interpolation=cv2.INTER_NEAREST)
        #im1 = im1.transpose([2,0,1])#im1.reshape([1, resize, resize])
        #im2 = im2.transpose([2,0,1])#im2.reshape([1, resize, resize])
        #img = np.concatenate([im1,im2], axis=0)
        
        sample = {'X': img, 'Y': seg}
        if self.transform:
            sample = self.transform(sample)

        return sample
        

def create_dataloaders_OxyTarget(path_data='/home/Data/OxyTargetData',
                                 batch_size=4,img_size=256,dataset_size=1,validation=True):
    data_tr = OxyTargetDataset(path_data, img_size, True, dataset_size,
                               transform=transforms.Compose([RandomFlip(),RandomRotate(),RandomShift(),AddNoise(),ToTensor()]))
    loader_tr = DataLoader(data_tr, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloaders = {}
    dataloaders['train'] = loader_tr
    if validation:
        data_va = OxyTargetDataset(path_data, img_size, False, transform=transforms.Compose([ToTensor()]))
        loader_va = DataLoader(data_va, batch_size=1, shuffle=False, num_workers=0)
        dataloaders['val'] = loader_va
    return dataloaders


class OxyTargetInferenceDataset(Dataset):
    def __init__(self, path_data, transform=None):
        self.path_data = path_data
        self.path_imgs = [join(path_data, name_img) for name_img in listdir(path_data)]
        self.transform = transform
    def __len__(self):
        return len(self.path_imgs)
    def __getitem__(self,idx):
        img = cv2.imread(self.path_imgs[idx])
        img,seg = preproc_img(img,resize=img.shape[0])
        
        sample = {'X': img, 'Y': seg, 'lab': self.path_imgs[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
def create_dataloaders_OxyTarget_inference(path_data='/home/Data/drugDiscovery/exp000', batch_size=4):
    data = drugDiscoveryInferenceDataset(path_data, transform=transforms.Compose([ToTensor()]))
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
    return loader
