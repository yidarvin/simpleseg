from __future__ import print_function
from __future__ import division

import copy
import glob
from mnist import MNIST
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
from PIL import Image
import time

import cv2
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
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead





def rotate_image_lin(image, angle):
  image_center = tuple(np.array(image.shape) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
  return result

def rotate_image_near(image, angle):
  image_center = tuple(np.array(image.shape) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_NEAREST)
  return result

class RandomTurn(object):
    def __init__(self, max_angle = 30.0):
        self.max_angle = int(max_angle)
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        angle = np.random.randn() * self.max_angle / 2
        for ii in range(X.shape[0]):
            X[ii,:,:] = rotate_image_lin((X[ii,:,:].astype(np.float32)+1)*128, angle) / 128 - 1.0
        Y[:,:] = np.round(rotate_image_near(Y[:,:].astype(np.float32), angle))
        return {'X':X, 'Y':Y}


class RandomRotate(object):
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        rotnum = np.random.choice(4)
        for ii in range(X.shape[0]):
            X[ii,:,:] = np.rot90(X[ii,:,:],k=rotnum,axes=(0,1))
        Y[:,:] = np.rot90(Y,k=rotnum,axes=(0,1))
        return {'X':X, 'Y':Y}

class RandomShift(object):
    def __init__(self, max_shift=32):
        self.max_shift = int(max_shift)
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        h,w = X.shape[1:]
        X_shift = np.zeros((X.shape[0], X.shape[1]+2*self.max_shift, X.shape[2]+2*self.max_shift))
        for ii in range(X.shape[0]):
            X_shift[ii,:,:] = np.pad(X[ii,:,:], self.max_shift, mode='constant',constant_values=-1)
        Y_shift = np.pad(Y, self.max_shift, mode='constant')
        top     = np.random.randint(0, 2*self.max_shift)
        left    = np.random.randint(0, 2*self.max_shift)
        X[:,:,:] = X_shift[:,top:(top+h), left:(left+w)]
        Y[:,:]   = Y_shift[top:(top+h), left:(left+w)]
        return {'X':X, 'Y':Y}

class RandomFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        if np.random.rand() > self.flip_prob:
            X[:,:,:] = X[:,:,::-1]
            Y[:,:]   = Y[:,::-1]
        return {'X':X, 'Y':Y}

class AddNoise(object):
    def __init__(self, noise_prob=0.5):
        self.noise_prob = noise_prob
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        if np.random.rand() > self.noise_prob:
            X += np.random.randn(X.shape[0],X.shape[1],X.shape[2]) / 100
        return {'X':X, 'Y':Y}

class IsolateColor(object):
    def __init__(self, noise_prob=1.0):
        self.noise_prob = noise_prob
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        if np.random.rand() > self.noise_prob:
            ind = np.random.choice(X.shape[0])
            for ii in range(X.shape[0]):
                if ii != ind:
                    X[ii,:,:] = 0
                X *= X.shape[0]
        return {'X':X, 'Y':Y}

class ColorJitter(object):
    def __init__(self, brightness=0.03, contrast=0.03, saturation=0.03, hue=0.03):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
    def __call__(self, sample):
        X,Y = sample['X'],sample['Y']
        X_new = Image.fromarray(np.uint8(128*(X.transpose(1,2,0) + 1)))
        X_new = self.transform(X_new)
        X_new = (np.array(X_new).astype(np.float32) / 128 - 1.0).transpose(2,1,0)
        return {'X':X_new, 'Y':Y}
        
class ToTensor(object):
    def __call__(self, sample):
        X,Y = sample['X'], sample['Y']
        sample['X'] = torch.from_numpy(X).float()
        sample['Y'] = torch.from_numpy(Y).long()
        return sample
