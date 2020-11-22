from __future__ import print_function
from __future__ import division

import copy
import glob
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import sys
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

def super_print(path_file, statement):
    sys.stdout.write(statement + '\n')
    sys.stdout.flush()
    f = open(path_file, 'a')
    f.write(statement + '\n')
    f.close()
    return 0

def inference_seg(model, dataloader, path_save='/home/Models', name_exp='experiment', path_vis=None, verbose=True):
    model = nn.DataParallel(model.cuda())
    best_state_dict = torch.load(join(path_save, name_exp + '_best.pth'))
    model.load_state_dict(best_state_dict)
    
    if path_vis != None:
        path_log = join(path_vis, name_exp + '.txt')

    model.eval()

    #Iterate over data.
    counter = 0
    for sample_batch in dataloader:
        X = sample_batch['X'].cuda()
        lab = sample_batch['lab']
        with torch.set_grad_enabled(False):
            Y = F.softmax(model(X), dim=1).detach().cpu().numpy()
        for ii in range(Y.shape[0]):
            name_img = lab[ii]
            pred = Y[ii,:,:,:]
            seg  = (pred[0,:,:] < 0.5) + 0
            if np.sum(seg) == 0:
                if verbose:
                    if path_vis != None:
                        super_print(path_log, name_img + ':No Predicted ROI')
                    else:
                        print(path_log, name_img + ':No Predicted ROI')
                continue
            probs = []
            for jj in range(1, Y.shape[1]):
                probs.append(np.mean(pred[jj,:,:][seg>0]))
            probs = np.array(probs)
            maxclass = np.argmax(probs)
            if verbose:
                if path_vis != None:
                    super_print(path_log, name_img + ':' + str(maxclass))
                    for jj in range(len(probs)):
                        super_print(path_log, '    ' + str(jj) + ':{:.4f}'.format(probs[jj]))
                else:
                    print(path_log, name_img + ':' + str(maxclass))
                    for jj in range(len(probs)):
                        print('    ' + str(jj) + ':{:.4f}'.format(probs[jj]))
            if path_vis != None:
                if X.shape[1] > 3:
                    img = X[ii,:3,:,:].detach().cpu().numpy().transpose(1,2,0)
                elif X.shape[1] == 3:
                    img = X[ii,:,:,:].detach().cpu().numpy().transpose(1,2,0)
                else:
                    img = np.zeros((X.shape[2], X.shape[3], 3))
                    for jj in range(3):
                        img[:,:,jj] = X[ii,0,:,:].detach().cpu().numpy()
                img -= img.min()
                img /= img.max() + 1e-6
                sav = np.zeros_like(img)
                sav[:,:,1] = pred[maxclass+1,:,:]
                sav[:,:,0] = np.sum(pred[1:,:,:], axis=0) - pred[maxclass+1,:,:]
                sav[:,:,2] = np.sum(pred[1:,:,:], axis=0) - pred[maxclass+1,:,:]
                img_sav = np.concatenate([img,sav], axis=1)
                cv2.imwrite(join(path_vis, str(counter) + '.png'), (255*img_sav).astype(np.uint8))
            counter += 1
        print()

    return 0


