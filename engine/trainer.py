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
    
def miou_pytorch(output, Y):
    prob = F.softmax(output,dim=1)
    pred = torch.argmax(prob,dim=1)
    miou = 0.0
    for ii in range(1,output.shape[1]):
        intersection = ((pred == ii) & (Y == ii)).float().sum((1,2))
        union        = ((pred == ii) | (Y == ii)).float().sum((1,2))
        miou = (intersection + 1e-6) / (union + 1e-6) / (output.shape[1] - 1)
    
    return miou.mean()

def take_one_step(model,X,Y,criterion,phase=None,alpha=1):
    alpha = np.clip(alpha,0,1)
    with torch.set_grad_enabled(phase == 'train'):
        output = model(X)
        loss = alpha*criterion(output,Y)
        loss += (1-alpha)*criterion(output,torch.argmax(output,dim=1))
        Y_copy = Y.detach().cpu().numpy()
        weight = np.ones_like(Y_copy)
        #for ii in range(Y.size(0)):
        #    Y_slice = Y_copy[ii,:,:]
        #    boundary = Y_slice > 0
        #    boundary = binary_dilation(boundary, iterations=2)
        #    weight[ii,:,:] += boundary * 9
        loss = torch.mean(loss * torch.from_numpy(weight).float().cuda())
        miou = miou_pytorch(output, Y)
    return miou,loss

def trainer_seg(model, dataloaders, path_save=None, name_exp='experiment', learning_rate=1e-4, num_epochs=25, verbose=True):
    since = time.time()
    criterion = nn.CrossEntropyLoss(reduction='none')
    model = nn.DataParallel(model.cuda())
    if path_save != None:
        path_log = join(path_save, name_exp + '.txt')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-44)
    ##scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,5,eta_min=1e-10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer,10,gamma=0.1,last_epoch=-1)
    best_acc = 0.0

    for epoch in range(num_epochs):
        if verbose:
            if path_save != None:
                super_print(path_log, 'Epoch {}/{}'.format(epoch, num_epochs - 1))
                super_print(path_log, '-' * 10)
            else:
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_acc = 0.0
            running_loss = 0.0

            #Iterate over data.
            optimizer.zero_grad()
            for ii,sample_batch in enumerate(dataloaders[phase]):
                X = sample_batch['X'].cuda()
                Y = sample_batch['Y'].cuda()
                acc,loss = take_one_step(model,X,Y,criterion,phase=phase)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.set_grad_enabled(False):
                        out = model(X)
                        img_save = np.zeros((X.shape[2], X.shape[3], 3))
                        seg_save = np.zeros((X.shape[2], X.shape[3], 3))
                        for jj in range(min(3, X.shape[1])):
                            img_save[:,:,jj] = X.detach().cpu().numpy()[0,jj,:,:]
                        seg_save[:,:,1] = 1 - (Y.detach().cpu().numpy()[0,:,:] == 0)
                        seg_save[:,:,0] = 1 - (np.argmax(out.detach().cpu().numpy()[0,:,:,:], axis=0) == 0)
                        seg_save[:,:,2] = 1 - (np.argmax(out.detach().cpu().numpy()[0,:,:,:], axis=0) == 0)
                        cv2.imwrite('/home/darvin/Data/scratch/' + str(ii)+'.png', (np.concatenate([img_save,seg_save],axis=1)*255).astype(np.uint8))
                running_acc += acc.item() * X.size(0)
                running_loss += loss.item() * X.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)

            if verbose:
                if path_save != None:
                    super_print(path_log,'{} Loss: {:.4f} IoU: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                else:
                    print('{} Loss: {:.4f} IoU: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #if phase == 'train':
            #    scheduler.step()
            if phase == 'val':
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    if path_save != None:
                        torch.save(model.state_dict(), join(path_save, name_exp + '_best.pth'))
        print()

    if path_save != None:
        torch.save(model.state_dict(), join(path_save, name_exp + '_late.pth'))
    time_elapsed = time.time() - since
    if verbose:
        if path_save != None:
            super_print(path_log, 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            super_print(path_log, 'Best val IoU: {:4f}'.format(best_acc))
        else:
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val IoU: {:4f}'.format(best_acc))

    return model
