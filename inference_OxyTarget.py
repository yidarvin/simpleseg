from __future__ import print_function
from __future__ import division

import argparse
import copy
import glob
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

from data.OxyTarget import *
from data.transforms import *
from engine.architectures import *
from engine.inference import *



def main(args):
    """
    Main function to parse arguments.
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Drug Discovery Experiments.")

    # Filepaths
    parser.add_argument("--pData", dest="path_data", type=str, default='/home/Data/drugDiscovery/exp000')
    parser.add_argument("--pModel", dest="path_model", type=str, default='/home/Models')
    parser.add_argument("--name", dest="name", type=str, default='default')

    # Creating Parser Object
    opts = parser.parse_args(args[1:])

    if not isdir(opts.path_model):
        mkdir(opts.path_model)

    in_chan = 2
    out_chan = 2
    batch_size   = 1
    name_exp = 'OxyTarget_' + opts.name
    path_save = opts.path_data + '_' + name_exp
    if not isdir(path_save):
        mkdir(path_save)

    # Create the Dataloaders
    dataloader = create_dataloaders_OxyTarget_inference(opts.path_data,batch_size)

    # Create the model
    model = densenet101(in_chan, out_chan, pretrained=False)

    # Do the training
    inference_seg(model, dataloader, path_save=opts.path_model, name_exp=name_exp, path_vis=path_save, verbose=True)


if __name__ == "__main__":
    main(sys.argv)
