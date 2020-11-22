import cv2
import numpy as np


def prep_UnitNorm(img, resize=None):
    img = img.astype(np.float32)
    if resize != None:
        img = cv2.resize(img, (resize,resize))
    img -= img.min()
    img /= (img.max() + 1e-6)
    return img.transpose([2,0,1])

def prep_ZeroCenter(img, resize=None):
    img = img.astype(np.float32)
    if resize != None:
        img = cv2.resize(img, (resize,resize))
    img = img / 128.0 - 1
    return img.transpose([2,0,1])

def prep_torchvision(img, resize=None):
    img = img.astype(np.float32)
    if resize != None:
        img = cv2.resize(img, (resize,resize))
    img -= np.array([0.485, 0.456, 0.406]).reshape([1,1,3])
    img /= np.array([0.229, 0.224, 0.225]).reshape([1,1,3])
    return img.transpose([2,0,1])
