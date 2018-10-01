from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

#import cPickle
import os
from scipy.io import loadmat
import hashlib

import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision

import torch.nn as nn

im = imread("face2.jpeg")
model = nn.Sequential(nn.Conv2d(1,
                        3,
                        3))


model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size(5, 5)))

#def visualize_cat(batch):
im = np.squeeze(batch, axis=0)
plt.imshow(im)
    
def print_img(model, data):
    batch = np.expand_dims(data, axis=0)
    conv2 = model.predict(batch)
    conv2 = np.squeeze(conv2, axis=0)
    