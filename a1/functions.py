from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import scipy.misc as misc
import matplotlib.image as mpimg
import matplotlib.cm as cm
import os
from scipy.ndimage import filters
import urllib
import pandas as pd
from numpy.linalg import norm


#================== helper functions ====================
def getimages(name, dic):
    x = np.empty((1024, 0))
    for i in dic.keys():
        if dic[i][0] == name:
            image = imread("cropped/" + i)
            data = (np.reshape(image, (1024, 1)))/255.0
            x = np.hstack((x, data))
    return x


def f(x, y, theta):
    x = np.vstack( (ones((1, x.shape[1])), x))
    return np.sum( (y - np.dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = np.vstack( (ones((1, x.shape[1])), x))
    return -2*np.sum((y-np.dot(theta.T, x))*x, 1)


def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        #if iter % 500 == 0:
            #print "Iter", iter
            #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            #print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

def build_tra(size):
    """
    (int) -> dict
    Buidling training set for act with different sizes from the original training set.
    """
    actor  = ['baldwin', 'hader', 'carell']
    actress = ['bracco', 'gilpin', 'harmon']
    result = {}
    
    for name in actor:
        count = 0
        for i in tra.keys():
            if (tra[i][0] == name) and (count < size):
                result[i] = [name, "M"]
                count += 1
    for name in actress:
        count = 0
        for i in tra.keys():
            if (tra[i][0] == name) and (count < size):
                result[i] = [name, "F"]
                count += 1
    return result
                

#============================ Part3 ==========================
def classifierBC(dataset, size):
    """
    A classifier that distingusihes pictures of Alec Baldwin form pictures of Steve Carell.
    Baldwin is 0. Carell is 1.
    """
    xB = getimages("baldwin", dataset)
    xC = getimages("carell", dataset)
    x = np.hstack((xB,xC))
    yB = np.array([0] * size)
    yC = np.array([1] * size)
    y = vstack((yB,yC))
    y = y.flatten()
    init_theta = np.array([0.3] * 1025)
    theta = grad_descent(f, df, x, y, init_theta, 0.00001)
    
    return theta

def part3(image):
    """
    Compute the output of the classifier. Either Steve Carell or Alec Baldwin is returned.
    """
    result = ""
    theta = classifierBC(tra, 70)
    im = imread(image)
    data = (np.reshape(im, (1024, 1)))/255.0
    data = np.vstack( (ones((1, data.shape[1])), data))
    y = np.dot(theta.T, data)            
    if y <= 0.5:
        result = "Alec Baldwin"
    elif y > 0.5:
        result = "Steve Carell"
    return result 


def testPart3(dataset, size):
    """
    Test the accuracy of the classifier
    """
    score = 0
    theta = classifierBC(tra, 70)
    for i in dataset.keys():
        if dataset[i][0] == "baldwin":
            im = imread("cropped/" + i)
            data = (np.reshape(im, (1024, 1)))/255.0
            data = np.vstack( (ones((1, data.shape[1])), data))
            re = np.dot(theta.T, data)            
            if re <= 0.5:
                score += 1
        if dataset[i][0] == "carell":
            im = imread("cropped/" + i)
            data = (np.reshape(im, (1024, 1)))/255.0
            data = np.vstack( (ones((1, data.shape[1])), data))
            re = np.dot(theta.T, data)            
            if re > 0.5:
                score += 1
    prob = (float(score) / (size * 2)) * 100
    return prob


#================= Part 5 ======================
def classifierGender(dataset, size):
    """
    A classifier that distingusihes pictures of Alec Baldwin form pictures of Steve Carell.
    Baldwin is 0. Carell is 1.
    """
    male = ['baldwin', 'hader', 'carell']
    female = ['bracco', 'gilpin', 'harmon']
    
    data = []
    for name in male:
        data.append(getimages(name, dataset))
    for name in female:
        data.append(getimages(name, dataset))
    x = np.hstack(data)
    yM = np.array([0] * (size*3))
    yF = np.array([1] * (size*3))
    y = vstack((yM,yF))
    y = y.flatten()
    init_theta = np.array([0.00] * 1025)
    theta = grad_descent(f, df, x, y, init_theta, 0.000005)
    
    return theta

def testPart5(tra_set, dataset, size, folder):
    """
    (dict, dict, int, str) -> float
    
    folder is either cropped or cropped5. cropped is the folder for the original 6 people given.
    cropped 5 is the folder of the 6 actors not included in act.
    
    Test the accuracy of the classifierGender.
    """
    score = 0
    theta = classifierGender(tra_set, (len(tra_set) / 6))
    for i in dataset.keys():
        if dataset[i][1] == "M":
            im = imread(folder + "/" + i)
            data = (np.reshape(im, (1024, 1)))/255.0
            data = np.vstack( (ones((1, data.shape[1])), data))
            re = np.dot(theta.T, data)            
            if re <= 0.5:
                score += 1
        if dataset[i][1] == "F":
            im = imread(folder+ "/" + i)
            data = (np.reshape(im, (1024, 1)))/255.0
            data = np.vstack( (ones((1, data.shape[1])), data))
            re = np.dot(theta.T, data)            
            if re > 0.5:
                score += 1
    prob = (float(score) / (size * 6)) * 100
    return prob


#================= Part 6 =========================
def f_part6(x, y, theta):
    x = np.vstack( (ones((1, x.shape[1])), x))
    return np.sum(np.sum((y - np.dot(theta.T, x)) ** 2))

def df_part6(x, y, theta):
    x = np.vstack( (ones((1, x.shape[1])), x))
    return -2*np.dot(x,(y - np.dot(theta.T, x)).T)

def testPart6():
    random.seed(0)
    y = np.reshape(np.random.rand(4 * 140), (4, 140))
    x = np.reshape(np.random.rand(1024 * 140), (1024, 140))
    theta = np.reshape(np.random.random(1025 * 4), (1025, 4))
    
    h = 0.00000001
    
    theta_h = np.zeros((1025, 4))
    theta_h[0,0] = h
    print((0,0))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta)) 
    
    theta_h = np.zeros((1025, 4))
    theta_h[1,2] = h
    print((1,2))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta))
    
    theta_h = np.zeros((1025, 4))
    theta_h[0,1] = h
    print((0,1))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta)) 
    
    theta_h = np.zeros((1025, 4))
    theta_h[1,1] = h
    print((1,1))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta)) 
    
    theta_h = np.zeros((1025, 4))
    theta_h[0,2] = h
    print((0,2))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta))     


#================== Part 7 ==========================

def classifierPerson(dataset, size):
    """
    A classifier that distingusihes pictures of Alec Baldwin form pictures of Steve Carell.
    Baldwin is 0. Carell is 1.
    """
    
    act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
    
    x = np.empty((1024, 0))
    #y = np.empty((6, 0), int)
    for name in act:
        data = getimages(name, dataset)
        x = np.hstack((x, data))
    #x = np.hstack(data)
    yBracco = np.zeros((6, size))
    yBracco[0, :] = 1
    yGilpin = np.zeros((6, size))
    yGilpin[1, :] = 1
    yHarmon = np.zeros((6, size))
    yHarmon[2, :] = 1
    yBaldwin = np.zeros((6, size))
    yBaldwin[3, :] = 1
    yHader = np.zeros((6, size))
    yHader[4, :] = 1
    yCarell = np.zeros((6, size))
    yCarell[5, :] = 1    
    y = hstack((yBracco, yGilpin, yHarmon, yBaldwin, yHader, yCarell))
    
    init_theta = np.array([0.25] * 1025 * 6)
    init_theta = np.reshape(init_theta, (1025, 6))
    theta = grad_descent(f_part6, df_part6, x, y, init_theta, 0.0000005)
    
    return theta

def testPart7(dataset, size):
    """
    Test the accuracy of the classifier
    """
    score = 0
    
    theta = classifierPerson(tra, 70)
    
    for i in dataset.keys():
        im = imread("cropped/" + i)
        data = (np.reshape(im, (1024, 1)))/255.0
        data = np.vstack( (ones((1, data.shape[1])), data))
        re = np.dot(theta.T, data)
        max_row = np.argmax(re)
        if (max_row == 0) and dataset[i][0] == "bracco":
            score += 1
        elif (max_row == 1) and dataset[i][0] == "gilpin":
            score += 1
        elif (max_row == 2) and dataset[i][0] ==  "harmon":
            score += 1
        elif (max_row == 3) and dataset[i][0] ==  "baldwin":
            score += 1
        elif (max_row == 4) and dataset[i][0] == "hader":
            score += 1
        elif (max_row == 5) and dataset[i][0] == "carell":
            score += 1
    prob = (float(score) / (size * 6)) * 100
    return prob
