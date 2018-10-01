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

from get_data import *


#================== helper functions ====================
def getimages(name, dic):
    """
    Read and stack images of the same person into an array from the dictionary dict.
    """
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
    Test the accuracy of the Part 3 classifier
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
    A classifier that distingusihes male and female.
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
    
    h = 0.000000001
    
    theta_h = np.zeros((1025, 4))
    theta_h[0,0] = h
    print((0,0))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta)) 
    
    theta_h = np.zeros((1025, 4))
    theta_h[0,1] = h
    print((0,1))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta)) 
    
    theta_h[0,2] = h
    print((0,2))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta)) 
    
    theta_h = np.zeros((1025, 4))
    theta_h[0,3] = h
    print((0,3))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta)) 
    
    theta_h = np.zeros((1025, 4))
    theta_h[1,1] = h
    print((1,1))
    print((f_part6(x, y, theta + theta_h) - f_part6(x, y, theta - theta_h)) / (2 * h))
    print(df_part6(x, y, theta))     


#================== Part 7 ==========================

def classifierPerson(dataset, size):
    """
    A classifier that distingusihes pictures of different actors and actresses.
    """
    
    act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
    
    x = np.empty((1024, 0))
    for name in act:
        data = getimages(name, dataset)
        x = np.hstack((x, data))
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
    Test the accuracy of the Part classifier.
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

if __name__ == "__main__":
    
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        
    tra = {}
    val = {}
    tes = {}
    download_save_im(act, tra, val, tes)
    
        
    #Part3
    print("Part 3:")
    prob1 = testPart3(tra, 70)
    print("The accuracy of the Part 3 classifier on training set is " + str(prob1) + "%")
    prob2 = testPart3(val, 10)
    print("The accuracy of the Part 3 classifier on validation set is " + str(prob2) + "%")
    prob3 = testPart3(tes, 10)
    print("The accuracy of the Part 3 classifier on test set is " + str(prob3) + "%")    
    
    #Part 4
    print("Part 4:")
    #Part 4(a)
    theta = classifierBC(tra, 70)
    resized = np.reshape(theta[1:], (32, 32))
    imshow(resized)
    show()
    
    dataset = {'carell48.jpg': ['carell', 'M'], 'baldwin23.jpg': ['baldwin', 'M']}
    theta1 = classifierBC(dataset, 1)
    resized1 = np.reshape(theta1[1:], (32, 32))
    imshow(resized1)
    show()
    
    #Part 4(b)
    #same way of producing image as part a, but the value of alpha in the function classifierBC is changed.
    theta = classifierBC(tra, 70)
    resized = np.reshape(theta[1:], (32, 32))
    imshow(resized)
    show()
    
    #Part 5
    print("Part 5:")
    prob70 = testPart5(tra, tra, 70, "cropped")
    print("The accuracy of Part 5 classifier build based on training set of size 70 per actor on traning set of size 70 per actor is " + str(prob70) + "%")
    prob71 = testPart5(tra, val, 10, "cropped")
    print("The accuracy of Part 5 classifier build based on training set of size 70 per actor on validation set of size 10 per actor is " + str(prob71) + "%")
    
    tra50 = build_tra(50)
    prob50 = testPart5(tra50, tra50, 50, "cropped")
    print("The accuracy of Part 5 classifier build based on training set of size 50 per actor on traning set of size 50 per actor is " + str(prob50) + "%")
    prob51 = testPart5(tra50, val, 10, "cropped")
    print("The accuracy of Part 5 classifier build based on training set of size 50 per actor on validation set of size 10 per actor is " + str(prob51) + "%")
    
    tra30 = build_tra(30)
    prob30 = testPart5(tra30, tra30, 30, "cropped")
    print("The accuracy of Part 5 classifier build based on training set of size 30 per actor on traning set of size 30 per actor is " + str(prob30) + "%")
    prob31 = testPart5(tra30, val, 10, "cropped")
    print("The accuracy of Part 5 classifier build based on training set of size 30 per actor on validation set of size 10 per actor is " + str(prob31) + "%") 
    
    tra10 = build_tra(10)
    prob10 = testPart5(tra10, tra10, 10, "cropped")
    print("The accuracy of Part 5 classifier build based on training set of size 10 per actor on traning set of size 10 per actor is " + str(prob10) + "%")
    prob11 = testPart5(tra10, val, 10, "cropped")
    print("The accuracy of Part 5 classifier build based on training set of size 10 per actor on validation set of size 10 per actor is " + str(prob11) + "%")
    
    training = [prob10, prob30, prob50, prob70]
    validation = [prob11, prob31, prob51, prob71]
    x_lab = [10, 30, 50, 70]
    plt.plot(x_lab, training, color='green', label='Training Set')
    plt.plot(x_lab, validation, color='red',  label='Validation Set')
    plt.legend(['Training Set', 'Validation Set'], loc='lower right')
    plt.xlabel('Size of Training Set', fontsize=12)
    plt.ylabel('Performance(%)', fontsize=12)
    plt.show()
    plt.close()

    
    print("Test the gender classifier on the actors and actresses not in act:")
    tes5 = {}
    
    act_part5 = ['Gerard Butler', 'Michael Vartan', 'Daniel Radcliffe', 'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']
    download_save_im_part5(act_part5, tes5)
    prob2 = testPart5(tra, tes5, 10, "cropped5")
    print("The accuracy of Part 5 classifier on test set of the six actors not included in act is " + str(prob2) + "%")
    
    
    #Part 6
    print("Part 6:")
    testPart6()
    
    
    #Part 7
    print("Part 7:")
    prob7_0 = testPart7(tra, 70)
    print("The accuracy of Part 7 classifier on the training set is " + str(prob7_0) + "%")
    
    prob7_1 = testPart7(val, 10)
    print("The accuracy of Part 7 classifier on the validation set is " + str(prob7_1) + "%")    
    
    #Part 8
    print("Part 8:")
    theta = classifierPerson(tra, 70)
      
    im1 = np.reshape(theta[1:,0], (32,32))
    im2 = np.reshape(theta[1:,1], (32,32))
    im3 = np.reshape(theta[1:,2], (32,32))
    im4 = np.reshape(theta[1:,3], (32,32))
    im5 = np.reshape(theta[1:,4], (32,32))
    im6 = np.reshape(theta[1:,5], (32,32))
    
    imshow(im1)
    show()
    imshow(im2)
    show()
    imshow(im3)
    show()
    imshow(im4)
    show()
    imshow(im5)
    show()
    imshow(im6)
    show()    
    
    
