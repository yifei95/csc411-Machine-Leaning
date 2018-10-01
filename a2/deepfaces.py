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

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Load images
test = np.load("test10.npy")
train = np.load("train10.npy")
validate = np.load("validate10.npy")
data = np.load("data10.npy")


#=========================================================================#
#                                Part 10                                  #
#=========================================================================#

#=============== functions to get data ======================
def download_images():

    raw_male = "facescrub_actors_male.txt"

    raw_female = "facescrub_actresses.txt"
    
    data = get_raw(raw_female, "F")
    male = get_raw(raw_male, "M")
    
    data.update(male)
    return data
    
    
act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
        
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray/255.

testfile = urllib.request.FancyURLopener() 
  
def get_raw(textfile, gender):  
    data = {}    
    
    #create directories to save photos
    if not os.path.isdir("uncropped10"):
        os.makedirs("uncropped10")
    if not os.path.isdir("cropped10"):
        os.makedirs("cropped10")    
    
    #Note: you need to create the uncropped folder first in order 
    #for this to work
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(textfile):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
         
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped10/"+filename), {}, 45)
            
                #croped out saved inmages
                dim = line.split()[5].split(",")
                x1 = int(dim[0])
                y1 = int(dim[1])
                x2 = int(dim[2])
                y2 = int(dim[3])  
                hashval = line.split()[6]
                print(filename + "----" + hashval)
                print(dim)
                #bracco0.jpg----4f9a8a6f1377b03133bc4df0cc240d924e938bc4f163703d4222c796c5d0bd92
                #['84', '44', '187', '147'] 
                #bracco1.jpg----963a9b134c26f8aff0714bbe2c11b88f7995fd89a5beb5a88bf21ae46244a850
                #['861', '971', '2107', '2217']                

                if not os.path.isfile("uncropped10/"+filename):
                    continue
                elif (hashlib.sha256(open("uncropped10/" + filename, "rb").read()).hexdigest()) != hashval:
                    continue #check hash values for invalid faces
                else:
                    try:
                        print(filename)
                        im = imread("uncropped10/"+filename)
                        cropp = im[y1:y2, x1:x2]
                        resized = imresize(cropp, (227, 227))                        
                        if (len(im.shape) == 2): #skip grey images
                            continue
                        else: #save color images
                            imsave("cropped10/"+filename, resized)
                            if os.path.isfile("cropped10/"+filename):
                                print(filename)
                                data[filename] = [name, gender]
                    except Exception:
                        print(filename + ":cannot read")
                i += 1
    return data

def seperate_dataset(data):
    train = {}
    validate = {}
    test = {}
    keys = data.keys()
    splitted_data = []
    #split each actors 
    for j in range(len(act)):
        splitted_data.append({})
        name = act[j].split()[1].lower()
        for element in keys:
            if data[element][0]  == name:
                splitted_data[j][element] = data[element]
                
    gilpin = 0
    for i in range(len(splitted_data)):
        if "gilpin" in list(splitted_data[i].keys())[0]:
            gilpin = i
    
    for j in range(len(splitted_data)):
        actors_data = splitted_data[j]
        if j != gilpin:
            i = 0 #to keep track of how many pictures added for each actor
            keys_a = list(actors_data.keys())
            print(keys_a)
            while i<100 and len(keys_a)>0 :
                index = random.randint(0, len(keys_a)-1) #add pictures randomly
                element = keys_a[index]
                if i < 70:
                    print(element + "------" + str(i))
                    train.update({element:data[element]})
                    i = i + 1
                elif i < 90:
                    print(element + "------" + str(i))
                    test.update({element:data[element]})
                    i = i + 1
                elif i < 100:
                    print(element + "------" + str(i))
                    validate.update({element:data[element]})
                    i = i + 1
                keys_a.remove(element)
        elif j == gilpin:
            i = 0 #to keep track of how many pictures added for each actor
            keys_a = list(actors_data.keys())
            print(keys_a)
            while i<86 and len(keys_a)>0 :
                index = random.randint(0, len(keys_a)-1) #add pictures randomly
                element = keys_a[index]
                if i < 56:
                    print(element + "------" + str(i))
                    train.update({element:data[element]})
                    i = i + 1
                elif i < 76:
                    print(element + "------" + str(i))
                    test.update({element:data[element]})
                    i = i + 1
                elif i < 86:
                    print(element + "------" + str(i))
                    validate.update({element:data[element]})
                    i = i + 1            
    return [train, validate, test]

#======================= get data ===========================
#read and save images
data10 = download_images()
sep_d = seperate_dataset(data10)
np.save("data10.npy", data10)
np.save("train10.npy", sep_d[0])
np.save("validate10.npy", sep_d[1])
np.save("test10.npy", sep_d[2])


#====================== AlexNet and tests on performance ===============

#Load images
test = np.load("test10.npy")
train = np.load("train10.npy")
validate = np.load("validate10.npy")
data = np.load("data10.npy")



#Class AlexNet
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.classifier(x)
        return x



#get data output to x and y and encode using ONE HOT encoding
def get_data(dataset):
    actork =['Bracco', 'Gilpin', 'Harmon', 'Baldwin', 'Hader', 'Carell']
    x = []
    y = []
    d = dict(dataset.flatten()[0])
    for pic in d.keys():
        if d[pic][0] == 'bracco':
            im = imread("cropped10/"+pic)[:,:,:3]
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)            
            x.append(im)
            y.append([1, 0, 0, 0, 0, 0])
        elif d[pic][0] == 'gilpin':
            im = imread("cropped10/"+pic)[:,:,:3]
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)            
            x.append(im)
            y.append([0, 1, 0, 0, 0, 0])
        elif d[pic][0] == 'harmon':
            im = imread("cropped10/"+pic)[:,:,:3]
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)            
            x.append(im)
            y.append([0, 0, 1, 0, 0, 0])
        elif d[pic][0] == 'baldwin':
            im = imread("cropped10/"+pic)[:,:,:3]
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)            
            x.append(im)
            y.append([0, 0, 0, 1, 0, 0])  
        elif d[pic][0] == 'hader':
            im = imread("cropped10/"+pic)[:,:,:3]
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)            
            x.append(im)
            y.append([0, 0, 0, 0, 1, 0])  
        elif d[pic][0] == 'carell':
            im = imread("cropped10/"+pic)[:,:,:3]
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)            
            x.append(im)
            y.append([0, 0, 0, 0, 0, 1])   

    return np.array(x), np.array(y)



#=================== get the activations and test data=======================

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


def getActivations(dataset):
    ANmodel = MyAlexNet()
    dataset_x, dataset_y = get_data(dataset)
    
    x = Variable(torch.from_numpy(dataset_x)).type(dtype_float)
    
    activations = ANmodel.forward(x)
    
    return activations, dataset_y


activations, train_y = getActivations(train)
activ_train = activations.data.numpy()
save("activ_train.npy", activ_train)
activ_tr = np.load("activ_train.npy")
x = Variable(torch.from_numpy(activ_tr)).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(train_y,1)), requires_grad=False).type(dtype_long)

iterations = []
test_performance = []
train_performance = []
validate_performance = []

train_x, train_y_iter = getActivations(train)
validation_x, validation_y = getActivations(validate)
test_x, test_y = getActivations(test)



#=================== run the model and test on performance =======================

dim_activations = 256 * 6 * 6
dim_h = 30
dim_out = 6



model = torch.nn.Sequential(
    torch.nn.Linear(dim_activations, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step
    if t%100 == 0:
        print("iteration" + str(t))
        
    #store iteration in list
    iterations.append(t)
    
    #compute train result and performance store in list
    y_pred1 = model(train_x).data.numpy()
    p_train = np.mean(np.argmax(y_pred1, 1) == np.argmax(train_y_iter, 1)) * 100   
    train_performance.append(p_train)
    
    #compute validate dataset and performance store in list
    y_pred2 = model(validation_x).data.numpy()
    p_validation = np.mean(np.argmax(y_pred2, 1) == np.argmax(validation_y, 1)) * 100
    validate_performance.append(p_validation)      

    #compute test dataset and performance store in list 
    y_pred3 = model(test_x).data.numpy()
    p_test = np.mean(np.argmax(y_pred3, 1) == np.argmax(test_y, 1)) * 100
    test_performance.append(p_test)
   

#plot the learning curve 
plt.plot(iterations, train_performance, color='blue', label="training set")
plt.plot(iterations, test_performance, color='green', label="test set")
plt.plot(iterations, validate_performance, color='red', label="validation set")
plt.xlabel('Number of Iterations', fontsize=12)
plt.ylabel('Performance(%)', fontsize=12)
plt.legend(loc='top left')
plt.show()

#print the performance on different sets
print("The performance of the training set is " + str(p_train) + "%")
print("The performance of the validation set is " + str(p_validation) + "%")
print("The performance of the test set is " + str(p_test) + "%")


