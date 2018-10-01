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
import torch
from torch.autograd import Variable
from scipy.io import loadmat
#import cPickle
import os
from scipy.io import loadmat

#import pickle

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#snapshot = pickle.load(open("snapshot50.pkl", "rb"), encoding="latin1")
#W0 = snapshot["W0"]
#b0 = snapshot["b0"].reshape((300,1))
#W1 = snapshot["W1"]
#b1 = snapshot["b1"].reshape((10,1))

#=========================================================================#
#                                Part 1                                   #
#=========================================================================#
def part1():
    # M - is a dictionary composed of 10 training set and 10 testing sets
    # each set contains a number from 0 to 9
    
    print(len(M["train5"]))
    print(len(M["train4"]))
    #each training set have around 5000 to 7000 number images 
    #these images is an array of size 784 (28 * 28)
    
    print(len(M["test7"]))
    print(len(M["test0"]))
    #each testing set have around 900 to 1200 number images 
    #these images is an array of size 784 (28 * 28)

    for i in range(10):
        display(i)


def displayNumbers(i):
    
    train = "train" + str(i)
    random.seed(1)
    rand = random.random(10) * len(M[train])
    r = [int(i) for i in rand]
    
    #Display 10 images for each number 
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(M[train][r[0]].reshape((28,28)), cmap=cm.gray)
    axarr[0, 1].imshow(M[train][r[1]].reshape((28,28)), cmap=cm.gray)
    axarr[0, 2].imshow(M[train][r[2]].reshape((28,28)), cmap=cm.gray)
    axarr[0, 3].imshow(M[train][r[3]].reshape((28,28)), cmap=cm.gray)   
    axarr[0, 4].imshow(M[train][r[4]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 0].imshow(M[train][r[5]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 1].imshow(M[train][r[6]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 2].imshow(M[train][r[7]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 3].imshow(M[train][r[8]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 4].imshow(M[train][r[9]].reshape((28,28)), cmap=cm.gray)
        
    # Fine-tune figure; make subplots farther from each other.
    f.subplots_adjust(hspace=0.3)
    
    plt.show()
    
def test_part1():
    for i in range(10):
        displayNumbers(i)
    
  
#=========================================================================#
#                                Part 2                                   #
#=========================================================================#
def calculate_output(X, W):
    X = np.vstack( (ones((1, X.shape[1])), X))
    output = dot(W.T, X)
    return output 

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def part_2(X, W):
    y = calculate_output(X, W)
    result = softmax(y)
    return result

#=========================================================================#
#                                Part 3                                   #
#=========================================================================#

def f_p3(x, y, w):
    """
    Use the sum of the negative log-probabilities of all the training cases 
    as the cost function.
    """
    return -sum(y * log(part_2(x, w)))

def df_p3(x, y, w):
    p = part_2(x, w)
    x = np.vstack( (ones((1, x.shape[1])), x))
    return dot(x, (p - y).T)

def part3():
    random.seed(0)
    x = reshape(random.rand(784 * 20), (784, 20))
    y = zeros((10, 20))
    y[0, :] = 1
    w = reshape(random.rand(785 * 10), (785, 10))
    
    h = zeros((785, 10))
    h[0, 0] = 1e-5
    
    print("Gradient Function value at position (0, 0)=======")
    print(str(df_p3(x, y, w)[0][0]))
    print("Finite Difference at position (0, 0) =======")
    print(str(((f_p3(x, y, w+h) - f_p3(x, y, w))/(h))[0][0]))
    
#=========================================================================#
#                                Part 4                                   #
#=========================================================================#
def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    weights = []
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 100 == 0:
            cur_t = t.copy()
            weights.append(cur_t)
            print("Iter" + str(iter))
            #print("Gradient: " + str(df(x, y, t)) + "\n")
        iter += 1
    return t, weights

def one_hot(dataset, size):
    """
    Make x and y for the one-hot encoding.
    param: size: size get from each separete dataset
    param: dataset: indicates if using "test" or "train" 
    """
    x = np.empty((784, 0))
    y = np.empty((10, 0))
    
    for i in range(10):
        data = dataset + str(i)
        x = np.hstack((x, M[data][:size].T))
        y_i = np.zeros((10,1))
        y_i[i] = 1
        for j in range(size):
            y = np.hstack((y, y_i)) 
    x = x / 255.0
    
    return x, y
    
def part_4_train(alpha):
    """
    Train the neural network using gradient descent.
    """
    
    init_weights = zeros((785, 10))
    random.seed(3)
    x, y = one_hot("train", 100)

    opt_w, weights = grad_descent(f_p3, df_p3, x, y, init_weights, alpha)
    return opt_w


def test_part4(dataset, size, alpha):
    '''
    Tests performance on the training and test sets
    :param optimized_weights: thetas that will be tested
    :return: performance values in a tuple
    '''
    
    score = 0
    theta = part_4_train(alpha)
    
    x_test, y_test = one_hot(dataset, size)
    y_pred = part_2(x_test, theta)
    
    for i in range(size*10):
        
        if argmax(y_pred.T[i]) == argmax(y_test.T[i]):
            score += 1
    
    return (score/float(size*10)) * 100

def part_4_plotlearningcurve(alpha, size):
    
    init_weights = zeros((785, 10))
    random.seed(3)
    x, y = one_hot("train", 100)

    opt_w, weights = grad_descent(f_p3, df_p3, x, y, init_weights, alpha)
    performance = []
    iterations = []
    performance_train = []
    x_test, y_test = one_hot("test", size)
    
    for i in range(len(weights)):
        
        #get the performance of the test set
        print("get the performance of the test set")
        y_pred = part_2(x_test, weights[i]) 
        score = 0
        for j in range(size*10):  
            if argmax(y_pred.T[j]) == argmax(y_test.T[j]):
                score += 1        
        performance.append((score/float(size*10)) * 100)
        
        #get performance of the training set
        print("get performance of the training set")
        y_pred_train = part_2(x, weights[i])        
        score = 0
        for j in range(100*10):  
            if argmax(y_pred_train.T[j]) == argmax(y.T[j]):
                score += 1        
        performance_train.append((score/float(100*10)) * 100)
                     
        #update number of iterations 
        iterations.append(i)        
    
    plt.plot(iterations, performance_train, color='blue', label="training set")
    plt.plot(iterations, performance, color='green', label="test set")
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Performance(%)', fontsize=12)
    plt.legend(loc='top left')
    plt.title("Learning Curve without Momentum")
    plt.show()

def part_4_plotweights():
    weights = np.load("weights.npy")
    theta = weights[29999]
    
    zero = np.reshape(theta.T[0][1:], (28, 28))
    one = np.reshape(theta.T[1][1:], (28, 28))
    two = np.reshape(theta.T[2][1:], (28, 28))
    three = np.reshape(theta.T[3][1:], (28, 28))
    four = np.reshape(theta.T[4][1:], (28, 28))
    five = np.reshape(theta.T[5][1:], (28, 28))
    six = np.reshape(theta.T[6][1:], (28, 28))
    seven = np.reshape(theta.T[7][1:], (28, 28))
    eight = np.reshape(theta.T[8][1:], (28, 28))
    nine = np.reshape(theta.T[9][1:], (28, 28))
    
    #Display 10 images for each number 
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(zero)
    axarr[0, 1].imshow(one)
    axarr[0, 2].imshow(two)
    axarr[0, 3].imshow(three)   
    axarr[0, 4].imshow(four)
    axarr[1, 0].imshow(five)
    axarr[1, 1].imshow(six)
    axarr[1, 2].imshow(seven)
    axarr[1, 3].imshow(eight)
    axarr[1, 4].imshow(nine)    
    
    f.subplots_adjust(hspace=0.3)
    
    plt.show()    
    
#=========================================================================#
#                                Part 5                                   #
#=========================================================================#

def grad_descent_with_momentum(f, df, x, y, init_t, alpha, gamma):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    weights = []
    v = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        v = gamma*v + alpha*df(x, y, t)
        t -= v
        if iter % 100 == 0:
            cur_t = t.copy()
            weights.append(cur_t)
            print("Iter" + str(iter))
            #print("Gradient: " + str(df(x, y, t)) + "\n")
        iter += 1
    return t, weights

def part_5_train(alpha, gamma):
    """
    Train the neural network using gradient descent.
    """
    
    init_weights = zeros((785, 10))
    random.seed(3)
    x, y = one_hot("train", 100)

    opt_w, weights = grad_descent_with_momentum(f_p3, df_p3, x, y, init_weights, alpha, gamma)
    return opt_w


def test_part5(dataset, size):
    '''
    Tests performance on the training and test sets
    :param optimized_weights: thetas that will be tested
    :return: performance values in a tuple
    '''
    
    score = 0
    theta = part_5_train(0.000001, 0.99)
    
    x_test, y_test = one_hot(dataset, size)
    y_pred = part_2(x_test, theta)
    
    for i in range(size*10):
        
        if argmax(y_pred.T[i]) == argmax(y_test.T[i]):
            score += 1
    
    return (score/float(size*10)) * 100

def part_5_plotlearningcurve(alpha, gamma, size):  
    
    init_weights = zeros((785, 10))
    random.seed(0)
    x, y = one_hot("train", 100)

    opt_w, weights = grad_descent_with_momentum(f_p3, df_p3, x, y, init_weights, alpha, gamma)
    performance = []
    iterations = []
    performance_train = []
    #np.save("weightsMOM.npy", weights)
    #np.save("opt_w_MOM.npy", opt_w)    
    x_test, y_test = one_hot("test", size)
    
    #for weights generated with momentum
    for i in range(len(weights)):
        
        #get the performance of the test set
        print("get the performance of the test set")
        y_pred = part_2(x_test, weights[i]) 
        score = 0
        for j in range(size*10):  
            if argmax(y_pred.T[j]) == argmax(y_test.T[j]):
                score += 1        
        performance.append((score/float(size*10)) * 100)
        
        #get performance of the training set
        print("get performance of the training set")
        y_pred_train = part_2(x, weights[i])        
        score = 0
        for j in range(100*10):  
            if argmax(y_pred_train.T[j]) == argmax(y.T[j]):
                score += 1        
        performance_train.append((score/float(100*10)) * 100)
                     
        #update number of iterations 
        iterations.append(i)
    
    plt.plot(iterations, performance_train, color='blue', label="training set")
    plt.plot(iterations, performance, color='green', label="test set")
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Performance(%)', fontsize=12)
    plt.legend(loc='top left')
    plt.show()
    
#=========================================================================#
#                                Part 6                                   #
#=========================================================================#
def contour_plot():
    w1_index = 378
    w2_index = 322
    index = 1
    
    init_weights = zeros((785, 10))
    random.seed(0)
    x, y = one_hot("train", 100)    
    opt_w, weights = grad_descent_with_momentum(f_p3, df_p3, x, y, init_weights, 0.00001, 0.9)
    opt_w1, weights1 = grad_descent(f_p3, df_p3, x, y, init_weights, 0.00001)
    
    #get all te value of w1 and w2 from weights with momentum
    w1_mom = []
    w2_mom = []    
    for i in range(len(weights)):
        w1_mom.append(weights[i][w1_index][index])
        w2_mom.append(weights[i][w2_index][index])
    
    theta_mom = weights[-1]
    X, Y = meshgrid(w1_mom, w2_mom)
    x, y = one_hot("test", 30)
    
    #for weights generated with momentum
    Z = np.zeros([len(w1_mom), len(w2_mom)])
    for i, w_1 in enumerate(w1_mom):
        for j, w_2 in enumerate(w2_mom):
            w = theta_mom.copy()
            w[w1_index][index] = w_1
            w[w2_index][index] = w_2
            Z[i,j] = f_p3(x, y, w)
    
    #tragectory with momentum
    w_traj_mom = []
    iter  = 0
    while iter < len(weights):
        if iter % 20 == 0:
            w_traj_mom.append(weights[iter])
        iter += 1
        
    w1_traj_mom = []
    w2_traj_mom = []
    mo_traj = [] #store the trajetory of momentumed weight
    for i in range(len(w_traj_mom)):
        w1_traj_mom.append(w_traj_mom[i][w1_index][index])
        w2_traj_mom.append(w_traj_mom[i][w2_index][index])
    for i in range(len(w1_traj_mom)):
        mo_traj.append((w1_traj_mom[i], w2_traj_mom[i]))
    
    #tragectory without momentum
    w_traj = []
    iter  = 0
    while iter < len(weights1):
        if iter % 20 == 0:
            w_traj.append(weights1[iter])
        iter += 1
        
    w1_traj = []
    w2_traj = []
    gd_traj = [] #store the trajetory of momentumed weight
    for i in range(len(w_traj)):
        w1_traj.append(w_traj[i][w1_index][index])
        w2_traj.append(w_traj[i][w2_index][index])
    for i in range(len(w1_traj)):
        gd_traj.append((w1_traj[i], w2_traj[i]))    

    #Part6a plot the counter plot with mom, the trajectory with mom and the trajectory without mom
    CS = plt.contour(X, Y, Z)   
    clabel(CS, inline=1, fontsize=10)
    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.legend(loc='upper left')
    plt.title('Contour plot')    
    plt.show()
  


if __name__ == "__main__":
    
    #PART1
    #test_part1()
