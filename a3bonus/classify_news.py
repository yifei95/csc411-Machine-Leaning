#!/usr/bin/python3

from pylab import *
import numpy as np
import operator
import math
import random

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import sys


real = []
for line in open("clean_real.txt"): #get real titles\
    l = line.split()
    real.append(np.array(l))
real = np.array(real)

fake = []
for line in open("clean_fake.txt"): #get fake titles\
    l = line.split()
    fake.append(l) 
fake = np.array(fake)  

training = np.append(real, fake)
training_label = [1] * len(real) + [0] * len(fake)

word_index = {}
num_words = 0
for headline in training:
    for word in headline:
        if word not in word_index: 
            word_index[word] = num_words
            num_words += 1



#-----------------get previously saved weights and run the algorithm --------------
training_set = np.load("training_set.npy")
#training_label = np.load("training_label.npy")
#num_words = 5832

clf = naive_bayes.MultinomialNB(alpha=0.8)
clf.fit(training_set, training_label) 



#--------------- get the headlines of the new file --------------
filename = str(sys.argv[1])
headlines = []
for line in open(filename): 
    l = line.split()
    headlines.append(np.array(l))
headlines = np.array(headlines) 

#---------- vectorize the headlines ------------
test_set = np.zeros((0, num_words))
for headline in headlines:
    i = np.zeros(num_words)
    for word in headline:
        i[word_index[word]] = 1.
    i = np.reshape(i, [1, num_words])
    test_set = np.vstack((test_set, i))    

predicted = clf.predict(test_set)    

for i in range(len(predicted)):
    if predicted[i] == 0:
        print(1)
    else:
        print(0)    