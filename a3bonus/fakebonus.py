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

#=============== PART 1 ================================#

#get all the lines
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

words_real = []
for headline in real:
    words_real = words_real + list(set(headline))
words_real = np.array(words_real)

words_fake = []
for headline in fake:
    words_fake = words_fake + list(set(headline))
words_fake = np.array(words_fake)

#get the count of each word in the lines read
unique_real, counts_real = np.unique(words_real, return_counts=True)
real_freq = dict(zip(unique_real, counts_real))
unique_fake, counts_fake = np.unique(words_fake, return_counts=True)
fake_freq = dict(zip(unique_fake, counts_fake))

#split the datas into train, validate, and test set by random
random.seed(0)

random.shuffle(fake)
random.shuffle(real)

train_real = real[:int(math.floor(len(real) * 0.7))]
train_fake = fake[:int(math.floor(len(fake) * 0.7))]

validate_real = real[int(math.ceil(len(real) * 0.7)):int(math.floor(len(real) * 0.85))]
validate_fake = fake[int(math.ceil(len(fake) * 0.7)):int(math.floor(len(fake) * 0.85))]

test_real = real[int(math.ceil(len(real) * 0.85)):]
test_fake = fake[int(math.ceil(len(fake) * 0.85)):]
    
#===============================================================
#                         Bonus
#===============================================================
def bonusPart1():
    # ------------ make vectorized x and y -------------------
    training = np.append(train_real, train_fake)
    training_label = [1] * len(train_real) + [0] * len(train_fake)
    
    validation = np.append(validate_real, validate_fake)
    validation_label = [1] * len(validate_real) + [0] * len(validate_fake)
    
    test = np.append(test_real, test_fake)
    test_label = [1] * len(test_real) + [0] * len(test_fake)
    
    #assign each word a unique number and save the number of total unique words as num_words
    word_index = {}
    num_words = 0
    for headline in training:
        for word in headline:
            if word not in word_index: 
                word_index[word] = num_words
                num_words += 1
    for headline in validation:
        for word in headline:
            if word not in word_index: 
                word_index[word] = num_words
                num_words += 1
    for headline in test:
        for word in headline:
            if word not in word_index: 
                word_index[word] = num_words
                num_words += 1
    """
    make training set, validation set and testing set into 2-D numpy arrays which shows occurrence of words in each headline.
    For a single headline, the length of the array is the number of unique words. At each index of the array, it is 1 if the word is
    in the headline, it is 0 otherwise.
    """
    training_set = np.zeros((0, num_words))
    validation_set = np.zeros((0, num_words)) 
    test_set = np.zeros((0, num_words))
    
    for headline in training:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        training_set = np.vstack((training_set, i))
    for headline in validation:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        validation_set = np.vstack((validation_set, i))
    for headline in test:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        test_set = np.vstack((test_set, i)) 
    
    clf = naive_bayes.MultinomialNB(alpha=0.8)
    #clf = linear_model.LogisticRegression()
    clf.fit(training_set, training_label)
    print("MultinomialNB training set accuracy: " + str(100*clf.score(training_set, training_label)))
    print("MultinomialNB validation set accuracy: " + str(100*clf.score(validation_set, validation_label)))
    print("MultinomialNB test set taccuracy: " + str(100*clf.score(test_set, test_label)))

    
    predicted = clf.predict(test_set)
    
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == 0:
            count += 1
    print(str(count))
    
    cm = metrics.confusion_matrix(test_label, list(predicted))   
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Confusion matrix')
    classes=['FAKE', 'REAL']
    plt.xticks(array([0, 1]), classes)
    plt.yticks(array([0, 1]), classes)    
    
    plt.text(0, 0, cm[0, 0], horizontalalignment="center", color="blue")
    plt.text(0, 1, cm[0, 1], horizontalalignment="center", color="blue")
    plt.text(1, 0, cm[1, 0], horizontalalignment="center", color="blue")
    plt.text(1, 1, cm[1, 1], horizontalalignment="center", color="blue")

    plt.ylabel('Predicated')
    plt.xlabel('True')
    show()
    
    