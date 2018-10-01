import numpy as np
from math import *
import matplotlib.pyplot as plt
import pylab


def distance(p1, p2):
    """
    Calculate the Euclidean distance between x and y.
    """
    summation = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    dist = sqrt(summation)
    return dist

def get_dist(item):
    return item[0]

def knn_classifier(p, x1, x2, k):
    """
    Classify the label of the given data p by k-nearest neighbours.
    """
    dist = []
    size_x1 = len(x1)
    size_x2 = len(x2)
    for i in range(size_x1):
        dist.append([distance(x1[i], p), 1])
    for j in range(size_x2):
        dist.append([distance(x2[j], p), 2])
    sorted_dist = sorted(dist, key=get_dist)
    
    label_1 = 0
    label_2 = 0
    for l in range(0, k):
        if sorted_dist[l][1] == 1:
            label_1 += 1
        elif sorted_dist[l][1] == 2:
            label_2 += 1
    
    if label_1 > label_2:
        return "x1"
    elif label_1 < label_2:
        return "x2"
    
#================= Test the Classifier ====================
def testClassifier1(p, label, k):
    """
    (list, str) -> float
    """
    score = 0
    size = len(p)
    for i in range(size):
        if knn_classifier(p[i], x1, x2, k) == label:
            score += 1
    result = (float(score) / size) * 100
    return result

def testClassifier2(p, k):
    """
    (list, str) -> float
    """
    score = 0
    size = len(p)
    for i in range(0, 50):
        if knn_classifier(p[i], x1, x2, k) == "x1":
            score += 1
    for j in range(50, 100):
        if knn_classifier(p[j], x1, x2, k) == "x2":
            score += 1            
    result = (float(score) / size) * 100
    return result


#generate random gaussian distribution data with seed = 0
np.random.seed(0)
mean1 = [0, 0]
cov1 = [[1,0.2],[0.2,1]]
x1 = np.random.multivariate_normal(mean1, cov1, 5)

mean2 = [2, 2]
cov2 = [[2,0],[0,2]]
x2 = np.random.multivariate_normal(mean2, cov2, 50)

x = pylab.vstack((x1, x2))

x1_t = x1.T
x2_t = x2.T
plt.scatter(x1_t[0], x1_t[1], color='green', label = 'X1')
plt.scatter(x2_t[0], x2_t[1], color='red', label = 'X2')
plt.legend(['X1', 'X2'], loc='lower right')
plt.title('numpy.random.seed(0)', fontsize=12)
plt.show()

performance = []
k = []
for i in range(1, 51, 2):
    k.append(i)
    performance.append(testClassifier1(x1, "x1", i))
    
plt.plot(k, performance, color='green')
plt.title('Performance of knn classifier (only x1 is used in the test)', fontsize=12)
plt.xlabel('k', fontsize=12)
plt.ylabel('Performance(%)', fontsize=12)
plt.show()


performance = []
k = []
for i in range(1, 101, 2):
    k.append(i)
    performance.append(testClassifier2(x, i))
    
plt.plot(k, performance, color='green')
plt.title('Performance of knn classifier (fullsize training set is used in the test)', fontsize=12)
plt.xlabel('k', fontsize=12)
plt.ylabel('Performance(%)', fontsize=12)
plt.show()

#generate random gaussian distribution data with seed = 20
np.random.seed(20)
mean1 = [0, 0]
cov1 = [[2,0],[0,2]]
x1 = np.random.multivariate_normal(mean1, cov1, 50)

mean2 = [2, 2]
cov2 = [[2,0],[0,2]]
x2 = np.random.multivariate_normal(mean2, cov2, 50)

x = pylab.vstack((x1, x2))

x1_t = x1.T
x2_t = x2.T
plt.scatter(x1_t[0], x1_t[1], color='green', label = 'X1')
plt.scatter(x2_t[0], x2_t[1], color='red', label = 'X2')
plt.legend(['X1', 'X2'], loc='lower right')
plt.title('numpy.random.seed(20)', fontsize=12)
plt.show()


performance = []
k = []
for i in range(1, 51, 2):
    k.append(i)
    performance.append(testClassifier1(x1, "x1", i))
    
plt.plot(k, performance, color='green')
plt.title('Performance of knn classifier (only x1 is used in the test)', fontsize=12)
plt.xlabel('k', fontsize=12)
plt.ylabel('Performance(%)', fontsize=12)
plt.show()


performance = []
k = []
for i in range(1, 101, 2):
    k.append(i)
    performance.append(testClassifier2(x, i))
    
plt.plot(k, performance, color='green')
plt.title('Performance of knn classifier (fullsize training set is used in the test)', fontsize=12)
plt.xlabel('k', fontsize=12)
plt.ylabel('Performance(%)', fontsize=12)
plt.show()

    
    