from pylab import *
import numpy as np
import operator
import math
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import tree
import graphviz 

Allwords = set()
#=============== PART 1 ================================#

#get all the lines
real = []
for line in open("clean_real.txt"): #get real titles\
    l = line.split()
    real.append(np.array(l))
    Allwords = Allwords.union(set(l))
real = np.array(real)


fake = []
for line in open("clean_fake.txt"): #get fake titles\
    l = line.split()
    fake.append(l) 
    Allwords = Allwords.union(set(l))
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

def part1():
    """
    choose words hillary, trumps and says as examples.
    """
    print("real:")
    print("('hillary', " + str(real_freq["hillary"]) + ")")
    print("('trumps', " + str(real_freq["trumps"]) + ")")
    print("('says', " + str(real_freq["says"]) + ")")
    print("fake:")
    print("('hillary', " + str(fake_freq["hillary"]) + ")")
    print("('trumps', " + str(fake_freq["trumps"]) + ")")
    print("('says', " + str(fake_freq["says"]) + ")") 

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


#=============== PART 2 ================================#

#making a word list with the word as key and 
#number of occurences in real and fake headlines as the value
word_list = {}
#value = [real, fake]
for i in range(len(train_real)):
    headline = set(train_real[i])
    for word in headline:
        if word not in word_list:
            word_list[word] = [1, 0]
        else:
            word_list[word] = [word_list[word][0] + 1, word_list[word][1]]
for i in range(len(train_fake)):
    headline = set(train_fake[i])
    for word in headline:
        if word not in word_list:
            word_list[word] = [0, 1]
        else:
            word_list[word] = [word_list[word][0], word_list[word][1] + 1]


#def Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p):
def Naive_Bayes_classifier(headline, word_list, training_set, training_label, m, p):
    """
    predicting whether a headline is real or fake.
    """
    #headline = headline.split()
    
    #calculate P(fake) and P(real)
    n = len(train_real) + len(train_fake)
    count_real = len(train_real)
    count_fake = len(train_fake)
    prob_fake = len(train_fake) / float(n)
    prob_real = 1.0 - prob_fake
    
    prob_word_real = []
    prob_word_fake = []   
    for i in word_list.keys():
        #P(word_i|real)
        P_word_i_real = (word_list[i][0]+m*p)/float(count_real + 1)
        #P(word_i|fake)
        P_word_i_fake = (word_list[i][1]+m*p)/float(count_fake + 1)
        
        if i in headline:
            prob_word_real.append(P_word_i_real)
            prob_word_fake.append(P_word_i_fake)           
        elif i not in headline:
            prob_word_real.append(1. - P_word_i_real)
            prob_word_fake.append(1. - P_word_i_fake)           
    
    #conditional independence is assumed by Naive Bayes
    #do multiplication to get P(words|real) and P(words|fake)
    multi_real = 0
    for p in prob_word_real:
        multi_real += math.log(p)
    multi_real = math.exp(multi_real)
    
    multi_fake = 0
    for p in prob_word_fake:
        multi_fake += math.log(p)
    multi_fake = math.exp(multi_fake)
    
    #compute P(class)*P(words|class)
    prob_real_words = prob_real * multi_real
    prob_fake_words = prob_fake * multi_fake
    
    #compute P(class)*(1 - P(words|class)) for part 3
    prob_real_not_words = prob_real * (1. - multi_real)
    prob_fake_not_words = prob_fake * (1. - multi_fake)   
    
    #probability that the given headline is fake, P(fake|words)
    prob = prob_fake_words/ (prob_fake_words + prob_real_words)
    
    #probability that the headline is fake when the word absence, P(fake|~words), for part 3
    prob_absence = prob_fake_not_words/ (prob_fake_not_words + prob_real_not_words)    
    
    result = "real"
    if prob > 0.5:
        result = "fake"
    
    return result, prob, prob_absence

def test_part2():
    m = 1
    p = 0.1
    
    count_train = 0
    n_train = len(train_real) + len(train_fake)
    for headline in train_real:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_train += 1
    for headline in train_fake:
        result, prob_fake, prob_absence= Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_train += 1
    performance_train = count_train / float(n_train) * 100
    print("The performance of the Naive Bayes classifer on the training set is " + str(performance_train) + "%")
    
    count_val = 0
    n_val = len(validate_real) + len(validate_fake)
    for headline in validate_real:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_val += 1
    for headline in validate_fake:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_val += 1
    performance_val = count_val / float(n_val) * 100
    print("The performance of the Naive Bayes classifer on the validationx set is " + str(performance_val) + "%")  
    
    count_test = 0
    n_test = len(test_real) + len(test_fake)
    for headline in test_real:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_test += 1
    for headline in test_fake:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_test += 1
    performance_test = count_test / float(n_test) * 100
    print("The performance of the Naive Bayes classifer on the test set is " + str(performance_test) + "%")     
    
#=============== PART 3 ================================#
#part 3a

def part3():
    """
    compute P(real|word), P(real|~word), P(fake|word), P(fake|~word), and print the top ten for each
    """
    
    words = np.array([])
    prob_real_word = np.array([])
    prob_real_not_word = np.array([])
    prob_fake_word = np.array([])
    prob_fake_not_word = np.array([])
    
    #compute P(fake|word) and P(fake|~word) for all words
    for word in word_list.keys():
        words = np.append(words, word)
        word = list(word)
        result, prob_fake, prob_absence = Naive_Bayes_classifier(word, word_list, train_real, train_fake, 1, 0.1)
        prob_fake_word = np.append(prob_fake_word, prob_fake)
        prob_fake_not_word = np.append(prob_fake_not_word, prob_absence)
    
    #compute P(real|word) and P(real|~word) for all words
    for i in range(len(prob_fake_word)):
        prob_real_word = np.append(prob_real_word, 1. - prob_fake_word[i])
        prob_real_not_word = np.append(prob_real_not_word, 1. - prob_fake_not_word[i])
    
    real_presence = dict(zip(words, prob_real_word))
    real_absence = dict(zip(words, prob_real_not_word))
    fake_presence = dict(zip(words, prob_fake_word))
    fake_absence = dict(zip(words, prob_fake_not_word))
    
    real_presence = sorted(real_presence.items(), key=operator.itemgetter(1))
    real_absence = sorted(real_absence.items(), key=operator.itemgetter(1))
    fake_presence = sorted(fake_presence.items(), key=operator.itemgetter(1))
    fake_absence = sorted(fake_absence.items(), key=operator.itemgetter(1))
       
    print("List the 10 words whose presence most strongly predicts that the news is real:")
    top10(real_presence)
    print("List the 10 words whose absence most strongly predicts that the news is real:")
    top10(real_absence)
    print("List the 10 words whose presence most strongly predicts that the news is fake:")
    top10(fake_presence)
    print("List the 10 words whose absence most strongly predicts that the news is fake:")
    top10(fake_absence)
    
    #-------- part 3b ----------------------
    
    real_presence = dict(zip(words, prob_real_word))
    real_absence = dict(zip(words, prob_real_not_word))
    fake_presence = dict(zip(words, prob_fake_word))
    fake_absence = dict(zip(words, prob_fake_not_word)) 
    
    for word in ENGLISH_STOP_WORDS:
        if word in real_presence: 
            del real_presence[word]
    for word in ENGLISH_STOP_WORDS:
        if word in real_absence: 
            del real_absence[word]
    for word in ENGLISH_STOP_WORDS:
        if word in fake_presence: 
            del fake_presence[word]    
    for word in ENGLISH_STOP_WORDS:
        if word in fake_absence: 
            del fake_absence[word] 
    
    real_presence = sorted(real_presence.items(), key=operator.itemgetter(1))
    real_absence = sorted(real_absence.items(), key=operator.itemgetter(1))
    fake_presence = sorted(fake_presence.items(), key=operator.itemgetter(1))
    fake_absence = sorted(fake_absence.items(), key=operator.itemgetter(1))    
    
    print("Part3b:")      
    print("List the 10 words whose presence most strongly predicts that the news is real:")
    top10(real_presence)
    print("List the 10 words whose absence most strongly predicts that the news is real:")
    top10(real_absence)
    print("List the 10 words whose presence most strongly predicts that the news is fake:")
    top10(fake_presence)
    print("List the 10 words whose absence most strongly predicts that the news is fake:")
    top10(fake_absence)        

def top10(rank):
    """
    print the last ten word in ascending array.
    """
    i = -1
    while i > -11:    
        print(rank[i])
        i -= 1

    
#=============== PART 4 ================================#
#===============logistic regression=====================# 
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
#Step1: formulate x and Y
def generateXY():    
    training_x = np.zeros((0, num_words))
    validation_x = np.zeros((0, num_words)) 
    test_x = np.zeros((0, num_words))
    
    for headline in training:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        training_x = np.vstack((training_x, i))
    for headline in validation:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        validation_x = np.vstack((validation_x, i))
    for headline in test:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        test_x = np.vstack((test_x, i))    
    
    training_y = np.asarray(training_label)
    training_y_complement = 1 - training_y
    training_y = np.vstack((training_y, training_y_complement)).transpose() 
    
    validation_y = np.asarray(validation_label)
    validation_y_complement = 1 - validation_y
    validation_y = np.vstack((validation_y, validation_y_complement)).transpose() 
    
    test_y = np.asarray(test_label)
    test_y_complement = 1 - test_y
    test_y = np.vstack((test_y, test_y_complement)).transpose()     
    
    return training_x, validation_x, test_x, training_y, validation_y, test_y, num_words

#Step2: model building using pytorch
dtype_float = torch.FloatTensor

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


#Step3: trian
def train_part4(learning_rate, numIterations):
    #generate x,y
    training_x, validation_x, test_x, training_y, validation_y, test_y, num_words = generateXY()
    
    input_size = num_words
    num_classes = 2    
    
    x_data = Variable(torch.from_numpy(training_x), requires_grad=False).type(torch.FloatTensor)
    y_data = Variable(torch.from_numpy(np.argmax(training_y, 1)), requires_grad=False).type(torch.LongTensor)

    model = LogisticRegression(input_size, num_classes)
    
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)     

    for epoch in range(numIterations):
        optimizer.zero_grad()
        y_pred = model(x_data)              
        loss = criterion(y_pred, y_data)      
        loss.backward()
        optimizer.step()  
        
        print(epoch, loss.data[0])

    
    return model
  
#Step4: test
def test_part4(learning_rate, numIterations):
    model = train_part4(learning_rate, numIterations)
    #generate x,y
    training_x, validation_x, x, training_y, validation_y, y, num_words = generateXY()
    
    
    x_test = Variable(torch.Tensor(x)) 
    
    y_test = model(x_test).data.numpy()
    y_test[y_test > 0.5] = 1
    y_test[y_test <= 0.5] = 0
    
    print(np.mean(y_test == y) * 100)
    

#Step5: Plot Learning Curve VS Iteration
def learning_curve():
    
    #generate x,y
    training_x, validation_x, test_x, training_y, validation_y, test_y, num_words = generateXY()
    
    iterations = []
    test_performance = []
    train_performance = []
    validate_performance = []
    
    learning_rate = 0.001
    input_size = num_words
    numIterations = 1000
    num_classes = 2
    
    x_data = Variable(torch.from_numpy(training_x), requires_grad=False).type(torch.FloatTensor)
    y_data = Variable(torch.from_numpy(np.argmax(training_y, 1)), requires_grad=False).type(torch.LongTensor)
    
    #generate test x, y
    x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(torch.FloatTensor) 
    
    #generate validate x, y
    x_validate = Variable(torch.from_numpy(validation_x), requires_grad=False).type(torch.FloatTensor)
     
    #model = Model()
    model = LogisticRegression(input_size, num_classes)
    
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)     
    
    for epoch in range(numIterations):
        optimizer.zero_grad()
        y_pred = model(x_data)              
        loss = criterion(y_pred, y_data)      
        loss.backward()
        optimizer.step()        
        
        #store iterations in list
        iterations.append(epoch)
        if epoch % 100 == 0:
            print("Iterations" + str(epoch))
        #compute train result and performance store in list 
        y_train = model(x_data).data.numpy()
        y_train[y_train >= 0.5] = 1
        y_train[y_train < 0.5] = 0
        accuracyTrain = np.mean(y_train == training_y) * 100    
        train_performance.append(accuracyTrain)
        
        #compute test result and performance store in list 
        y_test = model(x_test).data.numpy()
        y_test[y_test > 0.5] = 1
        y_test[y_test <= 0.5] = 0
        accuracyTest = np.mean(y_test == test_y) * 100    
        test_performance.append(accuracyTest)
        
        #compute validate result and performance store in list 
        y_validate = model(x_validate).data.numpy()
        y_validate[y_validate > 0.5] = 1
        y_validate[y_validate <= 0.5] = 0
        accuracyvalidate = np.mean(y_validate == validation_y) * 100    
        validate_performance.append(accuracyvalidate)  
        
    #plot the learning curve 
    plt.plot(iterations, train_performance, color='blue', label="training set")
    plt.plot(iterations, test_performance, color='green', label="test set")
    plt.plot(iterations, validate_performance, color='red', label="validation set")
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Performance(%)', fontsize=12)
    plt.legend(loc='top left')
    plt.show()
    
#=======================Part 6===========================================#
#Part 6 - a
def part6a():
    learning_rate = 0.001
    numIterations = 10000
    
    model = train_part4(learning_rate, numIterations)
    
    con = []
    for content in model.parameters():
        con.append(content)
    
    
    parameters = con[0].data[0].numpy().tolist()
    
    param_words = dict(zip(word_index.keys(), parameters))
        
    #sort the dictionary by parameter size 
    sorted_param = sorted(param_words.items(), key=operator.itemgetter(1))    
    
    print("List of Top 10 negative thetas:")
    print(sorted_param[:10])
    
    print("List of Top 10 positive thetas:")
    print(sorted_param[-10:])    
    
#filter out all the STOP words
def part6b():
    new_words = list(word_index.keys())
    for word in ENGLISH_STOP_WORDS:
        if word in new_words: 
            new_words.remove(word)
        
    learning_rate = 0.001
    numIterations = 10000

    model = train_part4(learning_rate, numIterations)

    con = []
    for content in model.parameters():
        con.append(content)
    
    
    parameters = con[0].data[0].numpy().tolist()
    
    param_words = dict(zip(new_words, parameters))
        
    #sort the dictionary by parameter size 
    sorted_param = sorted(param_words.items(), key=operator.itemgetter(1))    
    
    print("List of Top 10 negative thetas without STOP WORDS:")
    print(sorted_param[:10])
    
    print("List of Top 10 positive thetas without STOP WORDS:")
    print(sorted_param[-10:])  
    





#=============== PART 7 ================================#

def part7():
    #part7a
    #-----Train the decision tree classifier, find the best parameters-----------
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
    
    max_depth_list = [3, 10, 20, 50, 100, 150, 200, 300, 500, 700]
    
    for depth in max_depth_list:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(training_set, training_label)
        print("Max depth is: " + str(depth))
        print("Training: " + str(100*clf.score(training_set, training_label)) + " Validation: " + str(100*clf.score(validation_set, validation_label)))
        
    #Highest accuracy is achived at max_depth 300
    clf = tree.DecisionTreeClassifier(max_depth=300)
    clf = clf.fit(training_set, training_label)    

    #part7b
    #----------visualize the first two layers of the decision tree
    word = []
    for i in word_index.keys():
        word.append(i)  
    
    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=2, filled=True, rounded=True, class_names=['fake', 'real'], feature_names=word)
    graph = graphviz.Source(dot_data)
    graph.render(view=True)


#=============== PART 8 ================================#

def part8b():
    real_with_word = []
    for headline in train_real:
        if "trumps" in list(headline):
            real_with_word.append(headline)
            
    with_word_real_split = len(real_with_word)
    without_word_real_split = len(train_real) - with_word_real_split
    
    fake_with_word = []
    for headline in train_fake:
        if "trumps" in list(headline):
            fake_with_word.append(headline)
    
    with_word_fake_split = len(fake_with_word)
    without_word_fake_split = len(train_fake) - with_word_fake_split
    
    print("Number of total headlines: " + str(len(train_real) + len(train_fake)))
    print("Number of initial fake headlines: " + str(len(train_fake)))
    print("Number of initial real headlines: " + str(len(train_real)))
    print("\n")
    print("Number of headlines without word 'trumps': " + str(without_word_fake_split + without_word_real_split))
    print("Number of fake headlines in the dataset without word 'trumps': " + str(without_word_fake_split))
    print("Number of real headlines in the dataset without word 'trumps': " + str(without_word_real_split))
    print("\n")
    print("Number of headlines with word 'trumps': " + str(with_word_fake_split + with_word_real_split))
    print("Number of fake headlines in the dataset with word 'trumps': " + str(with_word_fake_split))
    print("Number of real headlines in the dataset with word 'trumps': " + str(with_word_real_split))