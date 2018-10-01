import random
import numpy as np


def build_sets():
    """
    Build training, testing and validation sets with labels 
    From content in clean_fake.txt and clean_real.txt
    PARAMETERS
    ----------
    None
    RETURNS
    -------
    training_set: list of list of strings
        contains headlines broken into words
    validation_set: list of list of strings
        contains headlines broken into words
    testing_set: list of list of strings
        contains headlines broken into words
    training_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in training_set
    validation_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in validation_set
    testing_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in testing_set
    REQUIRES
    --------
    clean_fake.txt and clean_test.txt to be present in working directory
    """
    fname_fake, fname_real = "clean_fake.txt", "clean_real.txt"

    content_fake, content_real = [], []

    with open(fname_fake) as f:
        _content_fake = f.readlines()

        for line in _content_fake:
            content_fake.append(str.split(line))

    with open(fname_real) as f:
        _content_real = f.readlines()

        for line in _content_real:
            content_real.append(str.split(line))

    random.seed(42)

    random.shuffle(content_fake)
    random.shuffle(content_real)

    # Split in sets
    training_set, validation_set, testing_set = [], [], []
    training_label, validation_label, testing_label = [], [], []

    for i in range(len(content_fake)):
        if i < 0.7*len(content_fake):
            training_set.append(content_fake[i])
            training_label.append(0)
        elif i < 0.85*len(content_fake):
            validation_set.append(content_fake[i])
            validation_label.append(0)
        else:
            testing_set.append(content_fake[i])
            testing_label.append(0)

    for i in range(len(content_real)):
        if i < 0.7*len(content_real):
            training_set.append(content_real[i])
            training_label.append(1)
        elif i < 0.85*len(content_real):
            validation_set.append(content_real[i])
            validation_label.append(1)
        else:
            testing_set.append(content_real[i])
            testing_label.append(1)

    return training_set, validation_set, testing_set, training_label, validation_label, testing_label 
