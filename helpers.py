import numpy as np
import matplotlib.pyplot as plt
import csv

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays


def importTrainingData(file_name, m, n):
    X_train = np.zeros((m, n))
    Y_train = np.zeros((m,))
    with open(file_name) as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for i, row in enumerate(reader):
            for j in range(n):
                X_train[i][j] = row[j]
            Y_train[i] = row[n]
    return X_train, Y_train



def scatter(x, y, title, y_label, x_label):
    plt.scatter(x, y, marker='x', c='r')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

def predictSingleLoop(x, w, b):
    '''
    Prediction for some features x and using weights w and bias b
    '''
    n = x.shape(0)
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p

def predict(x, w, b):
        '''
    Prediction with vectorization for some features x and using weights w and bias b
    '''
        p = np.dot(x,w) + b
        return p