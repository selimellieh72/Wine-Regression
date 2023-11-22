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


def z_score_normalization(X):
     # Calculate the mean and standard deviation for each feature
     means = np.mean(X, axis=0)
     stds = np.std(X, axis = 0)
     # Apply the normalization formula: (X - mean) / std
     X_standardized = (X - means) / stds
     print(means)
     return X_standardized, means, stds

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
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p

def predict(X, W, b):
        '''
    Prediction with vectorization for some features x and using weights w and bias b
    '''
        
        p = np.dot(X,W) + b
        return p


def computeCost(X, y, W, b):
     '''
     Given X, Y, w and b. Computes J(w, b) of w and b.
     '''
     m = X.shape[0]
     
     j = 0.0
     for i in range(m):

          f_wb_i = predict(X[i], W, b)
          j_i = pow(y[i] - f_wb_i, 2)
          j = j + j_i
     j = j / (2 * m)
     return j

