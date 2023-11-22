import numpy as np
import matplotlib.pyplot as plt
import csv
import math

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays


def importTrainingData(file_name, m, n):
    '''Import the wine training data, with m elements and n features'''
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
     '''
     Z-score normalization on X (m * n)
     '''
     # Calculate the mean and standard deviation for each feature
     means = np.mean(X, axis=0)
     stds = np.std(X, axis = 0)
     # Apply the normalization formula: (X - mean) / std
     X_normalized = (X - means) / stds
  
     return X_normalized, means, stds

def normalizeElement(X, means, stds):
     '''
     Normalize one element X, for prediction
     '''
     x_normalized = (X - means) / stds
     return x_normalized
     

def scatter(x, y, title, y_label, x_label):
    '''Scatter dots (x, y), as y in function of x'''
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

def computeGradient(X, y, W, b):
     '''
     Compute Gradient dJ/dWj and dJ/db
     '''
     m,n = X.shape
     # Array to hold dj/dw(j)
     dj_dw = np.zeros((n,))
     dj_db = 0
     # Summation from i = 0 to m -1
     for i in range(m):
          e = predict(X[i], W, b) - y[i]
          for j in range(n):
            # e * x[i, j]
            dj_dw[j] = dj_dw[j] + e * X[i, j]
          dj_db = dj_db + e
     dj_dw = dj_dw / m
     dj_db = dj_db / m
     return dj_dw, dj_db

def gradientDescent(X, y, w_in, b_in , learning_rate, num_iters):
     '''
     Perform batch gradient descent `num_iters` times to learn w and b.
     '''
     w = w_in.copy()
     b = b_in

     for i in range(num_iters):
          # Calculate gradient at each step
          dj_dw, dj_db = computeGradient(X, y, w, b)
          w = w - learning_rate * dj_dw
          b = b - learning_rate * dj_db

          if i % math.ceil(num_iters / 10) == 0:
               print(f"Iteration {i:4d}: Cost {computeCost(X, y, w, b):8.2f}")
     return w,b