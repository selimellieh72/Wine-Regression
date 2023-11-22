from helpers import *


FILE_NAME = "winequality-red.csv"
M = 1600
N = 11

def main():
    x_train, y_train = importTrainingData(FILE_NAME, M, N)
    print("type of x_train:", type(x_train))
    print("First five elements of x_train with all features:", x_train[:5])
    print("type of y_train:", type(y_train))
    print("First five elements of y_train", y_train[:5])
    print("The shape of x train", x_train.shape)
    print("The shape of  y_train", y_train.shape)
    print("Number of training examples (m)", len(x_train))
    print(y_train)
    # scatter(x_train[:, 0], y_train, "Quality vs Acidity", "Quality of Wine", "Fixed acidity")
    print(x_train[:, 0])
    b_init = np.random.rand()
    w_init = np.random.rand(N)
    




if __name__ == "__main__":
    main()