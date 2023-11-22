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
    #print(x_train[:, 0])
    # Normalize
    x_train_normalized, means, stds = z_score_normalization(x_train)
    print("Normalizing x-train....")
    print("Now, mean =", np.mean(x_train_normalized))
    print("Now, std =", np.std(x_train_normalized))
    b_init = np.random.rand() * 0.01
    w_init = np.random.rand(N) * 0.01
 
    print("b_init value:", b_init)
    print("w_init value:", w_init)
    
    x_vec = x_train_normalized[0, :]
    print("x_vec value:", x_vec)
    f_wb = predictSingleLoop(x_vec, w_init, b_init)
    print("prediction f_wb:", f_wb)

    print("Thus, initialliy cost at optimal w is:", computeCost(x_train_normalized, y_train, w_init, b_init))




if __name__ == "__main__":
    main()