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
    b_init = 0
    w_init = np.zeros((N, ))
 
    print("b_init value:", b_init)
    print("w_init value:", w_init)
    
    x_vec = x_train_normalized[0, :]
    print("x_vec value:", x_vec)
    f_wb = predictSingleLoop(x_vec, w_init, b_init)
    print("prediction f_wb:", f_wb)

    print("Thus, intially cost at w is:", computeCost(x_train_normalized, y_train, w_init, b_init))
    iterations = 3000
    alpha = 5.0e-7
    print(f"Performing batch gradient descent with iterations = {iterations} and learning rate = {alpha} ")
    w_optimal, b_optimal = gradientDescent(x_train_normalized, y_train, w_init, b_init, 0.01, iterations)
    print(f"Optimal w found by batch gradient descent is {w_optimal}!")
    print(f"Optimal b found by batch gradient descent is {b_optimal}!")
    print("Recall, x_vec value:", x_vec)
    f_wb = predictSingleLoop(x_vec, w_optimal, b_optimal)
    print("prediction f_wb using optimal w and b:", f_wb)
    print("Try to predict this new wine: ")
    '''
    Predict wine with characteristics:
    - Fixed Acidity: 8.5
    - Volatile Acidity: 0.45
    - Citric Acid: 0.3
    - Residual Sugar: 2.5
    - Chlorides: 0.08
    - Free Sulfur Dioxide: 20
    - Total Sulfur Dioxide: 75
    - Density: 0.9968
    - pH: 3.4
    - Sulphates: 0.65
    - Alcohol: 10.5
    '''
    x_pred = np.array([8.5,0.45,0.3,2.5,0.08,20,75,0.9968,3.4,0.65,10.5])
    print("Predicing x_pred =", x_pred)
    x_pred_normalized =normalizeElement(x_pred, means, stds)
    f_wb = predictSingleLoop(x_pred_normalized, w_optimal, b_optimal)
    print("prediction f_wb using optimal w and b:", f_wb)


if __name__ == "__main__":
    main()