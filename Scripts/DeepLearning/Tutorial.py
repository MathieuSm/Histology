"""
Tutorial deep learning python script adapted from
https://heartbeat.comet.ml/building-a-neural-network-from-scratch-using-python-part-1-6d399df8d432
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CurrentDirectory = os.getcwd()
DataDirectory = os.path.join(CurrentDirectory,'Scripts/DeepLearning/')

# Define Activation function and its derivative
def ReLU(x):
    '''
    The ReLu activation function is to performs a threshold
    operation to each input element where values less
    than zero are set to zero.
    '''
    return np.maximum(0, x)
def dReLU(x):

    """
    Return the derivative of ReLU activation function
    """

    y = x > 1
    y = y.replace(True,1)
    y = y.replace(False,0)

    return y

# Define Output function
def Sigmoid(x):
    '''
    The sigmoid function takes in real numbers in any range and
    squashes it to a real-valued output between 0 and 1.
    '''
    return 1 / (1 + np.exp(-x))

# Define Loss function
def EntropyLoss(y,y_hat):

    NSample = len(y)

    # if predicted value is 0, replace with small value for log computation
    if sum(y_hat == 0) > 0:
        y_hat[y_hat == 0] = 1E-10

    a = y * np.log(y_hat)
    b = (1.0 - y) * np.log(1.0 - y_hat)

    Loss = -1/NSample * np.sum(a + b)

    return Loss

# Define Forward propagation
def ForwardPropagation(X,Y,W1,b1,W2,b2):

    '''
    Performs the forward propagation
    '''

    Z1 = X.dot(W1) + b1
    A1 = ReLU(Z1)
    Z2 = A1.dot(W2) + b2
    y_hat = Sigmoid(Z2)
    Loss = EntropyLoss(Y, y_hat)

    return Z1, A1, Z2, y_hat, Loss

# Define Backward propagation
def BackwardPropagation(X, Y, y_hat, A1, Z1, W1, b1, W2, b2, LR):

    '''
    Computes the derivatives and update weights and bias accordingly.
    '''

    # if predicted value is 0, replace with small value for log computation
    if sum(y_hat == 0) > 0:
        y_hat[y_hat == 0] = 1E-10

    # Derivative of Loss function with respect to Z2
    dLoss_dy_hat = (1-Y) / (1-y_hat) - Y / y_hat
    dLoss_dSigmoid = y_hat * (1 - y_hat)
    dLoss_dZ2 = dLoss_dy_hat * dLoss_dSigmoid

    # Derivative of Loss function with respect to W2 and b2
    dLoss_dA1 = dLoss_dZ2.dot(W2.T)
    dLoss_dW2 = A1.T.dot(dLoss_dZ2)
    dLoss_db2 = np.sum(dLoss_dZ2, axis=0)

    # Derivative of Loss function with respect to W1 and b1
    dLoss_dZ1 = dLoss_dA1 * dReLU(Z1)
    dLoss_dW1 = X.T.dot(dLoss_dZ1)
    dLoss_db1 = np.sum(dLoss_dZ1, axis=0)

    # Update the weights and bias
    W1 = W1 - LR * dLoss_dW1
    W2 = W2 - LR * dLoss_dW2
    b1 = b1 - LR * dLoss_db1
    b2 = b2 - LR * dLoss_db2

    return W1, b1, W2, b2


# Load data of https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29 and set headers
Headers = ['Age', 'Sex','Chest pain','Resting blood pressure',
           'Serum cholestoral', 'Fasting blood sugar', 'Resting ecg results',
           'Max heart rate achieved', 'Exercise induced angina', 'Old peak',
           'Slope of the peak', 'Number of major vessels','Thal', 'Heart disease']
DataFrame = pd.read_csv(DataDirectory + 'Heart.dat', sep=' ', names=Headers)

# Replace target class with 0 and 1, 1 means "have heart disease" and 0 means "do not have heart disease"
DataFrame['Heart disease'] = DataFrame['Heart disease'].replace(1, 0)
DataFrame['Heart disease'] = DataFrame['Heart disease'].replace(2, 1)

# Select randomly 20% of the data set for test data
Test = DataFrame.sample(round(len(DataFrame)*0.2))
Train = DataFrame.drop(Test.index)

# Build X and Y matrices
X_Test = Test.drop(columns=['Heart disease'])
X_Train = Train.drop(columns=['Heart disease'])

Y_Test = Test['Heart disease'].values.reshape(X_Test.shape[0], 1)
Y_Train = Train['Heart disease'].values.reshape(X_Train.shape[0], 1)

# Standardize X matrices
X_Test = (X_Test - X_Test.mean()) / X_Test.std()
X_Train = (X_Train - X_Train.mean()) / X_Train.std()

# Initialize weights and biases for 2 layers (3-1) neural network
Layers = [DataFrame.shape[1]-1,8,1]

W1 = np.random.randn(Layers[0],Layers[1])
b1 = np.random.randn(Layers[1])
W2 = np.random.randn(Layers[1],Layers[2])
b2 = np.random.randn(Layers[2])

# Perform neural training
Iterations = 100
LR = 0.001  # Learning rate of the neural network
Losses = pd.DataFrame()
for Iteration in range(Iterations):

    # Perform forward propagation
    Z1, A1, Z2, y_hat, Loss = ForwardPropagation(X_Train,Y_Train,W1,b1,W2,b2)

    # Store loss
    Losses = Losses.append(Loss,ignore_index=True)

    # Perform backward propagation
    W1, b1, W2, b2 = BackwardPropagation(X_Train, Y_Train, y_hat, A1, Z1, W1, b1, W2, b2, LR)

Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
Axes.plot(Losses[::10],marker='o',fillstyle='none',linestyle='--',color=(1,0,0))
Axes.set_xlabel('Iteration (-)')
Axes.set_ylabel('Loss (-)')
plt.show()

# Perform prediction on test data
Z1_Test = X_Test.dot(W1) + b1
A1_Test = ReLU(Z1_Test)
Z2_Test = A1_Test.dot(W2) + b2
Prediction = np.round(Sigmoid(Z2_Test))

# Compute accuracy of the prediction
Train_Accuracy = int(np.sum(Y_Train == np.round(y_hat)) / len(Y_Train) * 100)
Test_Accuracy = int(np.sum(Y_Test == Prediction) / len(Y_Test) * 100)


