import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
    neural network for california housing price prediction
    without tensorflow
'''


# cost function
def cost_function(y, t):
    return (y - t) ** 2


# sigmoid function
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


# the differential of sigmoid function
def derivative_sigmoid(x):
    y = x * (1 - x)
    return y


# tanh function
def tanh(x):
    y = np.tanh(x)
    return y


# the differential of tanh function
def derivative_tanh(x):
    y = 1.0 - x ** 2
    return y


# neural network
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        # define the activate function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_der = derivative_sigmoid
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_der = derivative_tanh
        # initialize　the weights
        self.weights = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.normal(0.0, pow(layers[i], -0.5),
                                                 (layers[i - 1], layers[i])))
        # initialize the bias
        self.b = []
        for i in range(1, len(layers)):
            self.b.append(np.zeros(layers[i]))
        # learning rate
        self.lr = 0.1

    def predict(self, x):
        # store the outputs of each layer
        y = [np.array([x])]
        for i in range(len(self.weights) - 1):
            y.append(self.activation(np.dot(y[i], self.weights[i]) + self.b[i]))
        y.append(np.dot(y[-1], self.weights[-1]) + self.b[-1])
        return y

    def loss(self, y, t):
        # calculate the error, return the differential of the error
        # error = 0.5 * (t - y) ** 2
        dout = - (t - y)
        return dout

    def fit(self, x, t):
        # obtain the outputs of each layer by predict method
        y = self.predict(x)
        # use the last of output and objective variable to calculate the error
        error = self.loss(y[-1], t)
        # ----- Backpropagation -----
        # start from the output layer
        dout = error # * self.activation_der(y[2])
        # declare the lists to store the gradients of weights and bias
        dw = []
        db = []
        # Backpropagation reach the hidden layer
        # calculate the gradients of hidden layer
        dw.append(np.dot(y[1].T, dout))
        db.append(np.sum(dout))
        # reach the first layer
        dout = np.dot(dout, self.weights[1].T) * self.activation_der(y[1])
        # calculate the gradients based on errors
        dw.append(np.dot(y[0].T, dout))
        db.append(np.sum(dout))
        # Reverse the gradients　for convenience of calculation
        dw.reverse()
        db.reverse()
        # upgrade the weights and bias via gradients
        # 重みとバイアスの勾配に基づいて、重みとバイアスを更新
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * dw[i]
            self.b[i] -= self.lr * db[i]


# ---------- Read data ----------
# training set
data = pd.read_csv('../california_housing_train.csv')
# Normalize the training data set
data = (data - data.mean()) / data.std()
# Normalize the test data set
test = pd.read_csv('../california_housing_test.csv')
test = (test - test.mean()) / test.std()

# extrac explanatory variable from the training set
X_train = data.iloc[ :, : -1].values
# extac objective variable from the training set
Y_train = data.iloc[: , -1].values

# extrac explanatory variable from the test set
X_test = test.iloc[ :, : -1].values
# extac objective variable from the test set
Y_test = test.iloc[: , -1].values

# Hyperparameters
learning_rate = 0.05

network = NeuralNetwork([8, 10, 1])

X_axis = np.linspace(0, X_test.shape[0], X_test.shape[0])
Y_axis = np.zeros(X_test.shape[0])

for i in range(X_train.shape[0]):
    x = X_train[i]
    y = Y_train[i]

    network.fit(x, y)

for i in range(X_test.shape[0]):
    Y_axis[i] = network.predict(X_test[i])[-1]

# polt the part of the data set
start_index = 50
end_index = 100
plt.plot(range(start_index, end_index), Y_test[start_index: end_index])
plt.plot(range(start_index, end_index), Y_axis[start_index: end_index])

plt.show()
