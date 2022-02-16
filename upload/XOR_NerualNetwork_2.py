import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        # パラメータによって、活性化関数を定義
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_der = derivative_sigmoid
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_der = derivative_tanh
        # 重みの初期化
        self.weights = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.normal(0.0, pow(layers[i], -0.5),
                                                 (layers[i - 1], layers[i])))
        # バイアスの初期化
        self.b = []
        for i in range(1, len(layers)):
            self.b.append(np.zeros(layers[i]))
        # 学習率
        self.lr = 0.2

    def predict(self, x):
        # 各層における出力も格納する
        y = [np.array([x])]
        for i in range(len(self.weights)):
            y.append(self.activation(np.dot(y[i], self.weights[i]) + self.b[i]))
        return y

    def loss(self, y, t):
        # 誤差を計算、戻り値はその微分
        # error = 0.5 * (t - y) ** 2
        dout = - (t - y)
        return dout

    def fit(self, x, t):
        # predict関数を通して、各層の出力及び最終出力を得る。
        y = self.predict(x)
        # 最終の出力及び観測値で、逆伝搬の誤差を計算
        error = self.loss(y[-1], t)
        # 出力層を通す、逆伝搬
        dout = error * self.activation_der(y[2])
        # 重み及びバイアスの勾配を格納する配列を用意
        dw = []
        db = []
        # 逆伝搬が二層目に到達、その層の重みとバイアスの勾配を計算
        dw.append(np.dot(y[1].T, dout))
        db.append(np.sum(dout))
        # 続いて、前の層に伝搬、一層目に到着
        dout = np.dot(dout, self.weights[1].T) * self.activation_der(y[1])
        # 伝播された誤差に基づいて、その層の重みとバイアスの勾配を計算
        dw.append(np.dot(y[0].T, dout))
        db.append(np.sum(dout))
        # 計算の便利のため、重みとバイアスの勾配を逆転して、重みとバイアスの順番に合わせる
        dw.reverse()
        db.reverse()
        # 重みとバイアスの勾配に基づいて、重みとバイアスを更新
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * dw[i]
            self.b[i] -= self.lr * db[i]


nn = NeuralNetwork([2, 2, 1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])
for i in range(10000):
    k = np.random.randint(X.shape[0])
    nn.fit(X[k], Y[k])
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, nn.predict(i)[-1])