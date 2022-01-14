import numpy as np


# 損失関数
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# softmax関数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# sigmoid関数
def sigmoid(a):
    y = 1 / (1 + np.exp(-a))
    return y


# tanh関数
def tanh(a):
    return np.tanh(a)


#  勾配法による重みの更新
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = np.random.normal(0.0, pow(hidden_size, -0.5), (input_size, input_size))
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.normal(0.0, pow(output_size, -0.5), (hidden_size, output_size))
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = tanh(a1)
        a2 = np.dot(z1, W2) + b2
        y = tanh(a2)

        return y

    # x: 入力データ、t: 教師データ
    def loss(self, x, t):
        y = self.predict(x)

        return (y - t) ** 2
        # return cross_entropy_error(y, t)

    # def accuracy(self, x, t):
    #     y = self.predict()

    # x: 入力データ、t: 教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
Y = np.array([0, 1, 1, 0])

# ハイパーパラメータ
iters_num = 1000
learning_rate = 0.1

network = TwoLayerNet(input_size=2, hidden_size=2, output_size=1)

for i in range(iters_num):
    # ランダムにデータを一個抽出
    tmp = np.random.randint(X.shape[0])
    x = X[tmp]
    y = Y[tmp]

    # 勾配の計算
    grad = network.numerical_gradient(x, y)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, network.predict(i))

