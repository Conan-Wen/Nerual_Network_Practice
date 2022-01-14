import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        # NeuralNetworkのクラス関数，
        # それを基づいてNeuralNetworkのインスタンスを作る
        # _init_はコンストラクタ
        """
        :param layers: list型変数，層ごとのニューラル数を指示，
                少なくとも二層、入力層と出力層
        :param activation: 活性化関数
                活性化関数はtanh と logistics二つある，
                デフォルトは、tanh関数
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # weights重みの初期化,
        self.weights = []
        # len(layers)は層の数を表す。
        # 出力層を除き、各層に重みを与える
        for i in range(1, len(layers) - 1):
            # np.random.randomは、nunpyによる乱数
            # 重みは、[-0.25,0.25]の間にある。
            temp = 1
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * temp)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * temp)

    # ニューラルネットワークの学習メソッド
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # X: 特徴値、二次元で、
        X = np.atleast_2d(X)
        # np.ones各要素が 1 の行列を作る
        # X.shapeは、該当の行数、列数を返す。戻り値はリストで[行数，列数]
        # X.shape[0]は行数，X.shape[1]+1：一個要素追加で，biasに1を与える
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        # “ ：”全ての要素を取る。
        # “0：-1”、一つ目の要素から後ろ二つ目の要素まで、一番最後の要素を取らない
        temp[:, 0:-1] = X
        X = temp
        # y：classlabel，観測値
        y = np.array(y)
        # epochs回で学習を行う
        for k in range(epochs):
            # Xより、ランダムに一つのインスタンスを取り出して学習
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            # 順伝搬で、予測値を計算，len(self.weights)
            for l in range(len(self.weights)):
                # np.dotは二つの内積を計算，x.dot(y) イコール np.dot(x,y)
                # aとweightsの内積を計算し，活性化関数を通し、次の層に伝搬する
                # a各層の入力を格納，appendによって各層の入力を追加不断、最後は出力を追加
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            # 誤差を計算，y[i]は観測値，予測値はa[-1]に格納される
            error = y[i] - a[-1]
            # 逆伝搬で、各層の誤差を計算
            deltas = [error * self.activation_deriv(a[-1])]

            # 逆伝搬で、重みを更新
            # len(a)ニューラルの層数，入力層と出力層は計算しない
            # 从最后一层到第0层，每次-1
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            # reverseで、deltasを逆転
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                # layer.T.dot(delta)誤差内積、重みを更新
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

nn = NeuralNetwork([2, 2, 1], 'logistic')
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
nn.fit(X, y)
for i in [[0,0], [0,1], [1,0], [1,1]]:
    print(i, nn.predict(i))