import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Read data ----------
# training set
data = pd.read_csv('./california_housing_train.csv')
# test set
test = pd.read_csv('./california_housing_test.csv')

# extrac explanatory variable from the training set
X_train = data.iloc[:, : -1]
# extac objective variable from the training set
Y_train = data.iloc[:, -1]

# extrac explanatory variable from the test set
X_test = test.iloc[:, : -1]
# extac objective variable from the test set
Y_test = test.iloc[:, -1]


# ---------- learning by training set ----------
# X^t * X
X_trainTX_train = X_train.T.dot(X_train)
# (X^t * X)^-1
X_trainTX_train_inv = np.linalg.inv(X_trainTX_train)
# (X^t * X)^-1 * X^T
X_trainTX_train_invX_trainT = X_trainTX_train_inv.dot(X_train.T)
# beta = (X^t * X)^-1 * X^T * Y
# beta is parameters we got from learning
beta = X_trainTX_train_invX_trainT.dot(Y_train)

# ---------- Predict test data by model ----------
Y_pre = beta.dot(X_test.T)

# ---------- Evaluate the model by coefficient of determination ----------
Y_train_pre = beta.dot(X_train.T)
# Residual sum of squares
S_res = np.power(Y_train - Y_train_pre, 2).sum()
# Total Sum of Squares
S_all = np.power(Y_train - Y_train.mean(), 2).sum()
# coefficient of determination　　R2
R_2 = 1 - (S_res / S_all)
# adjusted coefficient of determination　　R2
adjR_2 = 1 - (1 - R_2)*((Y_train.size - 1) / (Y_train.size - beta.size - 1))

# ---------- pirnt the result ----------
print(R_2)
print(adjR_2)
print(beta)
# polt the part of the data set
start_index = 100
end_index = 200
plt.plot(range(start_index, end_index), Y_test[start_index: end_index])
plt.plot(range(start_index, end_index), Y_pre[start_index: end_index])
plt.show()
