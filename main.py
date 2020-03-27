import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox
from scipy.stats import norm
from functions import gradientDescent, ZNorm, COD, adj_r2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

np.random.seed(5)

data = pd.read_csv('real_estate.csv')

train, test = train_test_split(data, test_size=0.2)


x_train = train[train.columns.difference(['Y house price of unit area'])]
y_train = train.loc[:, 'Y house price of unit area']
x_test = test[test.columns.difference(['Y house price of unit area'])]
y_test = test.loc[:, 'Y house price of unit area']
train_len = len(y_train)
test_len = len(y_test)

(x_norm_train, mi, sigma) = ZNorm(x_train)
x_norm_test = (x_test - mi) / sigma

x_norm_train = np.column_stack((np.ones(train_len), x_norm_train))
x_norm_test = np.column_stack((np.ones(test_len), x_norm_test))

theta = (0.05 * np.random.randn(x_norm_train.shape[1], 1)).squeeze()
# =================== Cost and Gradient descent ===================
iterations = 10000
alpha = 0.001
alpha_increase = 5000

theta, history_train, history_test = gradientDescent(
    x_norm_train, x_norm_test, y_train, y_test, theta, alpha, iterations, alpha_increase)
fig, ax = plt.subplots()
ax.set_xlabel('Iterations')
ax.set_ylabel('J(Θ)')
ax.set_facecolor('xkcd:charcoal')
ax.set_ylim(0, 100)
ax.plot(history_train, color="#DC143C", label='train')
ax.plot(history_test, color="blue", label='test')
ax.legend(loc="upper right")

print("^^^^^^^^^^^^")
print("train:", history_train[-1])
print("test", history_test[-1])
plt.show()

r2 = COD(x_norm_test, y_test, theta)
print("COD:", r2)

num_of_params = x_norm_test.shape[1] - 1  # minus constant
num_of_feats = x_norm_test.shape[0]
adj_r2 = adj_r2(num_of_params, num_of_feats, r2)
print("Adjusted r2:", adj_r2)
