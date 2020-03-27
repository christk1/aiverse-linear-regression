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

np.random.seed(5)

data = pd.read_csv('real_estate.csv')
data = data.drop(columns=['No'])

corr = data.corr()
# saleprice correlation matrix
plt.gcf().subplots_adjust(bottom=0.3)
cols = data.columns
cm = np.corrcoef(data[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 yticklabels=cols.values, xticklabels=cols.values)

# plot scatters
fig, axs = plt.subplots(2, 3, squeeze=False)
for i, ax in enumerate(axs.flatten()):
    ax.set_xlabel(data.columns[i])
    ax.set_ylabel('Y house price of unit area')
    ax.set_facecolor('xkcd:charcoal')
    ax.scatter(data.iloc[:, i], data.loc[:,'Y house price of unit area'], color="#DC143C")
plt.show()

# Standardizing the features
data_land = pd.DataFrame(data['X5 latitude'].values * data['X6 longitude'].values, columns=['land'])
scaler = MinMaxScaler(feature_range=(0.1, 1.1))
data_land = scaler.fit_transform(data_land)
data = pd.concat([pd.DataFrame(data_land, columns=['land']), data], axis=1)

data['X1 transaction date'] = data['X1 transaction date'] - 2012

data['X2 house age'] = data['X2 house age'].replace(0, 1e-2)

data = data.drop(columns=['X5 latitude', 'X6 longitude'])

for i in range(data.shape[1]):
    if i == 4:
        continue
    data.iloc[:, i], _ = boxcox(data.iloc[:, i])

# drop outliers
data = data.drop(data['Y house price of unit area'].idxmax())
data = data.drop(data['Y house price of unit area'].idxmin())
data = data.drop(data['land'].idxmax())


# plot scatters
fig, axs = plt.subplots(2, 3, squeeze=False)
for i, ax in enumerate(axs.flatten()):
    ax.set_xlabel(data.columns[i])
    ax.set_ylabel('Y house price of unit area')
    ax.set_facecolor('xkcd:charcoal')
    ax.scatter(data.iloc[:, i], data.loc[:,
                                         'Y house price of unit area'], color="#DC143C")

# plot distributions
fig, axs = plt.subplots(2, 3, squeeze=False)
for i, ax in enumerate(axs.flatten()):
    ax.set_xlabel(data.columns[i])
    ax.set_ylabel('Y house price of unit area')
    ax.set_facecolor('xkcd:charcoal')
    sns.distplot(data.iloc[:, i], ax=ax, fit=norm,
                 color="#DC143C", fit_kws={"color": "#4e8ef5"})

# probability plot
fig1, axs1 = plt.subplots(2, 3, squeeze=False)
for i, ax in enumerate(axs1.flatten()):
    ax.set_xlabel(data.columns[i])
    ax.set_ylabel('Y house price of unit area')
    ax.set_facecolor('xkcd:charcoal')
    stats.probplot(data.iloc[:, i], plot=ax)

plt.show()

corr = data.corr()
# saleprice correlation matrix
plt.gcf().subplots_adjust(bottom=0.3)
cols = data.columns
cm = np.corrcoef(data[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

train, test = train_test_split(data, test_size=0.3)


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
iterations = 12000
alpha = 0.01
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
#plt.show()

r2 = COD(x_norm_test, y_test, theta)
print("COD:", r2)

num_of_params = x_norm_test.shape[1] - 1  # minus constant
num_of_feats = x_norm_test.shape[0]
adj_r2 = adj_r2(num_of_params, num_of_feats, r2)
print("Adjusted r2:", adj_r2)
