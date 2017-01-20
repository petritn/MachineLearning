import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Objects.CAdaline import AdalineGD, AdalineSGD

# Getting Iris data again...
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# Plotting the learning curve for adaptive linear neuron...
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)

ax[0, 0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0, 0].set_xlabel('Epochs')
ax[0, 0].set_ylabel('log(Sum-Squared-Error)')
ax[0, 0].set_title('Adaline - learning rate 0.01')
ax[0, 0].grid()

ada2 = AdalineGD(n_iter=10, eta = 0.0001).fit(X, y)
ax[0, 1].plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker='o')
ax[0, 1].set_xlabel('Epochs')
ax[0, 1].set_ylabel('log(Sum-Squared-Error)')
ax[0, 1].set_title('Adaline - learning rate 0.0001')
ax[0, 1].grid()

# Now second part of the exercise

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()

ada3 = AdalineGD(n_iter=15, eta=0.01)
ada3.fit(X_std, y)

ax[1, 0].plot(range(1, len(ada3.cost_)+1), np.log(ada3.cost_), marker='o')
ax[1, 0].set_xlabel('Epochs')
ax[1, 0].set_ylabel('log(Sum-Squared-Error)')
ax[1, 0].set_title('Adaline - learning rate 0.01 with standardized features')
ax[1, 0].grid()

# Now the exercise part with AdalineSGD - show convergence of algorithm

adasgd = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)
adasgd.fit(X_std, y)

ax[1, 1].plot(range(1, len(adasgd.cost_)+1), adasgd.cost_, marker='o')
ax[1, 1].set_xlabel('Epochs')
ax[1, 1].set_ylabel('Average Cost')
ax[1, 1].set_title('Adaline - Stochastic Gradient Descent')
ax[1, 1].grid()

plt.show()