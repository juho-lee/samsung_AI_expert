import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(
        n_samples=60,
        n_features=1,
        noise=10,
        random_state=42)

# put 1 for bias
X = np.concatenate([np.ones((60, 1)), X], -1)

X_train, y_train = X[:50], y[:50]
X_test, y_test = X[50:], y[50:]

# linear regression
### fill in this part ###
w =

# compute test error
pred_test = np.dot(X_test, w)
test_err = ((y_test - pred_test)**2).mean()
print('MSE: %.4f' % (test_err))

# plot
X_ = np.linspace(X.min(), X.max(), 100)
X_ = np.concatenate([np.ones((100, 1)), X_[:,None]], -1)
plt.plot(X_[:,1], np.dot(X_, w), 'k')
plt.scatter(X_train[:,1], y_train, edgecolor='b', facecolor='white', label='train')
plt.scatter(X_test[:,1], y_test, edgecolor='r', facecolor='white', label='test')
plt.legend()
plt.show()
