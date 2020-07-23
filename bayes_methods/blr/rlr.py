import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(
        n_samples=30,
        n_features=1,
        noise=30,
        random_state=42)

# put 1 for bias
X = np.concatenate([np.ones((30, 1)), X], -1)

X_train, y_train = X[:5], y[:5]
X_val, y_val = X[5:10], y[5:10]
X_test, y_test = X[10:], y[10:]

def ridge_regession(X, y, lamb):
    ### fill in this part ###
    w =
    return w

lamb_list = np.linspace(0, 5, 30)
val_errs = []
best_lamb = 0
best_err = np.inf
for lamb in lamb_list:
    ### fill in this part ###
    # find the best lambda w.r.t. the MSE #

plt.figure('lambda vs mse')
plt.plot(lamb_list, val_errs)
print('best lamb: %.4f' % best_lamb)

w = ridge_regession(X_train, y_train, best_lamb)
pred_test = np.dot(X_test, w)
test_err = ((y_test-pred_test)**2).mean()
print('test mse: %.4f' % test_err)

## plot
plt.figure('prediction')
X_ = np.linspace(X.min(), X.max(), 100)
X_ = np.concatenate([np.ones((100, 1)), X_[:,None]], -1)

w_lr = ridge_regession(X_train, y_train, 0.0)
plt.plot(X_[:,1], np.dot(X_, w_lr), 'k', label='linear regression')
plt.plot(X_[:,1], np.dot(X_, w), 'm', label='ridge regression')
plt.scatter(X_train[:,1], y_train, edgecolor='b', facecolor='white', label='train')
plt.scatter(X_test[:,1], y_test, edgecolor='r', facecolor='white', label='test')
plt.legend()
plt.show()
