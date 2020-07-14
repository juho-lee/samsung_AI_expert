import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import KFold
from itertools import product

X, y = load_digits(return_X_y=True)
#y[y<5] = -1
#y[y>5] = 1

n_sample = len(X)
n_train = int(0.8*n_sample)
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]

# range of hyperparameters
gamma_range = np.logspace(-9, 3, 10)
C_range = np.logspace(-2, 4, 10)

# K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# fit SVM for all parameters
best_acc = 0.
best_gamma = 0.
best_C = 0.
for gamma, C in product(gamma_range, C_range):

    avg_acc = 0.
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
        clf.fit(X_tr, y_tr)
        avg_acc += clf.score(X_val, y_val)

    avg_acc /= 5.
    print('gamma %.4f C %.4f acc %.4f' % (gamma, C, avg_acc))

    if avg_acc > best_acc:
        best_acc = avg_acc
        best_gamma = gamma
        best_C = C

print('best gamma: %.4f' % (best_gamma))
print('best C: %.4f' % (best_C))

clf = svm.SVC(kernel='rbf', gamma=best_gamma, C=best_C)
clf.fit(X_train, y_train)
print('test acc: %.4f' % (clf.score(X_test, y_test)))
