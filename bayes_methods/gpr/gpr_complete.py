import numpy as np
from scipy.stats import multivariate_normal as mvn, norm
from itertools import product
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)
X = rng.uniform(0, 5, 30)[:,None]
y = 0.5 * np.sin(3*X[:,0]) + rng.normal(0, 0.4, X.shape[0])

X_train, y_train = X[:10], y[:10]
X_test, y_test = X[10:], y[10:]

def RBFKernel(X1, X2, sigma):
    pdist = (X1[...,None,:] - X2[None])**2
    return np.exp(-0.5*pdist.sum(-1)/sigma**2)

def log_marginal_likelihood(X, y, sigma, beta):
    Sigma = RBFKernel(X, X, sigma) + 1./beta * np.eye(X.shape[0])
    mu = np.zeros(X.shape[0])
    return mvn.logpdf(y, mean=mu, cov=Sigma)

def predict(X_train, y_train, sigma, beta, X_test):
    k = RBFKernel(X_test, X_train, sigma)
    iK = np.linalg.inv(RBFKernel(X_train, X_train, sigma) + 1./beta * np.eye(X_train.shape[0]))
    mu = np.dot(np.dot(k, iK), y_train)
    k1 = np.diag(RBFKernel(X_test, X_test, sigma))
    sigma = k1 - np.diag(np.dot(np.dot(k, iK), k.T)) + 1./beta
    return mu, sigma

def pred_ll(y_test, mu, sigma):
    return norm.logpdf(y_test, loc=mu, scale=sigma).mean()

plt.figure('prediction with random parameters')
X_ = np.linspace(X.min(), X.max(), 100)[...,None]
mu, sigma = predict(X_train, y_train, 1.0, 1.0, X_)
plt.fill_between(X_[:,0], mu-sigma, mu+sigma, alpha=0.3)
plt.plot(X_[:,0], mu)
plt.scatter(X_train[:,0], y_train, edgecolor='b', facecolor='white', label='train')
plt.scatter(X_test[:,0], y_test, edgecolor='r', facecolor='white', label='test')
plt.legend()
mu, sigma = predict(X_train, y_train, 1.0, 1.0, X_test)
print('Pred ll %.4f' % pred_ll(y_test, mu, sigma))

sigma_list = np.logspace(-2, 2, 100)
beta_list = np.logspace(-2, 2, 100)

best_sigma, best_beta = 0, 0
best_lm = -np.inf
lms = []
for sigma, beta in product(sigma_list, beta_list):
    lm = log_marginal_likelihood(X_train, y_train, sigma, beta)
    lms.append(lm)
    if lm > best_lm:
        best_lm = lm
        best_sigma = sigma
        best_beta = beta

AB = np.array(list(product(sigma_list, beta_list)))
plt.figure('sigma, beta vs log-marginal likelihood')
plt.contour(sigma_list, beta_list, np.array(np.exp(lms)).reshape(100, 100))
plt.xscale('log')
plt.yscale('log')

print('best (sigma, beta): %.4f, %.4f' % (best_sigma, best_beta))
print('best log-marginal likelihood: %.4f' % (best_lm))

plt.figure('prediction with tuned parameters')
X_ = np.linspace(X.min(), X.max(), 100)[...,None]
mu, sigma = predict(X_train, y_train, best_sigma, best_beta, X_)
plt.fill_between(X_[:,0], mu-sigma, mu+sigma, alpha=0.3)
plt.plot(X_[:,0], mu)
plt.scatter(X_train[:,0], y_train, edgecolor='b', facecolor='white', label='train')
plt.scatter(X_test[:,0], y_test, edgecolor='r', facecolor='white', label='test')
plt.legend()
mu, sigma = predict(X_train, y_train, best_sigma, best_beta, X_test)
print('Pred ll %.4f' % pred_ll(y_test, mu, sigma))

plt.show()
