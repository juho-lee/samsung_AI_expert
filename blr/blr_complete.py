import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from itertools import product

X, y = datasets.make_regression(
        n_samples=30,
        n_features=1,
        noise=30,
        random_state=42)

# put 1 for bias
X = np.concatenate([np.ones((30, 1)), X], -1)

X_train, y_train = X[:10], y[:10]
X_test, y_test = X[10:], y[10:]

def posterior(X, y, alpha, beta):
    S = np.linalg.inv(alpha*np.eye(X.shape[1]) + beta*np.dot(X.T, X))
    m = beta*np.dot(S, np.dot(X.T, y))
    return m, S

def log_marginal_likelihood(X, y, alpha, beta):
    m, S = posterior(X, y, alpha, beta)
    s, logdetS = np.linalg.slogdet(S)
    logdetS *= s
    out = 0.5*logdetS + 0.5*X.shape[1]*np.log(alpha)  \
            + 0.5*X.shape[0]*(np.log(beta) - np.log(2*np.pi)) \
            + 0.5*np.dot(m.T, np.dot(np.linalg.inv(S), m)) \
            - 0.5*beta*np.dot(y.T, y)
    return out

alpha_list = np.logspace(-5, 1, 50)
beta_list = np.logspace(-5, 1, 50)

best_alpha, best_beta = 0, 0
best_lm = -np.inf
lms = []

for alpha, beta in product(alpha_list, beta_list):
    lm = log_marginal_likelihood(X_train, y_train, alpha, beta)
    lms.append(lm)
    if lm > best_lm:
        best_lm = lm
        best_alpha = alpha
        best_beta = beta

AB = np.array(list(product(alpha_list, beta_list)))
plt.figure('alpha, beta vs log-marginal likelihood')
plt.contour(alpha_list, beta_list, np.array(np.exp(lms)).reshape(50, 50))
plt.xscale('log')
plt.yscale('log')

print('best (alpha, beta): %.4f, %.4f' % (best_alpha, best_beta))
print('best log-marginal likelihood: %.4f' % (best_lm))

def predict(X_train, y_train, alpha, beta, X_test):
    m, S = posterior(X_train, y_train, alpha, beta)
    mu = np.dot(X_test, m)
    sigma = np.sqrt(1./beta + ((X_test*np.dot(X_test, S))**2).sum(1))
    return mu, sigma

plt.figure('prediction')
X_ = np.linspace(X.min(), X.max(), 100)
X_ = np.concatenate([np.ones((100, 1)), X_[:,None]], -1)
mu, sigma = predict(X_train, y_train, best_alpha, best_beta, X_)
plt.fill_between(X_[:,1], mu-sigma, mu+sigma, alpha=0.3)
plt.plot(X_[:,1], mu)
plt.scatter(X_train[:,1], y_train, edgecolor='b', facecolor='white', label='train')
plt.scatter(X_test[:,1], y_test, edgecolor='r', facecolor='white', label='test')
plt.legend()
plt.show()
