import numpy as np
from scipy.stats import multivariate_normal as mvn, norm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

rng = np.random.RandomState(42)
X = rng.uniform(0, 5, 30)[:,None]
y = 0.5 * np.sin(3*X[:,0]) + rng.normal(0, 0.4, X.shape[0])

X_train, y_train = X[:10], y[:10]
X_test, y_test = X[10:], y[10:]

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X_train, y_train)
X_ = np.linspace(X.min(), X.max(), 100)[...,None]
mu, cov = gp.predict(X_, return_cov=True)
sigma = np.sqrt(np.diag(cov))
print('fitted (sigma, beta): %.4f, %.4f' \
        % (gp.kernel_.k1.length_scale, 1./gp.kernel_.k2.noise_level))
print('optimized log-marginal-likelihood: %.4f' % gp.log_marginal_likelihood(gp.kernel_.theta))

plt.fill_between(X_[:,0], mu-sigma, mu+sigma, alpha=0.3)
plt.plot(X_[:,0], mu)
plt.scatter(X_train[:,0], y_train, edgecolor='b', facecolor='white', label='train')
plt.scatter(X_test[:,0], y_test, edgecolor='r', facecolor='white', label='test')
plt.legend()

mu, cov = gp.predict(X_test, return_cov=True)
sigma = np.sqrt(np.diag(cov))
print('Pred ll %.4f' % norm.logpdf(y_test, loc=mu, scale=sigma).mean())

plt.show()
