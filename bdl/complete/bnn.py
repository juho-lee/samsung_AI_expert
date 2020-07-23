import tensorflow as tf
from tensorflow.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from scipy.special import logsumexp

from layers import bayes_dense, bayes_dense_2

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--kl_coeff', type=float, default=1e-2)
parser.add_argument('--num_steps', type=int, default=10000)
parser.add_argument('--test', action='store_true')
parser.add_argument('--num_samples', type=int, default=50)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# generate data
def func(x, noise):
    eps = noise * np.random.randn(*x.shape)
    return x + eps + np.sin(4*(x + eps)) + np.sin(13*(x + eps))

num_train = 50
num_test = 20
np.random.seed(42)
x_train_np = np.concatenate([
    0.5 * np.random.rand(num_train//2, 1),
    0.8 + 0.2*np.random.rand(num_train//2, 1)], 0)
y_train_np = func(x_train_np, 0.03)
x_test_np = np.random.rand(num_test, 1)
y_test_np = func(x_test_np, 0.03)
np.random.seed(int(time.time()))

x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 1])

out, kld1 = bayes_dense(x, 100, activation='relu', name='dense1')
out, kld2 = bayes_dense(out, 100, activation='relu', name='dense2')
out, kld3 = bayes_dense(out, 2, name='dense3')
mu, sigma = tf.split(out, 2, axis=-1)
sigma = tf.nn.softplus(sigma)

# log-likelihood
ll = tf.reduce_mean(Normal(mu, sigma).log_prob(y))
# kl-divergence
kld = args.kl_coeff * (kld1 + kld2 + kld3) / np.float32(num_train)

if not args.test:
    if not os.path.isdir('results/bnn'):
        os.makedirs('results/bnn')

    saver = tf.train.Saver(tf.trainable_variables())
    lr = tf.placeholder(tf.float32)
    train_op = tf.train.AdamOptimizer(lr).minimize(-ll + kld)

    def get_lr(t):
        if t < 0.25 * args.num_steps:
            return args.lr
        elif t < 0.5 * args.num_steps:
            return 0.1 * args.lr
        else:
            return 0.01 * args.lr

    # training loop
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for t in range(args.num_steps):

        sess.run(train_op, {x:x_train_np, y:y_train_np, lr:get_lr(t)})

        if (t+1)% 100 == 0:
            train_ll, train_kld = sess.run([ll, kld], {x:x_train_np, y:y_train_np})
            print('step %d train ll %.4f train kld %.4f' % (t+1, train_ll, train_kld))
            saver.save(sess, os.path.join('results/bnn', 'model'))

    saver.save(sess, os.path.join('results/bnn', 'model'))

else:
    sess = tf.Session()
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, os.path.join('results/bnn', 'model'))

    x_np = np.linspace(0.0, 1.0, 100)[...,None]
    y_true_np = func(x_np, 0.0)

    mu_np_list, sigma_np_list = [], []
    ll_np_list = []
    for i in range(args.num_samples):
        mu_np, sigma_np = sess.run([mu, sigma], {x:x_np})
        mu_np, sigma_np = mu_np.squeeze(), sigma_np.squeeze()
        ll_np = sess.run(Normal(mu, sigma).log_prob(y), {x:x_test_np, y:y_test_np})
        mu_np_list.append(mu_np)
        sigma_np_list.append(sigma_np)
        ll_np_list.append(ll_np)

    ll_np_list = np.stack(ll_np_list, 0).squeeze(-1)
    print('test ll: %.4f' % (logsumexp(ll_np_list, 0) - np.log(args.num_samples)).mean())

    x_np = x_np.squeeze()
    y_true_np = y_true_np.squeeze()
    x_train_np = x_train_np.squeeze()
    y_train_np = y_train_np.squeeze()
    x_test_np = x_test_np.squeeze()
    y_test_np = y_test_np.squeeze()

    plt.figure()
    for i in range(args.num_samples):
        upper = mu_np_list[i].squeeze() + sigma_np_list[i].squeeze()
        lower = mu_np_list[i].squeeze() - sigma_np_list[i].squeeze()
        plt.fill_between(x_np, lower, upper, alpha=0.01, color='skyblue')
        plt.plot(x_np, mu_np_list[i].squeeze(), color='blue', alpha=0.05)

    plt.plot(x_train_np, y_train_np, 'r.', label='train')
    plt.plot(x_test_np, y_test_np, 'b*', label='test')
    plt.plot(x_np, y_true_np, 'k--', label='true')
    plt.legend()
    plt.show()
