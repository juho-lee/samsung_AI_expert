import tensorflow as tf
from tensorflow.distributions import Normal, kl_divergence

def dense(x, num_units, name='dense', activation=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [x.shape[1], num_units])
        b = tf.get_variable('b', [num_units],
                initializer=tf.zeros_initializer())
        x = tf.matmul(x, W) + b
        if activation == 'relu':
            x = tf.nn.relu(x)
        return x

def bayes_dense(x, num_units, name='dense', gamma=1.0, activation=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W_mu = tf.get_variable('W_mu', [x.shape[1], num_units])
        W_rho = tf.nn.softplus(
                tf.get_variable('W_rho', [x.shape[1], num_units],
                    initializer=tf.random_uniform_initializer(-3., -2.)))
        b_mu = tf.get_variable('b_mu', [num_units],
                initializer=tf.zeros_initializer())
        b_rho = tf.nn.softplus(
                tf.get_variable('b_rho', [num_units],
                    initializer=tf.random_uniform_initializer(-3., -2.)))

        # sample
        W = W_mu + W_rho * tf.random.normal(W_mu.shape)
        b = b_mu + b_rho * tf.random.normal(b_mu.shape)

        x = tf.matmul(x, W) + b
        if activation == 'relu':
            x = tf.nn.relu(x)

        # kl divergence
        kld_W = tf.reduce_sum(kl_divergence(Normal(W_mu, W_rho), Normal(0., gamma)))
        kld_b = tf.reduce_sum(kl_divergence(Normal(b_mu, b_rho), Normal(0., gamma)))
        kld = kld_W + kld_b

        return x, kld

def bayes_dense_2(x, num_units, name='dense', gamma=1.0, activation=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W_mu = tf.get_variable('W_mu', [x.shape[1], num_units])
        W_rho = tf.nn.softplus(
                tf.get_variable('W_rho', [x.shape[1], num_units],
                    initializer=tf.random_uniform_initializer(-3., -2.)))
        b_mu = tf.get_variable('b_mu', [num_units],
                initializer=tf.zeros_initializer())
        b_rho = tf.nn.softplus(
                tf.get_variable('b_rho', [num_units],
                    initializer=tf.random_uniform_initializer(-3., -2.)))

    xW_mean = tf.matmul(x, W_mu)
    xW_std = tf.sqrt(tf.matmul(tf.square(x), tf.square(W_rho)) + 1e-6)
    xW = xW_mean + xW_std*tf.random.normal(tf.shape(xW_mean))
    b = b_mu + b_rho * tf.random.normal(b_mu.shape)

    x = xW + b
    if activation == 'relu':
        x = tf.nn.relu(x)

    # kl divergence
    kld_W = tf.reduce_sum(kl_divergence(Normal(W_mu, W_rho), Normal(0., gamma)))
    kld_b = tf.reduce_sum(kl_divergence(Normal(b_mu, b_rho), Normal(0., gamma)))
    kld = kld_W + kld_b

    return x, kld
