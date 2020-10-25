import tensorflow as tf
import numpy as np
import os
import time
import shutil
from common import simulate


def phi(x):
    return np.exp(-1. / 50. * T) * tf.maximum(tf.reduce_max(x, 1, keepdims=True) - 100., 0.)


def h(x, u, w, z, dt):
    return u * (1. + z + 0.5 * z ** 2 - 0.5 * dt)


def sde(_d, n):
    x = [tf.compat.v1.random_normal([batch_size, _d, 1], stddev=np.sqrt(n * T / N)),
         tf.compat.v1.random_normal([batch_size, _d, 1], stddev=np.sqrt(T / N))]
    t = tf.reshape(np.array([n * T / N, (n + 1) * T / N], dtype=np.float32), [1, 1, 2])
    return tf.exp((mu - sigma ** 2 / 2.) * t +
                  sigma * tf.cumsum(tf.concat(x, axis=2), axis=2)) * tf.ones([1, _d, 1]) * 100.


def mc(_d):
    y = tf.compat.v1.random_normal([batch_size, _d, 1], stddev=np.sqrt(T))
    x = tf.exp((mu - sigma ** 2 / 2.) * T + sigma * y) * tf.ones([1, _d, 1]) * 100.
    x = phi(x)
    return tf.reduce_mean(x)


tf.compat.v1.disable_eager_execution()
batch_size = 1024
train_steps = 10000
lr_boundaries = [4000, 6000, 8000]
lr_values = [0.1, 0.01, 0.001, 0.0001]

T = 0.5
N = 20

path = '/tmp/bs'
_file = open('BlackScholes.csv', 'w')
_file.write('d, T, N, run, value, time, ref, pc\n')

for d in [1, 5, 10, 20]:

    neurons = [d + 50, d + 50, 1]

    for run in range(5):

        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()

        mu = np.reshape((np.sin(np.linspace(d * 1., 1. * d * d, d)) + 1.) / (1. * d), (1, d, 1))
        sigma = np.reshape(np.linspace(1., 1. * d, d) / (4. * d), (1, d, 1))

        z = np.random.normal(0., np.sqrt(T / N), (1, N))

        v_n = simulate(T, N, d, sde, phi, h, z, neurons, train_steps,
                       batch_size, lr_boundaries, lr_values, path)
        t_1 = time.time()

        b1 = np.cumsum(z, 1)
        u_reference = np.exp(b1[:, -1] - T / 2.)

        tf.compat.v1.reset_default_graph()
        ref_sol = mc(d)
        mc_val = 0.
        with tf.compat.v1.Session() as sess:
            for _ in range(1000):
                mc_val += sess.run(ref_sol)

        u_reference = u_reference * mc_val / 1000.

        _file.write('%i, %f, %i, %i, %f, %f, %f, %f\n' % (d, T, N, run, v_n,
                                                          t_1 - t_0, u_reference,
                                                          abs(v_n - u_reference) / u_reference))
        _file.flush()

_file.close()
