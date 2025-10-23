import tensorflow as tf
import numpy as np
import os
import time
import shutil
from common import simulate


def phi(x):
    return tf.reduce_sum(x ** 2, 1, keepdims=True)


def h(x, u, w, z, dt):
    return u + z


def sde(_d, n):
    x = [tf.compat.v1.random_normal(
         [batch_size, _d, 1], stddev=np.sqrt(2. * n * T / N)),
         tf.compat.v1.random_normal(
         [batch_size, _d, 1], stddev=np.sqrt(2. * T / N))]
    return tf.cumsum(tf.concat(x, axis=2), axis=2)


tf.compat.v1.disable_eager_execution()
batch_size = 1024
train_steps = 10000
lr_boundaries = [4000, 6000, 8000]
lr_values = [0.1, 0.01, 0.001, 0.0001]

T = 1.
N = 5

path = '/tmp/heat'
_file = open('heat.csv', 'w')
_file.write('d, T, N, run, value, time, ref, pc\n')

for d in [1, 5, 10, 20, 50, 100]:

    neurons = [d + 50, d + 50, 1]

    for run in range(5):

        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()
        z = np.random.normal(0., np.sqrt(T / N), (1, N))

        xi = np.zeros((d, ))

        v_n = simulate(T, N, d, sde, phi, h, z, neurons, train_steps,
                       batch_size, lr_boundaries, lr_values, path)
        t_1 = time.time()

        b1 = np.cumsum(z, 1)
        u_reference = b1[:, -1] + 2. * T * d + np.sum(xi ** 2, 0)
        _file.write('%i, %f, %i, %i, %f, %f, %f, %f\n'
                    % (d, T, N, run, v_n, t_1 - t_0, u_reference,
                       abs(v_n - u_reference) / u_reference))
        _file.flush()

_file.close()
