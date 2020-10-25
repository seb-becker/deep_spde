import tensorflow as tf
import numpy as np
import os
import time
import shutil
from common import simulate


def phi(x, _d):
    return ((alpha / 2. / np.pi) ** (_d / 2.)) * tf.exp(-alpha / 2. * tf.reduce_sum(x ** 2, axis=1, keepdims=True))


def h(x, u, w, z, dt, _d):
    sum_x2 = tf.reduce_sum(x ** 2, axis=1, keepdims=True)
    tmp0 = d / (1. + sum_x2) + 2. * sum_x2 / (1. + sum_x2) ** 2
    hz_sum = tf.reduce_sum(beta * x * z, axis=1, keepdims=True)
    tmp1 = u * hz_sum
    tmp2 = u / 2. * tf.reduce_sum(beta * x * z * hz_sum, axis=1, keepdims=True)
    tmp3 = u * T / N / 2. * tf.reduce_sum(beta * x * beta * x, axis=1, keepdims=True)
    return u - u * tmp0 * gamma + tmp1 + tmp2 - tmp3


def sde(_d, n):
    y = [tf.constant(np.ones((batch_size, _d), dtype=np.float32) * 0.)]
    for n_ in range(n+1):
        mu = gamma * y[-1] / (1. + tf.reduce_sum(y[-1] ** 2, axis=1, keepdims=True))
        sigma = tf.ones((batch_size, _d)) * tf.reduce_sum(tf.compat.v1.random_normal((batch_size, _d),
                                                            stddev=np.sqrt(T / N)), axis=1, keepdims=True)
        y.append(y[-1] + mu * T / N + sigma / np.sqrt(_d))
    return tf.stack(y[n:n + 2], axis=2)


def example():
    w = np.random.normal(0., np.sqrt(T / N), (d, N))
    v = np.random.normal(0., np.sqrt(T / N), (d, N))
    y = [np.zeros((d, ))]
    z = [np.zeros((d, ))]
    for i in range(N):
        z.append(z[-1] + beta * T / N * y[-1] + v[:, i])
        y.append(y[-1] + gamma * T/N * y[-1] / (1. + np.sum(y[-1] ** 2)) + np.sum(w[:, i]) / np.sqrt(d))

    return v, y, z


def ref_solution(v, y):
    v = tf.cumsum(tf.expand_dims(tf.cast(v, tf.float32), axis=0), axis=2)
    y = tf.cast(tf.expand_dims(tf.stack(y, axis=1), axis=0), tf.float32)
    x_0 = tf.zeros((batch_size, d)) + 0.
    w = tf.compat.v1.random_normal((batch_size, d, N), stddev=np.sqrt(T / N))
    yy = [x_0]
    BB = 0.
    fact = 0.5
    for i in range(N):
        vv = v[:, :, N - i - 1]
        vv_sum = tf.reduce_sum(v[:, :, N - i - 1], axis=1, keepdims=True)
        yy_norm = tf.reduce_sum(yy[-1] ** 2, axis=1, keepdims=True)
        BB += fact * T/N * tf.reduce_sum(0.5 * vv_sum * tf.ones((1, d)) * vv * beta ** 2, axis=1, keepdims=True)
        BB += fact * T/N * tf.reduce_sum(yy[-1] * y[:, :, N - i - 1] * beta ** 2 - 0.5 * (beta * yy[-1]) ** 2
                    - beta * gamma * vv * yy[-1] / (1. + yy_norm)
                    - gamma * (1. + yy_norm - 2. * yy[-1] ** 2) / ((1. + yy_norm) ** 2), axis=1, keepdims=True)
        yy.append(yy[-1] + T/N * (beta * vv_sum * tf.ones((1, d)) - gamma * yy[-1] / (1. + yy_norm))
                  + tf.reduce_sum(w[:, :, i], axis=1, keepdims=True) / np.sqrt(d))
        fact = 1.
    vv = v[:, :, 0] * 0.
    vv_sum = tf.reduce_sum(v[:, :, 0] * 0., axis=1, keepdims=True)
    yy_norm = tf.reduce_sum(yy[-1] ** 2, axis=1, keepdims=True)
    BB += 0.5 * T / N * tf.reduce_sum(0.5 * vv_sum * tf.ones((1, d)) * vv * beta ** 2, axis=1, keepdims=True)
    BB += 0.5 * T / N * tf.reduce_sum(yy[-1] * y[:, :, 0] * beta ** 2 - 0.5 * (beta * yy[-1]) ** 2
                    - beta * gamma * vv * yy[-1] / (1. + yy_norm)
                    - gamma * (1. + yy_norm - 2. * yy[-1] ** 2) / ((1. + yy_norm) ** 2), axis=1, keepdims=True)
    return tf.reduce_mean(((alpha / 2. / np.pi) ** (d / 2)) * tf.exp(BB - alpha / 2. * yy_norm), axis=0, keepdims=True)


tf.compat.v1.disable_eager_execution()
batch_size = 2048
train_steps = 12000
lr_boundaries = [5000, 10000]
lr_values = [0.01, 0.001, 0.0001]

alpha, beta, gamma = 2. * np.pi, 0.25, 0.1
T = 0.5
N = 25

path = '/tmp/zakai'
_file = open('Zakai.csv', 'w')
_file.write('d, T, N, run, value, time, ref, pc\n')

for d in [1, 5, 10, 20, 50]:

    neurons = [d + 50, d + 50, 1]

    for run in range(5):

        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()

        tf.compat.v1.reset_default_graph()
        v, y, z = example()

        z = np.stack(z, axis=1)
        z = np.diff(z, axis=1)

        v_mean = ref_solution(v, y)

        sum = 0.
        with tf.compat.v1.Session() as sess:
            for _ in range(1000):
                sum += sess.run(v_mean)

            sum /= 1000.

        b1 = np.cumsum(v, 1)
        u_reference = sum * np.exp(np.sum(beta * 0. * b1[:, -1]))

        tf.compat.v1.reset_default_graph()

        v_n = simulate(T, N, d, sde, lambda x: phi(x, d), lambda x, u, w, z, dt: h(x, u, w, z, dt, d),
                       z, neurons, train_steps, batch_size, lr_boundaries, lr_values, path)

        t_1 = time.time()

        _file.write('%i, %f, %i, %i, %f, %f, %f, %f\n' % (d, T, N, run, v_n,
                                                          t_1 - t_0, u_reference,
                                                          abs(v_n - u_reference) / u_reference))
        _file.flush()

_file.close()
