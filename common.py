import tensorflow as tf
import os
from glob import glob
from tensorflow.python.training.moving_averages import assign_moving_average


def neural_net(y, neurons, name, is_training, reuse=None, decay=0.9, dtype=tf.float32):

    def batch_normalization(x):
        beta = tf.compat.v1.get_variable('beta', [x.get_shape()[-1]], dtype,
                                         tf.zeros_initializer())
        gamma = tf.compat.v1.get_variable('gamma', [x.get_shape()[-1]], dtype,
                                          tf.ones_initializer())
        mv_mean = tf.compat.v1.get_variable('mv_mean', [x.get_shape()[-1]], dtype=dtype,
                                            initializer=tf.zeros_initializer(), trainable=False)
        mv_var = tf.compat.v1.get_variable('mv_var', [x.get_shape()[-1]], dtype=dtype,
                                           initializer=tf.ones_initializer(), trainable=False)
        mean, variance = tf.nn.moments(x, [0], name='moments')
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, assign_moving_average(mv_mean, mean, decay, zero_debias=True))
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, assign_moving_average(mv_var, variance, decay, zero_debias=False))

        if is_training:
            return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
        else:
            return tf.nn.batch_normalization(x, mv_mean, mv_var, beta, gamma, 1e-6)

    def layer(x, out_size, activation):
        w = tf.compat.v1.get_variable('weights', [x.get_shape().as_list()[-1], out_size], dtype,
                                      tf.initializers.glorot_uniform())
        return activation(batch_normalization(tf.matmul(x, w)))

    with tf.compat.v1.variable_scope(name, reuse=reuse):
        y = batch_normalization(y)
        for i in range(len(neurons) - 1):
            with tf.compat.v1.variable_scope('layer_%i_' % (i + 1)):
                y = layer(y, neurons[i], tf.nn.tanh)
        with tf.compat.v1.variable_scope('layer_%i_' % len(neurons)):
            return layer(y, neurons[-1], tf.identity)


def splitting_model(y, z, t, n, phi, h, net, neurons, batch_size, dtype=tf.float32):

    v_n = None

    _y = y[:, :, 1]
    _z = z[:, net]
    if net == 0:
        v_i = phi(_y)
    else:
        v_i = neural_net(_y, neurons, 'v_%i_' % net, False, dtype=dtype)
    grad_v = tf.gradients(v_i, _y)

    _z = tf.reshape(tf.constant(_z, dtype=tf.float32), (1, 1))

    if net == n - 1:
        v_n = tf.compat.v1.get_variable('v_%i_' % (net + 1), [], dtype, tf.random_uniform_initializer())
        v_j = tf.ones([batch_size, 1], dtype) * v_n
    else:
        v_j = neural_net(y[:, :, 0], neurons, 'v_%i_' % (net + 1), True, dtype=dtype)

    loss = (v_j - tf.stop_gradient(h(_y, v_i, grad_v[0], _z, t / n))) ** 2

    return tf.reduce_mean(loss), v_n


def simulate(t, n, d, sde, phi, h, z, neurons, train_steps, batch_size,
             lr_boundaries, lr_values, path, epsilon=1e-8):

    for i in range(n):

        tf.compat.v1.reset_default_graph()

        y = sde(d, n - i - 1)
        loss, v_n = splitting_model(y, z, t, n, phi, h, i, neurons, batch_size)

        global_step = tf.compat.v1.get_variable('global_step_%i_' % (i + 1), [], tf.int32,
                                      tf.zeros_initializer(), trainable=False)

        learning_rate = tf.compat.v1.train.piecewise_constant(global_step, lr_boundaries, lr_values)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, 'v_%i_' % (i + 1))
        with tf.control_dependencies(update_ops):
            train_op = tf.compat.v1.train.AdamOptimizer(learning_rate, epsilon=epsilon).minimize(loss, global_step=global_step)

        with tf.compat.v1.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            var_list_n = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, 'v_%i_' % (i + 1))
            saver_n = tf.compat.v1.train.Saver(var_list=var_list_n)

            if i > 0:
                saver_p = tf.compat.v1.train.Saver(var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, 'v_%i_' % i))
                saver_p.restore(sess, os.path.join(path, 'model_%i_' % i))

            for _ in range(train_steps):
                sess.run(train_op)

            saver_n.save(sess, os.path.join(path, 'model_%i_' % (i + 1)))
            try:
                for filename in glob(os.path.join(path, 'model_%i_*' % (i - 1))):
                    os.remove(filename)
            except OSError:
                pass

            if i == n - 1:
                return sess.run(v_n)
