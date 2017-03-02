import tensorflow as tf

def mse(output, target, scope=None):
    with tf.name_scope(scope or 'mse'):
        return tf.reduce_mean(tf.square(output - target))


def poisson_loss(output, target, bias=1e-15, axis=0, scope=None):
    with tf.name_scope(scope or 'poisson_loss'):
        return tf.reduce_mean(output - target * tf.log(output + bias), axis)