import tensorflow as tf

def l1_regularizer(scale, scope='l1_regularizer', name=None, use_mean=True):
    def l1(weights):
        with tf.name_scope(scope) as name:
            if use_mean:
                v = tf.reduce_mean(tf.abs(weights))
            else:
                v = tf.reduce_sum(tf.abs(weights))
            return tf.mul(scale, v, name=name)
    return l1


def l2_regularizer(scale, scope='l2_regularizer', use_mean=True):
    def l2(weights):
        with tf.name_scope(scope) as name:
            if use_mean:
                v = tf.reduce_mean(tf.square(weights))/2
            else:
                v = tf.reduce_sum(tf.square(weights))/2
            return tf.mul(scale, v, name=name)
    return l2


def sparsity_L1_regularizer(weight, alpha=0.01, name=None, add_summary=False):
    if name is None:
        name = 'L1_sparsity/' + weight.op.name
    l1_reg = l1_regularizer(alpha)(weight)
    tf.add_to_collection('regularization', l1_reg)
    if add_summary:
        tf.scalar_summary('regularizer/' + name, l1_reg)
    return l1_reg


def laplacian_smoothness_regularizer(weight, alpha=0.01, name=None, add_summary=False):
    # Laplacian filter for smoothness prior on convolution weights
    if name is None:
        name = 'Laplacian_smoothness/' + weight.op.name
    lap = tf.constant([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]])
    lap = tf.expand_dims(tf.expand_dims(lap, 2), 3)
    shape = tf.shape(weight)
    weight_collapsed = tf.reshape(weight, [shape[0], shape[1], 1, -1])
    w_lap = tf.nn.conv2d(tf.transpose(weight_collapsed, perm=[3, 0, 1, 2]),
                         lap, strides=[1, 1, 1, 1], padding='SAME')
    # smoothness regualization on conv weights
    smoothness = l2_regularizer(alpha)(w_lap)
    tf.add_to_collection('regularization', smoothness)
    if add_summary:
        tf.scalar_summary('regularizer/' + name, smoothness)
    return smoothness