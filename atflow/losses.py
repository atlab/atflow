import tensorflow as tf

def mse(output, target, scope=None):
    """
    Computes mean-squared error loss given output and target

    Args:
        output: output of the network
        target: target of the training
        scope: scope to define the operation under. Defaults to `mse`

    Returns:
        Ops computing MSE on output and target

    """
    with tf.name_scope(scope or 'mse'):
        return tf.reduce_mean(tf.square(output - target))


def poisson_loss(output, target, bias=1e-15, axis=0, scope=None):
    """
    Compute Poisson loss on the given output and target

    Poisson loss is computed as: x - t * log(x) where x is the output and t is the target

    Args:
        output: output of the network
        target: target of the training
        bias: safety margin to add to output when computing log
        axis: axis to take mean after poisson loss computation
        scope:

    Returns:
        Ops computing Poisson loss on output and target
    """
    with tf.name_scope(scope or 'poisson_loss'):
        return tf.reduce_mean(output - target * tf.log(output + bias), axis)