## Collection of utils for building networks
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from atflow import constraints


def conv2d_output_shape(input_shape, filter_shape, stride, padding):
    """
    Computes the shape of the output tensor from conv2d operation with the given configuration

    :param input_shape: shape of the input tensor, must be a list, numpy array or TensorShape
    :param filter_shape: shape of the convolution filter.
    :param stride: stride for the convolution
    :param padding: padding mode, either 'VALID' or 'SAME'
    :return: shape of the output tensor as a plain list of integers
    """
    filter_shape = tf.TensorShape(filter_shape).as_list()
    filter_out = filter_shape[-1]
    filter_patch_shape = np.array(filter_shape[0:2])
    input_shape_list = tf.TensorShape(input_shape).as_list()
    batch = input_shape_list[:-3]
    input_shape = np.array(input_shape_list[-3:])
    stride = np.array(stride)
    if padding == 'VALID':
        shift = -filter_patch_shape + 1
    elif padding == 'SAME':
        shift = 0
    else:
        raise ValueError('padding must be either "VALID" or "SAME", but "%s" was given' % padding)
    output_shape = np.ceil((input_shape[:2] + shift) / stride[1:3])
    return batch + output_shape.astype(np.int).tolist() + [filter_out]


def conv2d_config(input_shape, output_shape, filter_shape):
    """
    Based on the desired input, output and filter shape, figure out the correct 2D convolution configuration to use
    including the type (normal or full convolution), stride size, padding type/size
    :param input_shape:
    :param output_shape:
    :param filter_shape:
    :return:
    """
    input_shape = tf.TensorShape(input_shape).as_list()
    if len(input_shape) == 4:
        batch_size = input_shape[0]
    else:
        batch_size = None

    input_shape = np.array(input_shape[-3:])
    output_shape = np.array(tf.TensorShape(output_shape).as_list()[-3:])

    # Determine what kind of convolution to use
    if np.all(input_shape[-3:-1] >= output_shape[-3:-1]):
        conv_type = "NORMAL"
    elif np.all(input_shape[-3:-1] <= output_shape[-3:-1]):
        conv_type = 'FULL'
        # swap input and output shape
        input_shape, output_shape = output_shape, input_shape
    else:
        raise ValueError('Input shape dimensions must be both bigger than or both smaller than output shape dimensions')

    filter_shape = np.array(tf.TensorShape(filter_shape).as_list()[:2] + [input_shape[-1], output_shape[-1]])
    stride = np.ceil((input_shape[:2] - filter_shape[:2] + 1) / output_shape[:2]).astype(np.int)
    padding = output_shape[:2] * stride - input_shape[:2] + filter_shape[:2] - 1

    # Determine what type of padding can be used
    if np.all(np.ceil(input_shape[:2] / stride) == output_shape[:2]):
        padding_type = 'SAME'
    else:
        padding_type = 'VALID'

    # get padded input shape
    input_shape[:2] = input_shape[:2] + padding.astype(np.int)
    padded_shape = [batch_size] + input_shape.tolist()

    left_padding = np.ceil(padding / 2).astype(np.int)
    right_padding = np.floor(padding / 2).astype(np.int)

    padding = [[0, 0], [left_padding[0], right_padding[0]], [left_padding[1], right_padding[1]], [0, 0]]
    stride = [1, stride[0], stride[1], 1]

    return filter_shape.tolist(), stride, padding, padded_shape, conv_type, padding_type


def get_convolution_op(input_shape, output_shape, kernel_shape):
    """
    Given the desired shapes of the input, output and filter tensors, returns the shape of the appropriate
    convolution filter and a correctly configured op function. The returned op function should be called with the
    input tensor and weight tensor, and returns a result of 2D convolution that matches the desired output_shape
    :param input_shape: desired input shape into the convolution operation
    :param output_shape: desired output shape from the convolution operation
    :param kernel_shape: desired convolution kernel shape. Only the first two diemensions (height and width) will be used.
    :return: (filter_shape, conv_op) The shape of the appropriate convolution filter/weight to be used (filter_shape) and
    a function that can be invoked with inputs tensor and correctly sized filter tensor to define the convolution operation.
    """
    filter_shape, strides, padding, padded_shape, conv_type, padding_type = conv2d_config(input_shape, output_shape, kernel_shape)
    if conv_type == 'NORMAL':
        def conv_op(inputs, weight, name='generic_convolution'):
            with tf.name_scope(name):
                if padding_type=='VALID' and np.sum(padding) > 0:
                    inputs = tf.pad(inputs, padding, name='padding')
                return tf.nn.conv2d(inputs, weight, strides, padding_type, name='convolution')

    else:
        def conv_op(inputs, weight, name='generic_convolution'):
            if padding_type=='SAME':
                padded_output = [padded_shape[0]] + output_shape[-3:]
            else:
                padded_output = padded_shape
            with tf.name_scope(name):
                if padded_output[0] is None:
                    batch_size = tf.shape(inputs)[0]
                    padded_output = [batch_size] + padded_output[1:]

                output = tf.nn.conv2d_transpose(inputs, weight, padded_output, strides, padding_type, name='transpose_convolution')
                if padding_type=='VALID' and np.sum(padding) > 0:
                    output = tf.slice(output, [0, padding[1][0], padding[2][0], 0],
                                      [-1] + output_shape[-3:], name='cropping')
                return output

    return filter_shape, conv_op




def normalize_weights(w, dims=(0,), bias=1e-5):
    """
    L2 normalize weights of the given tensor along specified dimension(s).

    Args:
        w: Tensor to be normalized
        dims: dimension(s) along which to normalize the Tensor. Defaults to (0,)
        bias: Bias value added to the computed norm to prevent dividing by 0. Defaults to 1e-5

    Returns: Tensor of same type and shape as `w` whose norm is set to approximately 1 along the specificed dimensions
    """
    with tf.name_scope('normalization'):
        return w / (tf.sqrt(tf.reduce_sum(tf.square(w), dims, keep_dims=True) + bias))


def weight_variable(shape, name='weight', mean=0.0, stddev=1e-3, initializer=None, constrain=None, dtype=tf.float32):
    """
    Creates and returns a variable initialized with random_normal_initializer, suitable for use as a weight.

    In the current variable scope, creates (if necessary) and returns a named variable with `tf.random_normal_initializer`.

    Args:
        shape: Required. Shape of the variable
        name: Optional. Name of the variable. Defaults to 'weight'
        mean: Optional. Mean of the `random_normal_initializer`. Defaults to 0.0
        stddev: Optional. Standard deviation of the `random_normal_initializer`. Defaults to 1e-3
        dtype: Optional. Data type of the variable. Default to `tf.float32`.

    Returns: Weight variable with specified name and shape with random normal initialization.

    """
    if initializer is None:
        initializer = tf.random_normal_initializer(mean=mean, stddev=stddev)
    weights = tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype)
    if constrain is not None:
        constrain(weights)
    return weights


def bias_variable(shape, name='bias', value=0.0, initializer=None, constrain=None, dtype=tf.float32):
    """
    Creates and returns a variable initialized with random_normal_initializer, suitable for use as a bias.

    In the current variable scope, creates (if necessary) and returns a named variable with `tf.random_normal_initializer`.

    Args:
        shape: Required. Shape of the variable
        name: Optional. Name of the variable. Defaults to 'bias'
        value: Optional. Constant value to which the variable is initialized. Defaults to 0.0
        dtype: Optional. Data type of the variable. Default to `tf.float32`.

    Returns: Bias variable with specified name and shape initialized to a constant.

    """
    if initializer is None:
        initializer = tf.constant_initializer(value=value)
    biases = tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype)
    if constrain is not None:
        constrain(biases)
    return biases


def factorized_readout(inputs, n_outputs=100, constrain=True):
    width, height, n_features = inputs.get_shape()[1:].as_list()
    n_pixels = width * height

    with tf.variable_scope('readout'):
        # spatial readout
        w_spatial = weight_variable([n_pixels, 1, n_outputs], name='weight_spatial')
        if constrain:
            constraints.positive_constrain(w_spatial)
        w_spatial_norm = normalize_weights(w_spatial, dims=(0,))

        # feature readout
        w_feature = weight_variable([1, n_features, n_outputs], name='weight_feature')
        if constrain:
            constraints.positive_constrain(w_feature)
        w_feature_norm = normalize_weights(w_feature, dims=(1,))

        # scaling
        w_scale = bias_variable([n_outputs], name='weight_scale', value=1.0)
        if constrain:
            constraints.positive_constrain(w_scale)

        # total readout weight
        w_out = tf.reshape(w_spatial_norm * w_feature_norm * w_scale, [n_pixels * n_features, n_outputs],
                           'weight_readout')

        output = tf.matmul(tf.reshape(inputs, [-1, n_pixels * n_features]), w_out)

        return output, w_spatial_norm, w_feature_norm, w_scale, w_out


def batch_norm(inputs, *args, tag=None, add_summary=True, step=0, **kwargs):
    if step > 0 and 'updates_collections' not in kwargs:
        kwargs['updates_collections'] = 'dump'
    output = layers.batch_norm(inputs, *args, **kwargs)
    if add_summary:
        if tag is None:
            tag = inputs.op.name.split('/')[-1]
        tag = 'batch_norm/' + tag
        tf.histogram_summary(tag, inputs)
        tf.histogram_summary(tag + '_bn', output)
    return output