import hashlib
import logging
from collections import Iterable
import tensorflow as tf
from tensorflow.python.util import nest

import numpy as np

from functools import wraps


def get_structure(data):
    """
    Returns a nested list that represents the structure of the
    data passed in.
    :param data: data from which structure is extracted
    :return: nested list representing structure of the data
    """
    flat_data = nest.flatten(data)
    return nest.pack_sequence_as(data, range(len(flat_data)))

def nest_map(data, f):
    flat_data = nest.flatten(data)
    return nest.pack_sequence_as(data, list(map(f, flat_data)))

def nest_zip(*data):
    for d in data[1:]:
        nest.assert_same_structure(data[0], d)
    flat_data = [nest.flatten(d) for d in data]
    paired = [list(e) for e in zip(*flat_data)]
    return nest.pack_sequence_as(data[0], paired)

def nested_shape(data):
    flat_data = nest.flatten(data)
    flat_shapes = [x.get_shape() for x in flat_data]
    return nest.pack_sequence_as(data, flat_shapes)

def norecurse(f):
    f.called = False

    @wraps(f)
    def wrapper(*args, **kwargs):
        if f.called:
            f.called = False
            raise RecursionError('Call recursion detected while evaluating %s' % f.__name__)
        f.called = True
        ret = f(*args, **kwargs)
        f.called = False
        return ret

    return wrapper

def combine(*args):
    """
    Combine all arguments into a new flattened list. If argument is an iterable (e.g. list or tuple)
    the growing list is extended by the item. Other wise, the item is appended to the
    growing list. This does NOT work for an arbitrarily nested list.

    Example:
    >>> combine([1, 2, 3], 4, 5, (6, 7))
    [1, 2, 3, 4, 5, 6, 7]

    :param args: items to be combined into a single list
    :return: a new list containing all elements of arguments
    """
    x = []
    for k in args:
        if isinstance(k, Iterable):
            x.extend(k)
        else:
            x.append(k)
    return x

def combine_dicts(*dicts):
    """
    When passed in one or more dictionaries, return a new dictionary
    with items combined from all input dictionaries. If the same key
    exists in more than one dictionaries, the last dictionary's value
    for the key will be used.
    """
    combined = {}
    for d in dicts:
        if d is None:
            continue
        for k, v in d.items():
            combined[k] = v
    return combined

def rekey_dict(d, prefix='', postfix=''):
    """
    Create a new dict with keys with prefix and/or postfix added
    :param d: dictionary to make a copy with modified keys
    :param prefix: prefix to be added
    :param postfix: postfix to be added
    :return: new dict with modified keys
    """
    return {prefix+k+postfix: v for k, v in d.items()}



def batchify_shape(shape, dims=None):
    """
    Convert the leading dimension to a batch dimension of None.
    :param shape: shape of an array.
    :param dims: the final dimension of the shape after batchifying. If left None, will be len(shape)
    :return: a new shape list with a leading None for the batch dimension
    """
    dims = dims or len(shape)
    return [None] + list(shape[len(shape)-dims+1:])


def _conv2d_config(input_shape, output_shape, filter_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    batch_size = input_shape[0:1]
    input_shape = np.array(input_shape[-3:])
    output_shape = np.array(tf.TensorShape(output_shape).as_list()[-3:])
    filter_shape = np.array(tf.TensorShape(filter_shape).as_list() + [input_shape[-1], output_shape[-1]])
    stride = np.ceil((input_shape[-3:-1] - filter_shape[:-2] + 1) / output_shape[:-1]).astype(np.int)
    padding = output_shape[:-1] * stride - input_shape[-3:-1] + filter_shape[:-2] - 1

    # get padded input shape

    input_shape[:-1] = input_shape[:-1] + padding.astype(np.int)

    left_padding = np.ceil(padding / 2).astype(np.int)
    right_padding = np.floor(padding / 2).astype(np.int)

    padding = [[0, 0], [left_padding[0], right_padding[0]], [left_padding[1], right_padding[1]], [0, 0]]
    stride = [1, stride[0], stride[1], 1]

    return filter_shape, stride, padding, batch_size + input_shape.tolist()


def get_convolution(inputs, output_shape, filter_size, scope=None):
    with tf.variable_scope(scope or 'convolution'):
        #logger.debug('Generating convolution layer in %s' % tf.get_variable_scope().name)
        input_shape = inputs.get_shape()[-3:]
        if not input_shape.is_fully_defined():
            raise ValueError('Shape of input tensor must be fully defined!')
        input_shape = input_shape.as_list()
        output_shape = tf.TensorShape(output_shape).as_list()
        if np.all(np.array(input_shape[:-1]) >= np.array(output_shape[:-1])):
            logging.debug('Using convolution!')
            # regular convolution
            filter_shape, strides, padding, padded_shape = _conv2d_config(input_shape, output_shape, filter_size)
            weight = tf.get_variable('weight', shape=filter_shape, initializer=tf.random_normal_initializer())

            if np.sum(padding) > 0:
                inputs = tf.pad(inputs, padding, name='padding')
            output = tf.nn.conv2d(inputs, weight, strides, padding='VALID', name='convolution')
        else:
            logging.debug('Using transpose convolution!')
            if len(output_shape) < 4:
                output_shape = [None] + output_shape
            # transpose convolution
            filter_shape, strides, padding, padded_shape = _conv2d_config(output_shape, input_shape, filter_size)
            if padded_shape[0] is None:
                batch_size = tf.shape(inputs)[0]
                padded_shape = [batch_size] + padded_shape[1:]

            weight = tf.get_variable('weight', shape=filter_shape, initializer=tf.random_normal_initializer())
            output = tf.nn.conv2d_transpose(inputs, weight, padded_shape, strides, padding='VALID',
                                            name='transpose_convolution')
            if np.sum(padding) > 0:
                output = tf.slice(output, [0, padding[1][0], padding[2][0], 0],
                                  [-1] + output_shape[1:], name='cropping')
        return weight, output


def hash_list(values):
    """
    Returns MD5 digest hash values for a list of values
    """
    hashed = hashlib.md5()
    for v in values:
        hashed.update(str(v).encode())
    return hashed.hexdigest()