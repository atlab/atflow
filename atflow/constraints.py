# Utility functions for defining constrained variables
import tensorflow as tf
import numpy as np
from math import floor


def constrain_all_variables():
    """Apply constraining ops on all variables with constraints.

    Returns:
        An Op that applies variables constraints.
    """
    return tf.group(*tf.get_collection("constraints"))


def positive_constrain(var, scope=None):
    """Constrain the variable tensor `var` to be zero or positive in all entries"""
    if tf.get_variable_scope().reuse: return
    with tf.name_scope(scope or 'positive_constrain'):
        constrained_value = tf.select(tf.greater_equal(var, 0.0), var, tf.zeros_like(var))
        update = tf.assign(var, constrained_value)
        tf.add_to_collection('constraints', update)
        return update


def negative_constrain(var, scope=None):
    """Constrain the variable tensor `var` to be zero or negative in all entries"""
    if tf.get_variable_scope().reuse: return
    with tf.name_scope(scope or 'negative_constrain'):
        constrained_value = tf.select(tf.less_equal(var, 0.0), var, tf.zeros_like(var))
        update = tf.assign(var, constrained_value)
        tf.add_to_collection('constraints', update)
        return update


def offdiagonal_constrain(var, scope=None):
    """Constrain all offdiagonal values to be positive for variable `var`"""
    if tf.get_variable_scope().reuse: return
    with tf.name_scope(scope or 'offdiagonal_constrain'):
        diag = tf.diag_part(var)
        constrained_value = tf.select(tf.greater_equal(var, 0.0), var, tf.zeros_like(var))
        constrained_value = tf.batch_matrix_set_diag(constrained_value, diag)
        update = tf.assign(var, constrained_value)
        tf.add_to_collection('constraints', update)
        return update


def nonself_filter_constrain(var, scope=None):
    """Constrain all non-self position in the convolution filter to be positive"""
    if tf.get_variable_scope().reuse: return
    filter = np.zeros(var.get_shape().as_list())
    ch = floor(filter.shape[0]/2)
    cw = floor(filter.shape[1]/2)
    diag = np.eye(filter.shape[2], dtype=np.bool)
    filter[ch, cw][diag] = 1
    with tf.name_scope(scope or 'nonself_filter_constrain'):
        filter = tf.constant(filter, dtype=tf.bool)
        selection = tf.logical_or(tf.greater_equal(var, 0.0), filter)
        constrained_value = tf.select(selection, var, tf.zeros_like(var))
        update = tf.assign(var, constrained_value)
        tf.add_to_collection('constraints', update)
        return update, selection


def offcenter_constrain(var, scope=None):
    """Constrain all off center convolution kernel values to be positive for variable `var`"""
    if tf.get_variable_scope().reuse: return
    filter = np.zeros(var.get_shape().as_list())
    ch = floor(filter.shape[0]/2)
    cw = floor(filter.shape[1]/2)
    filter[ch, cw, ... ] = 1
    with tf.name_scope(scope or 'offcenter_constrain'):
        filter = tf.constant(filter, dtype=tf.bool)
        selection = tf.logical_or(tf.greater_equal(var, 0.0), filter)
        constrained_value = tf.select(selection, var, tf.zeros_like(var))
        update = tf.assign(var, constrained_value)
        tf.add_to_collection('constraints', update)
        return update, selection

