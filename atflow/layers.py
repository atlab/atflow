import tensorflow as tf

import functools

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               initializers={},
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
    with variable_scope.variable_scope(scope, 'BatchNorm', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        axis = list(range(inputs_rank - 1))
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta_initializer = initializers.get('beta', init_ops.zeros_initializer)
            beta = variables.model_variable('beta',
                                            shape=params_shape,
                                            dtype=dtype,
                                            initializer=beta_initializer,
                                            collections=beta_collections,
                                            trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections,
                                                               'gamma')
            gamma_initializer = initializers.get('gamma', init_ops.ones_initializer)
            gamma = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=gamma_initializer,
                                             collections=gamma_collections,
                                             trainable=trainable)
        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections.
        moving_mean_collections = utils.get_variable_collections(
            variables_collections, 'moving_mean')
        moving_mean_initializer = initializers.get('moving_mean', init_ops.zeros_initializer)
        moving_mean = variables.model_variable(
            'moving_mean',
            shape=params_shape,
            dtype=dtype,
            initializer=moving_mean_initializer,
            trainable=False,
            collections=moving_mean_collections)
        moving_variance_collections = utils.get_variable_collections(
            variables_collections, 'moving_variance')
        moving_variance_initializer = initializers.get('moving_variance', init_ops.ones_initializer)
        moving_variance = variables.model_variable(
            'moving_variance',
            shape=params_shape,
            dtype=dtype,
            initializer=moving_variance_initializer,
            trainable=False,
            collections=moving_variance_collections)

        # If `is_training` doesn't have a constant value, because it is a `Tensor`,
        # a `Variable` or `Placeholder` then is_training_value will be None and
        # `needs_moments` will be true.
        is_training_value = utils.constant_value(is_training)
        need_moments = is_training_value is None or is_training_value
        if need_moments:
            # Calculate the moments based on the individual batch.
            # Use a copy of moving_mean as a shift to compute more reliable moments.
            shift = math_ops.add(moving_mean, 0)
            mean, variance = nn.moments(inputs, axis, shift=shift)
            tf.add_to_collection('moments_mean', mean)
            tf.add_to_collection('moments_variance', variance)
            tf.add_to_collection('mavg_mean', moving_mean)
            tf.add_to_collection('mavg_variance', moving_variance)

            moving_vars_fn = lambda: (moving_mean, moving_variance)
            if updates_collections is None:
                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay)
                    with ops.control_dependencies([update_moving_mean,
                                                   update_moving_variance]):
                        return array_ops.identity(mean), array_ops.identity(variance)

                mean, variance = utils.smart_cond(is_training,
                                                  _force_updates,
                                                  moving_vars_fn)
            else:
                def _delay_updates():
                    """Internal function that delay updates moving_vars if is_training."""
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay)
                    print('Defining delay update')

                    return update_moving_mean, update_moving_variance

                update_mean, update_variance = utils.smart_cond(is_training,
                                                                _delay_updates,
                                                                moving_vars_fn)
                ops.add_to_collections(updates_collections, update_mean)
                ops.add_to_collections(updates_collections, update_variance)
                # Use computed moments during training and moving_vars otherwise.
                vars_fn = lambda: (mean, variance)
                mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
        else:
            mean, variance = moving_mean, moving_variance
        # Compute batch_normalization.
        tf.add_to_collection('final_batch_norm_mean', mean)
        tf.add_to_collection('final_batch_norm_var', variance)
        outputs = nn.batch_normalization(
            inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


def batch_norm_old(
        inputs,
        decay=0.999,
        center=True,
        scale=False,
        epsilon=0.001,
        activation_fn=None,
        param_initializers=None,
        updates_collections=ops.GraphKeys.UPDATE_OPS,
        is_training=True,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        batch_weights=None,
        data_format=DATA_FORMAT_NHWC,
        scope=None):
    class Container:
        pass

    values = Container()

    with variable_scope.variable_op_scope([inputs],
                                          scope, 'BatchNorm', reuse=reuse) as sc:

        inputs = ops.convert_to_tensor(inputs)
        values.inputs = inputs
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if batch_weights is not None:
            batch_weights = ops.convert_to_tensor(batch_weights)
            values.batch_weights = batch_weights
            inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
            # Reshape batch weight values so they broadcast across inputs.
            nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
            batch_weights = array_ops.reshape(batch_weights, nshape)
        axis = list(range(inputs_rank - 1))
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if not param_initializers:
            param_initializers = {}
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta_initializer = param_initializers.get('beta',
                                                      init_ops.zeros_initializer)
            beta = variables.model_variable('beta',
                                            shape=params_shape,
                                            dtype=dtype,
                                            initializer=beta_initializer,
                                            collections=beta_collections,
                                            trainable=trainable)
            values.beta = beta
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections,
                                                               'gamma')
            gamma_initializer = param_initializers.get('gamma',
                                                       init_ops.ones_initializer)
            gamma = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=gamma_initializer,
                                             collections=gamma_collections,
                                             trainable=trainable)
            values.gamma = gamma

        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections. We disable variable partitioning while creating
        # them, because assign_moving_average is not yet supported for partitioned
        # variables.
        partitioner = variable_scope.get_variable_scope().partitioner
        try:
            variable_scope.get_variable_scope().set_partitioner(None)
            moving_mean_collections = utils.get_variable_collections(
                variables_collections, 'moving_mean')
            moving_mean_initializer = param_initializers.get(
                'moving_mean', init_ops.zeros_initializer)
            moving_mean = variables.model_variable(
                'moving_mean',
                shape=params_shape,
                dtype=dtype,
                initializer=moving_mean_initializer,
                trainable=False,
                collections=moving_mean_collections)
            values.moving_mean = moving_mean
            moving_variance_collections = utils.get_variable_collections(
                variables_collections, 'moving_variance')
            moving_variance_initializer = param_initializers.get(
                'moving_variance', init_ops.ones_initializer)
            moving_variance = variables.model_variable(
                'moving_variance',
                shape=params_shape,
                dtype=dtype,
                initializer=moving_variance_initializer,
                trainable=False,
                collections=moving_variance_collections)
            values.moving_variance = moving_variance
        finally:
            variable_scope.get_variable_scope().set_partitioner(partitioner)

        # If `is_training` doesn't have a constant value, because it is a `Tensor`,
        # a `Variable` or `Placeholder` then is_training_value will be None and
        # `needs_moments` will be true.
        is_training_value = utils.constant_value(is_training)
        need_moments = is_training_value is None or is_training_value
        if need_moments:
            # Calculate the moments based on the individual batch.
            if batch_weights is None:
                # Use a copy of moving_mean as a shift to compute more reliable moments.
                shift = math_ops.add(moving_mean, 0)
                values.shift = shift
                mean, variance = nn.moments(inputs, axis, shift=shift)
            else:
                mean, variance = nn.weighted_moments(inputs, axis, batch_weights)

            values.moment_mean, values.moment_variance = mean, variance

            moving_vars_fn = lambda: (moving_mean, moving_variance)
            if updates_collections is None:
                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay)
                    with ops.control_dependencies([update_moving_mean,
                                                   update_moving_variance]):
                        return array_ops.identity(mean), array_ops.identity(variance)

                mean, variance = utils.smart_cond(is_training,
                                                  _force_updates,
                                                  moving_vars_fn)
            else:
                def _delay_updates():
                    """Internal function that delay updates moving_vars if is_training."""
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay)
                    return update_moving_mean, update_moving_variance

                update_mean, update_variance = utils.smart_cond(is_training,
                                                                _delay_updates,
                                                                moving_vars_fn)
                ops.add_to_collections(updates_collections, update_mean)
                ops.add_to_collections(updates_collections, update_variance)
                # Use computed moments during training and moving_vars otherwise.
                vars_fn = lambda: (mean, variance)
                mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
        else:
            mean, variance = moving_mean, moving_variance

        values.mean, values.variance = mean, variance

        # Compute batch_normalization.
        outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                         epsilon)
        values.output = outputs
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)

        tf.add_to_collection('batch_values', values)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


@add_arg_scope
def batch_norm2(inputs,
                decay=0.999,
                center=True,
                scale=False,
                epsilon=0.001,
                activation_fn=None,
                updates_collections=ops.GraphKeys.UPDATE_OPS,
                is_training=True,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
    with variable_scope.variable_op_scope([inputs],
                                          scope, 'BatchNorm', reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        axis = list(range(inputs_rank - 1))
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta = variables.model_variable('beta',
                                            shape=params_shape,
                                            dtype=dtype,
                                            initializer=init_ops.zeros_initializer,
                                            collections=beta_collections,
                                            trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections,
                                                               'gamma')
            gamma = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=init_ops.ones_initializer,
                                             collections=gamma_collections,
                                             trainable=trainable)
        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections.
        moving_mean_collections = utils.get_variable_collections(
            variables_collections, 'moving_mean')
        moving_mean = variables.model_variable(
            'moving_mean',
            shape=params_shape,
            dtype=dtype,
            initializer=init_ops.zeros_initializer,
            trainable=False,
            collections=moving_mean_collections)
        moving_variance_collections = utils.get_variable_collections(
            variables_collections, 'moving_variance')
        moving_variance = variables.model_variable(
            'moving_variance',
            shape=params_shape,
            dtype=dtype,
            initializer=init_ops.ones_initializer,
            trainable=False,
            collections=moving_variance_collections)

        # If `is_training` doesn't have a constant value, because it is a `Tensor`,
        # a `Variable` or `Placeholder` then is_training_value will be None and
        # `needs_moments` will be true.
        is_training_value = utils.constant_value(is_training)
        need_moments = is_training_value is None or is_training_value
        if need_moments:
            # Calculate the moments based on the individual batch.
            mean, variance = nn.moments(inputs, axis, shift=moving_mean)
            moving_vars_fn = lambda: (moving_mean, moving_variance)
            if updates_collections is None:
                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay)
                    with ops.control_dependencies([update_moving_mean,
                                                   update_moving_variance]):
                        return array_ops.identity(mean), array_ops.identity(variance)

                mean, variance = utils.smart_cond(is_training,
                                                  _force_updates,
                                                  moving_vars_fn)
            else:
                def _delay_updates():
                    """Internal function that delay updates moving_vars if is_training."""
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay)
                    return update_moving_mean, update_moving_variance

                update_mean, update_variance = utils.smart_cond(is_training,
                                                                _delay_updates,
                                                                moving_vars_fn)
                ops.add_to_collections(updates_collections, update_mean)
                ops.add_to_collections(updates_collections, update_variance)
                # Use computed moments during training and moving_vars otherwise.
                vars_fn = lambda: (mean, variance)
                mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
        else:
            mean, variance = moving_mean, moving_variance
        # Compute batch_normalization.
        outputs = nn.batch_normalization(
            inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
