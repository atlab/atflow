import tempfile
from os import makedirs
from os.path import join, isdir

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from atflow import constraints


class Trainer:
    def __init__(self, loss, inputs, targets, is_training=None,
                 session=None, session_config=None,
                 log_dir=None,
                 add_summary=False,
                 regularize=True, constrain=True,
                 optimizer_op=tf.train.AdamOptimizer, optimizer_config=None):
        self.loss = loss
        self.graph = loss.graph
        self.inputs_ = inputs
        self.targets_ = targets
        self.is_training_ = is_training
        self.session = session
        self.session_config = session_config

        self._log_dir = None


        self.log_dir = log_dir

        self.add_summary = add_summary

        self.regularize = regularize
        self.constrain = constrain
        self.optimizer_op = optimizer_op
        self.optimizer_config = optimizer_config or {}
        self.build()
        self.init_session()

    def add_gradient_summary(self):
        """
        Add histogram summary for each variable, their gradients, and the gradient norm
        """
        # Compute gradients
        gradients = self.gradients

        # Add histograms for variables, gradients and gradient norms.
        for gradient, variable in gradients:
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            tf.histogram_summary(variable.name, variable)
            tf.histogram_summary(variable.name + "/gradients", grad_values)
            tf.histogram_summary(variable.name + "/gradient_norm",
                                 tf.global_norm([grad_values]))

    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        if log_dir is None:
            log_dir = tempfile.mkdtemp()
        self._log_dir = log_dir
        # create directory if not present
        if not isdir(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        self.train_summary = tf.train.SummaryWriter(join(self.log_dir, 'train'), graph=self.graph)
        self.validation_summary = tf.train.SummaryWriter(join(self.log_dir, 'validation'))
        self.test_summary = tf.train.SummaryWriter(join(self.log_dir, 'test'))

    @property
    def checkpoint_dir(self):
        return join(self.log_dir, 'checkpoints')

    def build(self):
        """
        Builds the graph.
        """
        with self.graph.as_default():
            self.total_loss = self.loss
            # add regularizers if present
            if self.regularize and tf.get_collection('regularization'):
                self.total_loss += tf.add_n(tf.get_collection('regularization'))
            tf.scalar_summary('total_loss', self.total_loss)
            self.optimizer = self.optimizer_op(**self.optimizer_config)
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.gradients = self.optimizer.compute_gradients(self.total_loss)
            if self.add_summary:
                self.add_gradient_summary()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # make sure you perform updates before next train step
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            self.summary = tf.merge_all_summaries()

            # configure to save and load all variables except for the global step value
            variables_to_load = [v for v in tf.all_variables() if 'global_step' not in v.name]
            self.saver = tf.train.Saver(max_to_keep=10)
            self.saver_best = tf.train.Saver(variables_to_load, max_to_keep=1)
            self.init = tf.initialize_all_variables()
            self.apply_constraints = constraints.constrain_all_variables()

    def add_custom_saver(self, name_map):
        """
        Add a custom saver to the graph.
        :param name_map: dictionary specifying the name mapping when saving this graph's variables. Mapping should be
        key=name in graph , values=name in checkpoint file
        :return: custom saver object
        """
        with self.graph.as_default():
            name_to_var = {v.op.name: v for v in tf.get_collection('variables')}
            save_map = {old_name: name_to_var[new_name] for new_name, old_name in name_map.items()}
            self.custom_saver = tf.train.Saver(save_map)
            return self.custom_saver

    def restore_custom(self, filename='best', checkpoint_dir=None):
        """
        Restore graph state from the custom checkpoint.
        :param filename:  Name of the custom checkpoint file.
        :param checkpoint_dir: Directory of the custom checkpoint file. Defaults to the self.checkpoint_dir
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        checkpoint_path = join(checkpoint_dir, filename)
        self.custom_saver.restore(self.session, checkpoint_path)

    def save_custom(self, filename, checkpoint_dir=None):
        """
        Save current graph state into the custom checkpoint.
        :param filename: Name of the custom checkpoint file
        :param checkpoint_dir: Directory of the custom checkpoint flie. Defaults to self.checkpoint_dir
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        checkpoint_path = join(checkpoint_dir, filename)
        self.custom_saver.save(self.session, checkpoint_path)

    def init_session(self, config=None):
        """
        Initialize a new session and initialize the graph. All variables are initialized and (if necessary) constrained.
        :param config: Optional session config to be passed in during Session creation
        """
        if config is None:
            config = self.session_config
        self.session = self.session or tf.Session(graph=self.graph, config=config)

        # initialize and then constrain all variables
        self.session.run(self.init)
        if self.constrain:
            self.session.run(self.apply_constraints)
        self.min_total_loss = None

    def save(self, step=None):
        """
        Save current state of the graph into a checkpoint file.
        :param step: step number to tag the checkpoint file with. If not given, defaults to the value of global_step variable.
        """
        if step is None:
            step = self.session.run(self.global_step)
        self.saver.save(self.session, join(self.checkpoint_dir, 'step'), global_step=step)

    def restore(self, step=None):
        """
        Restore saved checkpoint from the specified step. If step is not given, restores the latest
        checkpoint.
        :param step: global step value of the checkpoint to be loaded
        """
        if step is None:
            filename = tf.train.latest_checkpoint(self.checkpoint_dir)
        else:
            filename = join(self.checkpoint_dir, 'step%d' % step)
        self.saver.restore(self.session, filename)

    def save_best(self):
        """
        Save the current state of the graph as the "best" state.
        """
        self.saver_best.save(self.session, join(self.checkpoint_dir, 'best'))

    def restore_best(self):
        """
        Restores the last saved "best" state of the graph.
        """
        self.saver_best.restore(self.session, join(self.checkpoint_dir, 'best'))

    @property
    def last_checkpoints(self):
        return self.saver.last_checkpoints

    def make_feed_dict(self, inputs=None, targets=None, is_training=False, feed_dict=None):
        if feed_dict is not None:
            fd = dict(feed_dict)
        else:
            fd = {}

        # feed in inputs
        if inputs is not None:
            nest.assert_same_structure(self.inputs_, inputs)
            inputs_ph_list = nest.flatten(self.inputs_)
            inputs_list = nest.flatten(inputs)

            for ph, val in zip(inputs_ph_list, inputs_list):
                fd[ph] = val

        # feed in targets
        if targets is not None:
            nest.assert_same_structure(self.targets_, targets)
            targets_ph_list = nest.flatten(self.targets_)
            targets_list = nest.flatten(targets)

            for ph, val in zip(targets_ph_list, targets_list):
                fd[ph] = val

        fd[self.is_training_] = is_training
        # remove any entries with None as value
        fd = {k: v for k,v in fd.items() if v is not None}
        return fd

    def train(self, dataset, batch_size=256, max_steps=1000, feed_dict=None,
              train_summary_freq=10,
              validation_freq=50,
              validation_summary_freq=None,
              early_stopping_steps=5,
              burn_in_steps=None,
              test_freq=100,
              save_freq=500,
              load_best=True,
              callback_fn=None):
        """
        Train the network.
        :param dataset: Dataset object containing inputs and targets
        :param batch_size: batch size of the training.
        :param max_steps: Max number of batches to train before terminating training. There will be no
        capping if max_steps <= 0
        :param feed_dict: Optional feed_dict to feed values into tensors in the Graph. Use this if the Graph contains
        placeholders not managed by the Trainer (e.g. inputs, targets, and is_training).
        :param train_summary_freq: How often to write training summary. Never if train_summary_freq <= 0
        :param validation_freq: How often to perform validation check for early stopping. If validation_freq <= 0,
        validation check is not performed. If validation_freq <= 0 and max_steps <=0, this may lead to non-stopping
        training loop!
        :param validation_summary_freq:  How often to write validation summary. Never if validation_summary_freq <= 0.
        If None (default), write summary at every validation check.
        :param early_stopping_steps: How many unsuccessful validation checks to perform before early stopping.
        :param burn_in_steps: Number of steps to skip validation checks at the beginning of training. Defaults to
        4 * validation_freq. Only applies to the globa_step value.
        :param test_freq: How often to check on testset and write summary. Never if test_freq <= 0.
        :param save_freq: How often to save the current state of the graph. Never if save_freq <= 0.
        :param load_best: If True, loads the network state with best validation score at the end of the training.
        :param callback_fn: Optional call back function to be called at the end of each batch. The function signature
        will be callback_fn(trainer, step, global_step), where trainer is this Trainer object, step is the step count
        during this training, and global_step is the value of the global_step variable.
        :return: the minimum total loss achieved during the training.
        """

        # Handle absence of validation and/or test set
        if dataset.n_test_samples == 0:
            # skip testing if no test set present
            test_freq = 0

        if dataset.n_validation_samples == 0:
            # skip validation if no validation set present
            validation_freq = 0



        validation_summary_freq = validation_summary_freq or validation_freq
        burn_in_steps = burn_in_steps if burn_in_steps is not None else validation_freq * 4

        dataset.next_epoch() # initialize dataset

        sess = self.session
        train_step = self.train_step

        # number of checks performed without update observed
        checks_without_update = 0

        # save validation score at the very beginning
        if validation_freq > 0 and self.min_total_loss is None:
            validation_inputs, validation_targets = dataset.validation_set
            fd = self.make_feed_dict(validation_inputs, validation_targets, is_training=False, feed_dict=feed_dict)
            total_loss = sess.run(self.total_loss, feed_dict=fd)
            print('\rInitial validation cost=%.5f' % total_loss, flush=True)
            self.min_total_loss = total_loss
            self.save_best()

        step = 0
        while step < max_steps or max_steps <= 0:
            if save_freq > 0 and step % save_freq == 0:
                self.save()

            batch_inputs, batch_targets = dataset.minibatch(batch_size)
            fd = self.make_feed_dict(batch_inputs, batch_targets, is_training=True, feed_dict=feed_dict)

            _, summary, global_step = sess.run([train_step, self.summary, self.global_step], feed_dict=fd)

            # constrain variables
            if self.constrain:
                self.session.run(self.apply_constraints)

            # invoke call back function after each training step
            if callback_fn is not None: callback_fn(self, step, global_step)

            # generate training summary every train_summary_freq
            if train_summary_freq > 0 and step % train_summary_freq == 0:
                self.train_summary.add_summary(summary, global_step=global_step)

            # generate test summary every test_freq
            if test_freq > 0 and step % test_freq == 0:
                test_inputs, test_targets = dataset.test_set
                fd = self.make_feed_dict(test_inputs, test_targets, is_training=False, feed_dict=feed_dict)
                summary = sess.run(self.summary, feed_dict=fd)
                self.test_summary.add_summary(summary, global_step=global_step)

            # run validation, generate validation summary, and early stop if necessary
            if validation_freq > 0 and global_step > burn_in_steps and step % validation_freq == 0:
                validation_inputs, validation_targets = dataset.validation_set
                fd = self.make_feed_dict(validation_inputs, validation_targets, is_training=False, feed_dict=feed_dict)

                # output validation summary every validation_summary_freq
                if validation_summary_freq > 0 and step % validation_summary_freq == 0:
                    total_loss, summary = sess.run([self.total_loss, self.summary], feed_dict=fd)
                    self.validation_summary.add_summary(summary, global_step=global_step)
                else:
                    total_loss = sess.run(self.total_loss, feed_dict=fd)
                print('\rGlobal Step %04d Step %04d: validation cost=%.5f' % (global_step, step, total_loss),
                      flush=True, end='')

                # perform early stopping check
                if self.min_total_loss is None or total_loss < self.min_total_loss:
                    self.min_total_loss = total_loss
                    checks_without_update = 0
                    print(' Updated min total loss! Saving...')
                    self.save_best()
                else:
                    checks_without_update += 1
                    if checks_without_update == early_stopping_steps:
                        print('\nEarly stopping!')
                        break

            # get new step value (incremented due to training step)
            step += 1

        # Restore the best parameter
        if load_best:
            print('Restoring the best parameters')
            self.restore_best()

        return self.min_total_loss

    def evaluate(self, inputs=None, targets=None, ops=None, is_training=False, feed_dict=None):
        """
        Evaluate the network on OPS
        :param inputs: inputs data
        :param targets: targets data
        :param ops: targets (list of) ops to evaluate. Defaults to total_loss
        :param is_training: boolean indicating whether network is in training mode
        :param feed_dict: optional feed_dict to use when evaluating
        :return: return values of the specified OPS.
        """
        if ops is None:
            ops = self.total_loss
        fd = self.make_feed_dict(inputs=inputs, targets=targets, is_training=is_training, feed_dict=feed_dict)
        return self.session.run(ops, feed_dict=fd)

class LossOpTrainer(Trainer):
    def __init__(self, loss_op, input_shape=None, target_shape=None,
                 graph=None, **kwargs):
        self.loss_op = loss_op
        graph = graph if graph is not None else tf.Graph()
        with graph.as_default():
            flat_inputs_ = []
            flat_targets_ = []
            for i, shape in enumerate(nest.flatten(input_shape)):
                flat_inputs_.append(tf.placeholder(tf.float32, shape=shape, name='inputs_%d' % i))
            inputs_ = nest.pack_sequence_as(input_shape, flat_inputs_)
            for i, shape in enumerate(nest.flatten(target_shape)):
                flat_targets_.append(tf.placeholder(tf.float32, shape=shape, name='targets_%d' % i))
            targets_ = nest.pack_sequence_as(target_shape, flat_targets_)
            is_training_ = tf.placeholder(tf.bool, name='is_training')
            loss_ = self.loss_op(inputs_, targets_, is_training_)

        super().__init__(loss_, inputs_, targets_, is_training=is_training_, **kwargs)

    def init_session(self, *args, **kwargs):
        super().init_session(*args, **kwargs)
        self.loss_op.session = self.session



class Scheduler:
    """
    Utility object to keep track of training hyperparameter schedule.
    """
    def __init__(self):
        self.schedule_map = {}

    def add_schedule(self, feature, schedule):
        times = np.sort(np.array(list(schedule.keys())))
        self.schedule_map[feature] = (times, schedule)

    def get_value(self, feature, step=0):
        times, schedule = self.schedule_map[feature]
        idx = times[times <= step].max()
        return schedule[idx]

    def get_all(self, step=0):
        values = {}
        for k in self.schedule_map:
            values[k] = self.get_value(k, step)

        return values

