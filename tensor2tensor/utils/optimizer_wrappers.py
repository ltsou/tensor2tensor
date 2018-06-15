# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import os
from tensor2tensor.utils import yellowfin

import tensorflow as tf
from tensorflow.python.util import nest

from tensorflow.python.framework import dtypes
from tensorflow.python.eager import context

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

class ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams, use_tpu=False):
    if optimizer_name == "Adam" and use_tpu:
      # LazyAdamOptimizer does not work on TPU
      optimizer_name = "TrueAdam"

    tf.logging.info("Using optimizer %s", optimizer_name)

    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr,
          momentum=hparams.optimizer_momentum_momentum,
          use_nesterov=hparams.optimizer_momentum_nesterov)
    elif optimizer_name == "YellowFin":
      self._opt = yellowfin.YellowFinOptimizer(
          learning_rate=lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "TrueAdam":
      self._opt = tf.train.AdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Adafactor":
      self._opt = AdafactorOptimizer(lr / 500.0)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list=None, **kwargs):
    return self._opt.compute_gradients(loss, var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._opt.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)



class EWCOptimizer(ConditionalOptimizer):
  def __init__(self, optimizer_name, lr, hparams, use_tpu=False):
    super(EWCOptimizer, self).__init__(optimizer_name, lr, hparams, use_tpu)
    self.load_vars = hparams.ewc_load_vars
    self.save_vars = hparams.ewc_save_vars
    self.ewc_loss_weight = hparams.ewc_loss_weight
    self.model_dir = hparams.model_dir

    self.lag_set = hparams.ewc_lagged_collect
    self.fisher_set = hparams.ewc_fisher_collect

    self.final_step = hparams.train_steps
    self.save_ewc_step = 0
    self.fisher_accum_steps = 1
    self.set_steps(hparams)
    self.cond_name = 'ewc_cond'
    self.ewc_checkpoint = os.path.join(self.model_dir, hparams.ewc_checkpoint)

  def set_steps(self, hparams):
    if self.save_vars:
      self.save_ewc_step = hparams.train_steps - hparams.ewc_fisher_accum_steps
      tf.logging.info('Train {} steps before accumulating fisher'.format(self.save_ewc_step))
      self.fisher_accum_steps = hparams.ewc_fisher_accum_steps
      tf.logging.info('Then accumulate for {} steps'.format(self.fisher_accum_steps))


  def accumulate_fisher_and_lagged(self, grads_and_vars, global_step, name):
    fisher_vars = tf.get_collection(self.fisher_set)
    lagged_vars = tf.get_collection(self.lag_set)
    ewc_ops = [global_step.assign_add(1)]
    scale = 1.0 / self.fisher_accum_steps
    for grad_var_pair, f, l in zip(grads_and_vars, fisher_vars, lagged_vars):
      grad, var = grad_var_pair
      ewc_ops.append(f.assign_add(tf.square(grad) * scale))
      ewc_ops.append(l.assign(var))
    return control_flow_ops.group(*ewc_ops)
      

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      maybe_accumulate_fisher = tf.cond(tf.logical_and(
        tf.constant(self.save_vars, dtype=tf.bool),
        tf.greater_equal(global_step, self.save_ewc_step)),
                                        lambda: self.accumulate_fisher_and_lagged(
                                          grads_and_vars, global_step=global_step, name=name),
                                        lambda: self._opt.apply_gradients(
                                          grads_and_vars, global_step=global_step, name=name),
                                        name=self.cond_name)
      return maybe_accumulate_fisher

  def compute_gradients(self, loss, var_list=None, **kwargs):
    tf.logging.info('Creating lagged variables for EWC loss')
    #self.create_fisher_vars()
    if self.load_vars:
      loss += self.get_ewc_loss()
    return self._opt.compute_gradients(loss, var_list, **kwargs)

  def create_fisher_vars(self):
    for idx, var in enumerate(tf.trainable_variables()):
      with tf.variable_scope('lagged'):
        lagged_var = tf.Variable(tf.zeros_like(var), trainable=False)
      with tf.variable_scope('fisher'):
        fisher_var = tf.Variable(tf.zeros_like(var), trainable=False)
      tf.add_to_collection(self.lag_set, lagged_var)
      tf.add_to_collection(self.fisher_set, fisher_var)


  def check_checkpoint_vars(self):
    '''May be saving EWC variables to a model previously trained without them
    if so, need to add EWC vars to the checkpoint: this is a hacky way to do so.
    For some reason this has to be done on CPU, but once we have a checkpoint 
    with the needed variables, accumulation can be done on GPU.
    '''
    new_vars = tf.get_collection(self.lag_set) + tf.get_collection(self.fisher_set)
    ewc_checkpoint = self.ewc_checkpoint
    tf.logging.info('Checking checkpoint {} for EWC variables'.format(ewc_checkpoint))
    reader = tf.train.NewCheckpointReader(ewc_checkpoint)
    sample_var_name = new_vars[0].name.split(':')[0]
    if not reader.has_tensor(sample_var_name):
      tf.logging.info('{} not found: adding EWC vars to checkpoint file'.format(
        new_vars[0].name))
      new_var_saver = tf.train.Saver(new_vars)
      with tf.Session() as s:
        s.run(tf.variables_initializer(new_vars))
        new_var_saver.save(s, ewc_checkpoint, write_state=False)
      '''
      restorer = self.get_old_restorer(do_not_restore=new_vars)
      saver = tf.train.Saver()
      with tf.Session() as s:
        s.run(tf.variables_initializer(new_vars))
        restorer.restore(s, checkpoint_file)
        tf.logging.info('Resaving model with EWC variables to {}'.format(ewc_checkpoint))
        saver.save(s, ewc_checkpoint)
      '''

  def get_old_restorer(self, do_not_restore):
    # restore variables from previous checkpoint that are NOT related to EWC
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    restore_ignore_set = set(do_not_restore)
    restore_vars = []
    for v in global_vars:
      if v not in restore_ignore_set:
        if self.cond_name not in v.name: # resets Adam-specific beta_power variables
          restore_vars.append(v)
        else:
          do_not_restore.append(v)
    return tf.train.Saver(restore_vars)
      

  def get_ewc_loss(self):
    tf.logging.info('Adding EWC penalty to loss with lambda {}'.format(self.ewc_loss_weight))
    lagged_vars = tf.get_collection(self.lag_set)
    fisher_vars = tf.get_collection(self.fisher_set)
    penalty = tf.add_n([tf.reduce_sum(tf.square(l - t) * f)
                        for l, t, f in zip(
                            lagged_vars, tf.trainable_variables(), fisher_vars)])
    ewc_loss = self.ewc_loss_weight * penalty
    return ewc_loss
