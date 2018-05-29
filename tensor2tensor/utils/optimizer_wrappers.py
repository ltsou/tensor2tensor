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

from tensor2tensor.utils import yellowfin

import tensorflow as tf
from tensorflow.python.util import nest

from tensorflow.python.framework import dtypes
from tensorflow.python.eager import context

from tensorflow.python.framework import ops
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
    self.save_ewc_step = hparams.train_steps - hparams.ewc_fisher_accum_steps
    tf.logging.info('we do {} steps before accumulating fisher'.format(self.save_ewc_step))
    self.fisher_accum_batches = hparams.ewc_fisher_accum_steps
    tf.logging.info('we then accumulate for {} steps'.format(self.fisher_accum_batches))
    self.lag_set = hparams.ewc_lagged_collect
    self.fisher_set = hparams.ewc_fisher_collect


  def compute_gradients(self, loss, var_list=None, **kwargs):
    # NB: THIS FUNCTION UPDATES THE FISHER VARS, WHICH MAY THEN BE USED TO CALCULATE THE LOSS
    # TODO: this optimizer should update temporary variables which should only update the
    # working fisher vars at the end of training.
    global_step = tf.train.get_or_create_global_step()
    return tf.cond(tf.greater_equal(global_step, self.save_ewc_step),
                   lambda: self._compute_fisher_vars(loss),
                   lambda: self._opt.compute_gradients(loss, var_list, **kwargs))

 
  def _compute_fisher_vars(self, loss):
    tf.logging.info('possibly returning the fisher vars function')
    train_vars = nest.flatten(tf.trainable_variables())
    fisher_vars = tf.get_collection(self.fisher_set)
    grads = tf.gradients(loss, train_vars) 
    # doesn't work, wants to optimize. maybe should happen at the end instead of being optimized?
    # basically exactly like largebatch adam but 
    # instead of accumulating to var, accumulate square to fisher var.
    # could have a dictionary mapping var to fisher var
    sum_square_grads = [tf.reduce_sum(tf.square(g)) for g in grads]
    grads_and_vars = list(zip(sum_square_grads, fisher_vars))
    return grads_and_vars

  def _finish(self, update_ops, name_scope):
    train_vars = tf.trainable_variables()
    lagged_vars = tf.get_collection(self.lag_set)
    maybe_update_lag_vars = tf.no_op()
    update_ops.append(maybe_update_lag_vars)
    return self._opt._finish(update_ops, name_scope)
