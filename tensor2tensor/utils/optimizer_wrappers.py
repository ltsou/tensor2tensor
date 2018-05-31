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
    self.final_step = hparams.train_steps
    self.save_ewc_step = hparams.train_steps - hparams.ewc_fisher_accum_steps
    tf.logging.info('we do {} steps before accumulating fisher'.format(self.save_ewc_step))
    self.fisher_accum_steps = hparams.ewc_fisher_accum_steps
    tf.logging.info('we then accumulate for {} steps'.format(self.fisher_accum_steps))
    self.lag_set = hparams.ewc_lagged_collect
    self.fisher_set = hparams.ewc_fisher_collect

  '''
  def compute_gradients(self, loss, var_list=None, **kwargs):
    # NB: THIS FUNCTION UPDATES THE FISHER VARS, WHICH MAY THEN BE USED TO CALCULATE THE LOSS
    # TODO: this optimizer should update temporary variables which should only update the
    # working fisher vars at the end of training.
    global_step = tf.train.get_or_create_global_step()
    return tf.cond(tf.greater_equal(global_step, self.save_ewc_step),
                   lambda: [tf.no_op() for v in var_list],#self._compute_fisher_vars(loss),
                   lambda: self._opt.compute_gradients(loss, var_list, **kwargs))
  '''
 

  def accumulate_fisher(self, grads_and_vars, global_step, name):
    fisher_vars = tf.get_collection(self.fisher_set)
    accumulate_fisher_ops = [global_step.assign_add(1)]
    scale = 1.0 / self.fisher_accum_steps
    for grad_var_pair, f in zip(grads_and_vars, fisher_vars):
      grad, _ = grad_var_pair
      sum_square = tf.reduce_sum(tf.square(grad))
      accumulate_fisher_ops.append(f.assign_add(sum_square * scale))
    return control_flow_ops.group(*accumulate_fisher_ops)
      




  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    maybe_accumulate_fisher = tf.cond(tf.greater_equal(global_step, self.save_ewc_step),
                                      lambda: self.accumulate_fisher(
                                        grads_and_vars, global_step=global_step, name=name),
                                      lambda: self._opt.apply_gradients(
                                        grads_and_vars, global_step=global_step, name=name))
    with ops.control_dependencies([maybe_accumulate_fisher]):
      def update_lagged_op():
        train_vars = tf.trainable_variables()
        lagged_vars = tf.get_collection(self.lag_set)
        update_lagged_ops = []
        for v, t in zip(lagged_vars, train_vars):
          update_lagged_ops.append(v.assign(t))
        return control_flow_ops.group(*update_lagged_ops)

      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      maybe_update_lagged = tf.cond(tf.equal(global_step, self.final_step),
                                    lambda: update_lagged_op(),
                                    lambda: tf.no_op())
      train_op.append(maybe_update_lagged)
      return maybe_update_lagged


  def _apply_dense(self, grad, var):
    return self._apply_cond(
      self._opt._apply_sparse_shared, grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._apply_cond(
      self._opt._apply_sparse_shared, grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    return self._apply_cond(
      self._opt._apply_sparse_shared, grad, var, indices, scatter_add)

  def _apply_sparse(self, grad, var):
    return self._apply_cond(
      self._opt._apply_sparse, grad, var)

  def _create_slots(self, var_list):
    """Like subclass method, but additionally creates slots for
    fisher and lagged parameter updates
    """
    self._opt._create_slots(var_list)
    # Create slots for fisher grad accumulation
    for v in tf.get_collection(self.lag_set):
      self._opt._zeros_slot(v, "fisher_acc", self._name)
      

  def _apply_cond(self, apply_fn, grad, var, *args, **kwargs):
    """Call `apply_fn conditionally
    Args:
      apply_fn: Callback function for applying gradients.
      grad: Gradients (grad variable in _apply_*() methods)
      var: variable (var variable in _apply_*() methods)
      *args: Passed through to `apply_fn`
      **kwargs: Passed through to `apply_fn`
    """
    """
    def accum_fisher(

    """
    return apply_fn(grad, var, *args, **kwargs)
