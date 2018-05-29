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

from tensorflow.python.framework import dtypes





class AdafactorOptimizer(tf.train.Optimizer):
  """Optimizer that implements the Adafactor algorithm.

  Adafactor is similar to RMSProp (ADAM, etc.), but takes advantage of the
  structure of weight matrices to use less memory and to be more resilient to
  sudden large gradients.

  The RMSProp algorithm works on each component independently as follows:
    w -= grad * learning_rate / sqrt(estimated_mean_square_grad)

    learning_rate is the desired update magnitude, and
    estimated_mean_square_grad is computed by exponential smoothing of the
    square of the gradient.

  Adafactor addresses two shortcomings of RMSProp:

  1. In RMSProp (ADAM, etc), maintaining estimated_mean_square_grad requires
     memory equal to the number of parameters.  This can be an impediment to
     training large models on GPU/TPU systems with limited memory.

     Adafactor uses less memory.
     For an AxB weight matrix, instead of keeping a full AxB
     estimated_mean_square_grad matrix, Adafactor keeps only
     exponentially-smoothed row and column means, and bases its estimates on
     those means.   Thus the memory requirements drop from `2AB` to `A+B`.

  2. Depending on the decay rate of the exponential smoothing, we run into one
     of two problems.

     If the decay rate is high (short memory), we see the problem described
     here - worse final quality:
       On the Convergence of Adam and Beyond
       https://openreview.net/forum?id=ryQu7f-RZ

     If the decay rate is low (long memory), then the estimate does not adjust
     rapidly to suddenly large gradients, and the model diverges.
     Suddenly large gradients (which we will call anomalies), may happen either
     due to weird training data, or because the model has just learned something
     important and can now rush to exploit it.  Momentum (as in ADAM) can help
     prevent divergence, but it also requires more memory.  Gradient clipping
     can also help prevent divergence, but it is irritating in that setting
     the right threshold depends on the knowing the scale of the gradients.

     Adafactor uses a relatively long memory (setting the decay rate to
     step_num^-0.8), but detects and corrects for anomalies.   An anomaly
     is detected if the mean-square gradient for the current step
     (across the entire weight matrix) is much greater than the historical
     average.  When this occurs, we increase estimated_mean_square_grad
     for the current step for all weights in the matrix.  Note: it is important
     to detect anomalies based on entire matrices, rather than individual
     weights, since any individual weight may legitimately have a pattern
     of many small gradients and occasional very large ones.
  HYPERPARAMETERS:
    learning_rate: desired magnitude of variable updates.  a scalar - can be a
      constant, but more likely should have a warmup and then decay
      proportionally to rsqrt(step_num)
    epsilon: 1e-20 - a small floating point value to avoid division by zero.
    horizon_exponent: 0.8 - a value between 0 and 1 - The effective decay
      horizon of the second-moment estimator is step_num^horizon_exponent.
    anomaly_threshold: 2.0 - a value greater than 1.  Suppress anomalies
      where the mean-square-gradients for a step exceed the long-term average
      by at least this factor.

  ALGORITHM:

  We initialize
  ```
  t <- 0
  if var is 2-dimensional:
    v_r <- zeros([num_rows])
    v_c <- zeros([num_cols])
  else:
    v <- zeros(shape(var))
  ```

  The update rule is as follows:
  ```
  t <- t + 1
  decay_rate = 1 - t ^ (-horizon_exponent)
  grad_squared = tf.square(grad) + epsilon
  if var is 2-dimensional:
    v_r <- decay_rate * v_r + (1 - decay_rate) * reduce_mean(grad_squared, 1)
    v_c <- decay_rate * v_c + (1 - decay_rate) * reduce_mean(grad_squared, 0)
    anomaly_factor = max(1.0,
      reduce_mean(grad_squared) / reduce_mean(v_r) / anomaly_threshold)
    est_v = anomaly_factor * outer_prod(v_r, v_c) / reduce_mean(v_r)
  else:
    v <- decay_rate * v + (1 - decay_rate) * grad_squared
    anomaly_factor = max(1.0,
      reduce_mean(grad_squared) / reduce_mean(v) / anomaly_threshold)
    est_v = v * anomaly_factor
  var <- var - lr * grad / sqrt(est_v)
  ```
  TODO(noam): write a paper.
  TODO(noam): we should also apply the 2d logic to the two final dimensions.
    of >2d convolutional kernels.
  """

  def __init__(self,
               learning_rate=0.001,
               epsilon=1e-20,
               horizon_exponent=0.8,
               anomaly_threshold=2.0,
               use_locking=False,
               name="Adafactor"):
    """Construct a new Adafactor optimizer.

    See class comment.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      epsilon: A small constant for numerical stability.
      horizon_exponent: a floating point value between 0 and 1
      anomaly_threshold: a floating point value >= 1.0
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "AdafactorOptimizer".
    """
    super(AdafactorOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._epsilon = epsilon
    self._horizon_exponent = horizon_exponent
    self._anomaly_threshold = anomaly_threshold

  def _should_use_factored_second_moment_estimate(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.

    Args:
      shape: a list of integers
    Returns:
      a boolean
   """
    return len(shape) == 2

  def _create_slots(self, var_list):
    for v in var_list:
      shape = v.get_shape().as_list()
      if self._should_use_factored_second_moment_estimate(shape):
        r_val = tf.zeros([shape[0]], dtype=tf.float32)
        c_val = tf.zeros([shape[1]], dtype=tf.float32)
        self._get_or_make_slot(v, r_val, "vr", self._name)
        self._get_or_make_slot(v, c_val, "vc", self._name)
    else:
        self._zeros_slot(v, "v", self._name)

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    grad_squared = tf.square(grad) + self._epsilon
    grad_squared_mean = tf.reduce_mean(grad_squared)
    lr = tf.to_float(self._lr)
    global_step = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
    # HACK: Make lr and global_step dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    lr += grad_squared_mean * 1e-30
    global_step += grad_squared_mean * 1e-30
    # END HACK
    mixing_rate = tf.pow(global_step, -self._horizon_exponent)
    decay_rate = 1.0 - mixing_rate
    shape = var.get_shape().as_list()
    updates = []
    if self._should_use_factored_second_moment_estimate(shape):
      grad_squared_row_mean = tf.reduce_mean(grad_squared, 1)
      grad_squared_col_mean = tf.reduce_mean(grad_squared, 0)
      vr = self.get_slot(var, "vr")
      new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
      vc = self.get_slot(var, "vc")
      new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
      vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
      vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
      updates = [vr_update, vc_update]
      long_term_mean = tf.reduce_mean(new_vr)
      anomaly_factor = self._anomaly_factor(grad_squared_mean, long_term_mean)
      # This is the computation we should do.
      # est_v = (tf.expand_dims(new_vr, 1) * tf.expand_dims(new_vc, 0)
      #          * anomaly_factor / long_term_mean)
      # subtrahend = grad * lr / tf.sqrt(est_v)
      # Instead we do the following, which is mathematically equivalent.
      r_factor = lr * tf.rsqrt(new_vr * anomaly_factor / long_term_mean)
      c_factor = tf.rsqrt(new_vc)
      subtrahend = (
          grad * tf.expand_dims(r_factor, 1) * tf.expand_dims(c_factor, 0))
    else:
      v = self.get_slot(var, "v")
      new_v = decay_rate * v + mixing_rate * grad_squared
      v_update = tf.assign(v, new_v, use_locking=self._use_locking)
      updates = [v_update]
      long_term_mean = tf.reduce_mean(new_v)
      anomaly_factor = self._anomaly_factor(grad_squared_mean, long_term_mean)
      # This is the computation we should do.
      # est_v = (new_v * anomaly_factor)
      # subtrahend = grad * lr / tf.sqrt(est_v)
      # Instead we do the following, which is mathematically equivalent.
      subtrahend = grad * (lr / tf.sqrt(anomaly_factor)) * tf.rsqrt(new_v)
    var_update = tf.assign_sub(var, subtrahend, use_locking=self._use_locking)
    updates = [var_update] + updates
    return tf.group(*updates)

  def _anomaly_factor(self, grad_squared_mean, long_term_mean):
    """Multiplier for second-moment estimator, due to short-term anomalies.

    A step may have gradients with magnitudes much larger than the long-term
    average.  This can cause the model to diverge.  In these cases, we want to
    temoporarily increase the second-moment estimators to reflect that these
    steps are anomalous.

    It is important to make these calculations on whole weight matrices, rather
    than on individual parameters, since we want to allow individual parameters
    to have occasional large updates.

    Args:
      grad_squared_mean: A scalar.  The mean square gradient on the varaible
         for the current step.
      long_term_mean: A scalar.  The mean of the long-term second-moment
         estimator.
    Returns:
      a scalar that should be multiplied into the second-moment-estimator for
      this step.
    """
    ratio = grad_squared_mean / long_term_mean
    return tf.maximum(1.0, ratio / self._anomaly_threshold)


