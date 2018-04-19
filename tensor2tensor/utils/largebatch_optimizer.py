# coding=utf-8

"""Optimizer variants which makes it possible to use very large batch sizes with
limited GPU memory. This optimizer accumulates the gradients for n batches, and
calls the actual optimizer every n batches with the accumulated gradients.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope


class LargebatchAdamOptimizer(tf.contrib.opt.LazyAdamOptimizer):
  """Large batch variant for Adam."""

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam", n=2):
    super(LargebatchAdamOptimizer, self).__init__(
      learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, 
      use_locking=use_locking, name=name)
    self._n = n  # Call Adam optimizer every n batches with accumulated grads
    self._n_t = None  # Tensor version
    self._iter = None  # If zero, call Adam. Otherwise, accumulate gradients

  def _create_slots(self, var_list):
    """Like super class method, but additionally creates slots for
    the gradient accumulator `acc_grad` and the counter variable.
    """
    first_var = min(var_list, key=lambda x: x.name)

    if (self._beta1_power is None or
        self._beta1_power.graph is not first_var.graph):
      with ops.colocate_with(first_var):
        self._beta1_power = variable_scope.variable(self._beta1,
                                                    name="beta1_power",
                                                    trainable=False)
        self._beta2_power = variable_scope.variable(self._beta2,
                                                    name="beta2_power",
                                                    trainable=False)
        self._iter = variable_scope.variable(0,
                                             name="iter",
                                             trainable=False,
                                             dtype=tf.int32)
    # Create slots for the first and second moments and grad accumulator.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)
      self._zeros_slot(v, "grad_acc", self._name)

  def _prepare(self):
    super(LargebatchAdamOptimizer, self)._prepare()
    self._n_t = ops.convert_to_tensor(self._n, name="n")

  def _apply_cond(self, apply_fn, grad, var, *args,**kwargs):
    """Call `apply_fn only if the current counter value (iter) is zero. This
    method couples common functionality for all _apply_*() implementations
    in Adam.

    Args:
      apply_fn: Callback function for applying gradients.
      grad: Gradients (grad variable in _apply_*() methods)
      var: variable (var variable in _apply_*() methods)
      *args: Passed through to `apply_fn`
      **kwargs: Passed through to `apply_fn`
    
    Returns:
      Adam op for applying gradients if iter=0, or an op for accumulating the
      gradient in the `grad_acc` slot otherwise.
    """
    grad_acc = self.get_slot(var, "grad_acc")

    def apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs):
      total_grad = (grad_acc + grad) / math_ops.cast(self._n_t, 
                                                     grad.dtype.base_dtype)
      adam_op = apply_fn(total_grad, var, *args, **kwargs)
      with ops.control_dependencies([adam_op]):
        grad_acc_to_zero_op = grad_acc.assign(grad_acc * 0.0,
                                              use_locking=self._use_locking)
      return control_flow_ops.group(adam_op, grad_acc_to_zero_op)

    def accumulate_gradient(grad_acc, grad):
      assign_op = tf.assign_add(grad_acc, grad, use_locking=self._use_locking)
      return control_flow_ops.group(assign_op)  # Strip return value
      
    return tf.cond(tf.equal(self._iter, 0),
                   lambda: apply_adam(
                     grad_acc, apply_fn, grad, var, *args, **kwargs),
                   lambda: accumulate_gradient(grad_acc, grad))

  def _apply_dense(self, grad, var):
    return self._apply_cond(
      super(LargebatchAdamOptimizer, self)._apply_dense, grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._apply_cond(
      super(LargebatchAdamOptimizer, self)._resource_apply_dense, grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    return self._apply_cond(
      super(LargebatchAdamOptimizer, self)._apply_sparse_shared, grad, var,
      indices, scatter_add)

  def _apply_sparse(self, grad, var):
    # TODO: Implement a sparse version of that
    dense_grad = tf.convert_to_tensor(grad)
    return self._apply_cond(
      super(LargebatchAdamOptimizer, self)._apply_dense, dense_grad, var)

  def _finish(self, update_ops, name_scope):
    """Like super class method, but updates beta_power variables only every
    n batches. The iter variable is updated with
      iter <- iter + 1 mod n
    """
    with ops.control_dependencies(update_ops):
      with ops.colocate_with(self._iter):

        def update_beta_op():
          update_beta1 = self._beta1_power.assign(
              self._beta1_power * self._beta1_t,
              use_locking=self._use_locking)
          update_beta2 = self._beta2_power.assign(
              self._beta2_power * self._beta2_t,
              use_locking=self._use_locking)
          return control_flow_ops.group(update_beta1, update_beta2)
        maybe_update_beta = tf.cond(tf.equal(self._iter, 0),
          lambda: update_beta_op(),
          lambda: tf.no_op())
        with ops.control_dependencies([maybe_update_beta]):
          update_iter = self._iter.assign(tf.mod(self._iter + 1, self._n_t), 
                                          use_locking=self._use_locking)
    return control_flow_ops.group(
      *update_ops + [update_iter, maybe_update_beta], name=name_scope)

