# coding=utf-8
"""ValidationMonitor which stores the best N checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.monitors import ValidationMonitor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib


class ExportingValidationMonitor(ValidationMonitor):
  """Runs evaluation of a given estimator, at most every N steps.
  Note that the evaluation is done based on the saved checkpoint, which will
  usually be older than the current step.
  Can do early stopping on validation metrics if `early_stopping_rounds` is
  provided.
  """

  def __init__(self, x=None, y=None, input_fn=None, batch_size=None,
               eval_steps=None,
               every_n_steps=100, metrics=None, hooks=None,
               early_stopping_rounds=None,
               early_stopping_metric="loss",
               early_stopping_metric_minimize=True, name=None,
               export_metric=None,
               keep_checkpoints=0):
    super(ExportingValidationMonitor, self).__init__(x=x, y=y, input_fn=input_fn, 
        batch_size=batch_size, eval_steps=eval_steps, 
        every_n_steps=every_n_steps, metrics=metrics, hooks=hooks,
        early_stopping_rounds=early_stopping_rounds, 
        early_stopping_metric=early_stopping_metric,
        early_stopping_metric_minimize=early_stopping_metric_minimize, 
        name=name)
    self.export_metric = export_metric
    print("METRICS")
    print(metrics)
    self.keep_checkpoints = keep_checkpoints

  def _scan_best_dir(self, best_dir):
    existing_checkpoints = []
    for file_name in tf.gfile.Glob("%s/model.ckpt-*.index" % best_dir):
      match = re.search(r"score([.0-9]+).index$", file_name)
      if match:
        existing_checkpoints.append((float(match.group(1)), file_name[:-6]))
      else:
        logging.info("Checkpoint %s did not match pattern." % file_name)
    return existing_checkpoints

  def _copy_checkpoint(self, src, trg):
    logging.info("Create new best checkpoint file: %s" % trg)
    for src_file in tf.gfile.Glob("%s.*" % src):
      ext = os.path.splitext(src_file)[1]
      tf.gfile.Copy(src_file, "%s%s" % (trg, ext))

  def _remove_checkpoint(self, path):
    logging.info("Remove from best checkpoint dir: %s" % path)
    for file_path in tf.gfile.Glob("%s.*" % path):
      tf.gfile.Remove(file_path)

  def _rebuild_checkpoint_file(self, path, checkpoints):
    logging.info("Rebuild checkpoint file with %d entries: %s"
      % (len(checkpoints), path))
    prefix = "%s/" % os.path.dirname(path)
    l = len(prefix)
    paths = [p[l:] if p.startswith(prefix) else p for p in checkpoints]
    with tf.gfile.Open(path, "w") as f:
      f.write('model_checkpoint_path: "%s"\n' % paths[-1])
      for checkpoint_path in paths:
        f.write('all_model_checkpoint_paths: "%s"\n' % checkpoint_path)

  def every_n_step_end(self, step, outputs):
    super(ValidationMonitor, self).every_n_step_end(step, outputs)
    # TODO(mdan): The use of step below is probably misleading.
    # The code should probably use the step from the checkpoint, because
    # that's what is being evaluated.
    if self._estimator is None:
      raise ValueError("Missing call to set_estimator.")
    # Check that we are not running evaluation on the same checkpoint.
    latest_path = saver_lib.latest_checkpoint(self._estimator.model_dir)
    if latest_path is None:
      logging.debug("Skipping evaluation since model has not been saved yet "
                    "at step %d.", step)
      return False
    if latest_path is not None and latest_path == self._latest_path:
      logging.debug("Skipping evaluation due to same checkpoint %s for step %d "
                    "as for step %d.", latest_path, step,
                    self._latest_path_step)
      return False
    self._latest_path = latest_path
    self._latest_path_step = step

    # Run evaluation and log it.
    validation_outputs = self._evaluate_estimator()
    stats = []
    for name in validation_outputs:
      stats.append("%s = %s" % (name, str(validation_outputs[name])))
    logging.info("Validation (step %d): %s", step, ", ".join(stats))

    # Store best checkpoints logic
    if self.export_metric:
      best_dir = "%s/best" % os.path.dirname(latest_path)
      tf.gfile.MakeDirs(best_dir)
      print("val outputs")
      print(validation_outputs)
      current_best_checkpoints = self._scan_best_dir(best_dir)
      current_best_checkpoints.append(
        (validation_outputs[self.export_metric], None))
      current_best_checkpoints.sort(key=operator.itemgetter(0),
                                    reverse=self.early_stopping_metric_minimize)
      new_best_paths = []
      rebuild_checkpoint_file = False
      for score, path in current_best_checkpoints[-self.keep_checkpoints:]:
        if path is None:  # This is the new checkpoint
          rebuild_checkpoint_file = True
          path = "%s/model.ckpt-%d_score%.4f" % (best_dir, step, score)
          self._copy_checkpoint(latest_path, path)
        new_best_paths.append(path)
      for _, path in current_best_checkpoints[:-self.keep_checkpoints]:
        if path is not None:
          rebuild_checkpoint_file = True
          self._remove_checkpoint(path)
      if rebuild_checkpoint_file:
        self._rebuild_checkpoint_file("%s/checkpoint" % best_dir,
                                      new_best_paths)

    # Early stopping logic.
    if self.early_stopping_rounds is not None:
      if self.early_stopping_metric not in validation_outputs:
        raise ValueError("Metric %s missing from outputs %s." % (
            self.early_stopping_metric, set(validation_outputs.keys())))
      current_value = validation_outputs[self.early_stopping_metric]
      if (self._best_value is None or (self.early_stopping_metric_minimize and
                                       (current_value < self._best_value)) or
          (not self.early_stopping_metric_minimize and
           (current_value > self._best_value))):
        self._best_value = current_value
        self._best_metrics = copy.deepcopy(validation_outputs)
        self._best_value_step = step
      stop_now = (step - self._best_value_step >= self.early_stopping_rounds)
      if stop_now:
        logging.info("Stopping. Best step: {} with {} = {}."
                     .format(self._best_value_step,
                             self.early_stopping_metric, self._best_value))
        self._early_stopped = True
        return True
    return False

