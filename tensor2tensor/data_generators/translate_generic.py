"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


# Data-set location

_TRAIN_DATASETS = [("train/training.src",
                     "train/training.trg")]

_TEST_DATASETS = [("dev/dev.src",
     "dev/dev.trg")]

_BPE_VOCABS = [("bpe/bpe.src",
     "bpe/bpe.trg")]


class TranslateGenericProblem(problem.Text2TextProblem):
  """Base class for translation problems."""

  @property
  def is_character_level(self):
    return False

  @property
  def num_shards(self):
    return 100

  @property
  def vocab_name(self):
    return "vocab.txt"

  @property
  def use_subword_tokenizer(self):
    return True

  # Space ids are really only used for multi-task models, default is 0=GENERIC
  @property
  def input_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def targeted_vocab_size(self):
    return FLAGS.vocab_size


# Generic generators used later for multiple problems.
def bi_vocabs_token_generator(source_path,
                              target_path,
                              source_token_vocab,
                              target_token_vocab,
                              eos=None):
  """Generator for sequence-to-sequence tasks that uses tokens.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are token ids from the " "-split source (and target, resp.) lines
  converted to integers using the token_map.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    source_token_vocab: text_encoder.TextEncoder object.
    target_token_vocab: text_encoder.TextEncoder object.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        source_ints = source_token_vocab.encode(source.strip()) + eos_list
        target_ints = target_token_vocab.encode(target.strip()) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}
        source, target = source_file.readline(), target_file.readline()


# Generators.

def _preprocess_sgm(line, is_sgm):
  """Preprocessing to strip tags in SGM files."""
  if not is_sgm:
    return line
  # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
  if line.startswith("<srcset") or line.startswith("</srcset"):
    return ""
  if line.startswith("<doc") or line.startswith("</doc"):
    return ""
  if line.startswith("<p>") or line.startswith("</p>"):
    return ""
  # Strip <seg> tags.
  line = line.strip()
  if line.startswith("<seg") and line.endswith("</seg>"):
    i = line.index(">")
    return line[i+1:-6]  # Strip first <seg ...> and last </seg>.


def _compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  with tf.gfile.GFile(filename + ".src", mode="w") as src_resfile:
    with tf.gfile.GFile(filename + ".trg", mode="w") as trg_resfile:
      for dataset in datasets:
        orig_dir = FLAGS.raw_data_dir
        src_filename, trg_filename = dataset
        src_filepath = os.path.join(orig_dir, src_filename)
        trg_filepath = os.path.join(orig_dir, trg_filename)
        is_sgm = (src_filename.endswith("sgm") and
                  trg_filename.endswith("sgm"))

        if not (os.path.exists(src_filepath) and
                os.path.exists(trg_filepath)):
            raise ValueError("Can't find files %s %s." % (src_filepath, trg_filepath))
        if src_filepath.endswith(".gz"):
          new_filepath = src_filepath.strip(".gz")
          generator_utils.gunzip_file(src_filepath, new_filepath)
          src_filepath = new_filepath
        if trg_filepath.endswith(".gz"):
          new_filepath = trg_filepath.strip(".gz")
          generator_utils.gunzip_file(trg_filepath, new_filepath)
          trg_filepath = new_filepath
        with tf.gfile.GFile(src_filepath, mode="r") as src_file:
          with tf.gfile.GFile(trg_filepath, mode="r") as trg_file:
            line1, line2 = src_file.readline(), trg_file.readline()
            while line1 or line2:
              line1res = _preprocess_sgm(line1, is_sgm)
              line2res = _preprocess_sgm(line2, is_sgm)
              if line1res or line2res:
                src_resfile.write(line1res.strip() + "\n")
                trg_resfile.write(line2res.strip() + "\n")
              line1, line2 = src_file.readline(), trg_file.readline()

  return filename


@registry.register_problem
class TranslateGeneric(TranslateGenericProblem):
  """Problem spec for generic wordpiece translation."""

  @property
  def source_vocab_name(self):
    return "vocab.src.%d" % self.targeted_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab.trg.%d" % self.targeted_vocab_size

  def generator(self, data_dir, tmp_dir, train):
    datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
    source_datasets = [[FLAGS.raw_data_dir, [item[0]]] for item in datasets]
    target_datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in datasets]
    source_vocab = generator_utils.get_or_generate_vocab_nocompress(
        data_dir, self.source_vocab_name,
        self.targeted_vocab_size, source_datasets)
    target_vocab = generator_utils.get_or_generate_vocab_nocompress(
        data_dir, self.target_vocab_name,
        self.targeted_vocab_size, target_datasets)
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "generic_tok_%s" % tag)
    return bi_vocabs_token_generator(data_path + ".src", data_path + ".trg",
                                     source_vocab, target_vocab, EOS)

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir,
                                         self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir,
                                         self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }


@registry.register_problem
class TranslateGenericBpe(TranslateGenericProblem):
  """Problem spec for generic translation, BPE version."""

  @property
  def source_vocab_name(self):
    return "vocab.bpe.src.%d" % self.targeted_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab.bpe.trg.%d" % self.targeted_vocab_size

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_encoder = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov="UNK")
    target_encoder = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov="UNK")
    return {"inputs": source_encoder, "targets": target_encoder}

  def generator(self, data_dir, tmp_dir, train):
    datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
    source_datasets = [[FLAGS.raw_data_dir, [item[0]]] for item in datasets]
    target_datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in datasets]
    # BPE vocab
    source_vocab_path = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_path = os.path.join(data_dir, self.target_vocab_name)
    tf.gfile.Copy(os.path.join(FLAGS.raw_data_dir, _BPE_VOCABS[0]), source_vocab_path)
    tf.gfile.Copy(os.path.join(FLAGS.raw_data_dir, _BPE_VOCABS[1]), target_vocab_path)
    with tf.gfile.GFile(source_vocab_path, mode="a") as f:
      f.write("UNK\n")  # Add UNK to the vocab.
    with tf.gfile.GFile(target_vocab_path, mode="a") as f:
      f.write("UNK\n")  # Add UNK to the vocab.
    source_token_vocab = text_encoder.TokenTextENcoder(source_vocab_path, replace_oov="UNK")
    target_token_vocab = text_encoder.TokenTextENcoder(target_vocab_path, replace_oov="UNK")
    data_path = _compile_data(tmp_dir, datasets, "generic_tok_%s" % tag)
    return bi_vocabs_token_generator(data_path + ".src", data_path + ".trg",
                                     source_token_vocab, target_token_vocab, EOS)
