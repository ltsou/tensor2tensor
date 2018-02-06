"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
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

_VOCABS = ("vocab/vocab.src",
           "vocab/vocab.trg",
           "vocab/vocab.shared")

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
        with tf.gfile.GFile(src_filepath, mode="r") as src_file:
          with tf.gfile.GFile(trg_filepath, mode="r") as trg_file:
            line1, line2 = src_file.readline(), trg_file.readline()
            while line1 or line2:
              src_resfile.write(line1.strip() + "\n")
              trg_resfile.write(line2.strip() + "\n")
              line1, line2 = src_file.readline(), trg_file.readline()

  return filename

def copyVocab(orig_path, target_path):
    # Add <pad> (idx:0), <EOS> (idx:1) and <unk> (idx:2) to vocab
    with tf.gfile.GFile(target_path, mode="w") as f:
        f.write("<pad>\n<EOS>\n<unk>\n");
        with tf.gfile.Open(orig_path) as origF:
            for line in origF:
                tokens = line.strip().split("\t")
                # To handle 3-column vocab file: "<idx>\t<token>\t<count>"
                if len(tokens) > 1:
                    f.write("%s\n" % tokens[1])
                else:
                    f.write("%s\n" % tokens[0])

@registry.register_problem
class TranslateGeneric(translate.TranslateProblem):
  """Problem spec for generic wordpiece translation."""

  @property
  def input_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def targeted_vocab_size(self):
    return FLAGS.targeted_vocab_size

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
    return translate.bi_vocabs_token_generator(data_path + ".src", data_path + ".trg",
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
class TranslateGenericExistingVocab(translate.TranslateProblem):
  """Problem spec for generic translation, using existing vocab """

  @property
  def input_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def source_vocab_name(self):
    return "vocab.src"

  @property
  def target_vocab_name(self):
    return "vocab.trg"

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_encoder = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov="<unk>")
    target_encoder = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov="<unk>")
    return {"inputs": source_encoder, "targets": target_encoder}

  def generator(self, data_dir, tmp_dir, train):
    datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
    source_datasets = [[FLAGS.raw_data_dir, [item[0]]] for item in datasets]
    target_datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in datasets]
    # Copy vocab to data directory
    source_vocab_path = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_path = os.path.join(data_dir, self.target_vocab_name)
    if os.path.exists(source_vocab_path):
        os.remove(source_vocab_path)
    if os.path.exists(target_vocab_path):
        os.remove(target_vocab_path)
    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[0]), source_vocab_path)
    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[1]), target_vocab_path)
    source_token_vocab = text_encoder.TokenTextEncoder(source_vocab_path, replace_oov="<unk>")
    target_token_vocab = text_encoder.TokenTextEncoder(target_vocab_path, replace_oov="<unk>")
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "generic_tok_%s" % tag)
    return translate.bi_vocabs_token_generator(data_path + ".src", data_path + ".trg",
                                     source_token_vocab, target_token_vocab, EOS)


@registry.register_problem
class TranslateGenericExistingSharedVocab(translate.TranslateProblem):
  """Problem spec for generic translation, using existing vocab
  which is shared between source and target """

  @property
  def input_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def vocab_name(self):
    return "vocab.shared"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_name)
    encoder = text_encoder.TokenTextEncoder(vocab_filename, replace_oov="<unk>")
    return {"inputs": encoder, "targets": encoder}

  def generator(self, data_dir, tmp_dir, train):
    datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
    source_datasets = [[FLAGS.raw_data_dir, [item[0]]] for item in datasets]
    target_datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in datasets]
    # Copy vocab to data directory
    vocab_path = os.path.join(data_dir, self.vocab_name)
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[2]), vocab_path)
    token_vocab = text_encoder.TokenTextEncoder(vocab_path, replace_oov="<unk>")
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "generic_tok_%s" % tag)
    return translate.token_generator(data_path + ".src", data_path + ".trg",
                                     token_vocab, EOS)
