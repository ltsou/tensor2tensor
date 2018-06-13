#!/bin/bash

# Usage: run-t2t.sh <workdir> <raw_data_dir> <train_steps> <problem> <model> <hparams>

# NOTE:
# 1. See what problems, models, and hyperparameter sets are available
#    by using the following command. You can easily swap between them (and add new ones).
#       python /home/centos/tools/tensor2tensor/tensor2tensor/bin/t2t-trainer --registry_help
#
# 2. enjp 32k takes about 38 hours to train (250k steps) on a p2.xlarge instance
#
# 3. Location of the training/dev data is specified by variables
#    _TRAIN_DATASETS and _TEST_DATASETS in
#       /home/centos/tools/tensor2tensor/tensor2tensor/data_generators/translate_generic.py
#
# 4. To run any of the translate_generic* PROBLEM, please upload data to the
#    assumed location found in point 3. above.

WORKDIR=${1:-/home/centos/workspace/t2t_workspace}
RAW_DATA_DIR=${2:-/home/centos/data}
TRAIN_STEPS=${3:-250000} # Default 250000
PROBLEM=${4:-translate_generic} # or translate_generic_existing_vocab
MODEL=${5:-transformer}
HPARAMS_SET=${6:-transformer_base_single_gpu}

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  T2T_HOME="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$T2T_HOME/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
T2T_HOME="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

T2T_BIN=$T2T_HOME/tensor2tensor/bin
DATA_DIR=$WORKDIR/data
TMP_DIR=$WORKDIR/tmp_data
TRAIN_DIR=$WORKDIR/train
OUTPUT_DIR=$WORKDIR/output
LOG_DIR=$WORKDIR/log
LOG=$LOG_DIR/log.run-t2t

# Train options
VOCAB_SIZE=${VOCAB_SIZE:-50000}
NUM_CKPT=${NUM_CKPT:-20}
SAVE_NPZ=${SAVE_NPZ:-0}
VAR_PREFIX=${VAR_PREFIX:-transformer}
TRAINER_FLAGS=${TRAINER_FLAGS:-""}
DECODER_FLAGS=$TRAINER_FLAGS
PREV_MODEL=${PREV_MODEL:-""}

# Decode options
DECODE_FILE=${DECODE_FILE:-$RAW_DATA_DIR/test/test.src} # decode will only be performed if this file exists
BEAM_SIZE=${BEAM_SIZE:-4}
ALPHA=${ALPHA:-0.6}

mkdir -p $LOG_DIR

function displaytime {
  local T=$1
  local D=$((T/60/60/24))
  local H=$((T/60/60%24))
  local M=$((T/60%60))
  local S=$((T%60))
  (( $D > 0 )) && printf '%d days ' $D
  (( $H > 0 )) && printf '%d hours ' $H
  (( $M > 0 )) && printf '%d minutes ' $M
  (( $D > 0 || $H > 0 || $M > 0 )) && printf 'and '
  printf '%d seconds\n' $S
}

function logMessage {
    echo "[`date`]: $1" >> $LOG
}


# get number of gpu
if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
    ngpu=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l`
    logMessage "Using all available GPUs: $ngpu"
else
    ngpu=`echo $CUDA_VISIBLE_DEVICES | tr , '\n' | wc -l`
    logMessage "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [[ $HPARAMS_SET = transformer_base* || $HPARAMS_SET = transformer_big* ]]; then
    if [[ $ngpu -eq 1 && $HPARAMS_SET != *_single_gpu ]]; then
        HPARAMS_SET=${HPARAMS_SET}_single_gpu
    elif [[ $ngpu -gt 1 ]]; then
        if [[ $HPARAMS_SET == *_single_gpu ]]; then
            HPARAMS_SET=${HPARAMS_SET%"_single_gpu"}
        fi
    fi
fi

logMessage "=== Start $0 ==="
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $OUTPUT_DIR

#### Since t2t-trainer now contains a --generate_data flag, 
#### we don't need to separately run t2t-datagen anymore
### Generate data
#if ls $DATA_DIR/$PROBLEM-train* 1> /dev/null 2>&1; then
#    logMessage "SKIPPING datagen, prepared data exists: $DATA_DIR/$PROBLEM-train*"
#else
#    SECONDS=0
#    logMessage "START preparing data..."
#    python $T2T_BIN/t2t-datagen \
#        --targeted_vocab_size=$VOCAB_SIZE \
#        --raw_data_dir=$RAW_DATA_DIR \
#        --data_dir=$DATA_DIR \
#        --tmp_dir=$TMP_DIR \
#        --problem=$PROBLEM >> $LOG_DIR/log.t2t-datagen.out 2>> $LOG_DIR/log.t2t-datagen.err
#    logMessage "END data preparation, elapsed time: $SECONDS seconds (`displaytime $SECONDS`)"
#fi

### Train
if ls $TRAIN_DIR/model.ckpt-$TRAIN_STEPS.* 1> /dev/null 2>&1; then
    logMessage "SKIPPING training. Checkpoint for step $TRAIN_STEPS already exists."
else
    # Need to specify a targeted vocab size if doing "translate_generic"
    if [ $PROBLEM = translate_generic ]; then
        TRAINER_FLAGS="$TRAINER_FLAGS --targeted_vocab_size=$VOCAB_SIZE"
    fi

    # Is there a need to generate data?
    if [ ! -f $DATA_DIR/generate_data.DONE ]; then
        TRAINER_FLAGS="$TRAINER_FLAGS --generate_data"
    fi

    # Copy over previous model if exists
    if [ -d "$PREV_MODEL" ]; then
        cp -r $PREV_MODEL/* $TRAIN_DIR
    fi

    SECONDS=0
    logMessage "START training... to step: $TRAIN_STEPS"
    cmd="python $T2T_BIN/t2t-trainer
      --raw_data_dir=$RAW_DATA_DIR
      --tmp_dir=$TMP_DIR
      --train_steps=$TRAIN_STEPS
      --data_dir=$DATA_DIR
      --problems=$PROBLEM
      --model=$MODEL
      --hparams_set=$HPARAMS_SET
      --worker_gpu=$ngpu
      --output_dir=$TRAIN_DIR
      --keep_checkpoint_max=$NUM_CKPT $TRAINER_FLAGS"
    logMessage "$cmd"
    $cmd >> $LOG_DIR/log.t2t-trainer.out 2>> $LOG_DIR/log.t2t-trainer.err
    touch $DATA_DIR/generate_data.DONE # Mark data generation as done
    logMessage "END training, elapsed time: $SECONDS seconds (`displaytime $SECONDS`)"
fi

if ls $OUTPUT_DIR/averaged.ckpt-* 1> /dev/null 2>&1; then
    logMessage "SKIPPING averaging. Averaged checkpoint already exists."
else
    additional_flags=''

    SECONDS=0
    logMessage "START Averaging latest checkpoints..."
    cmd="python $T2T_HOME/tensor2tensor/utils/avg_checkpoints.py
      --prefix $TRAIN_DIR/
      --num_last_checkpoints $NUM_CKPT
      --output_path $OUTPUT_DIR/averaged.ckpt
      $additional_flags"
    logMessage "$cmd"
    $cmd >> $LOG_DIR/log.avg_checkpoints.out 2>> $LOG_DIR/log.avg_checkpoints.err 

    # For now, save both ckpt and npz if SAVE_NPZ is on
    if [ $SAVE_NPZ -eq 1 ]; then
        additional_flags="$additional_flags --save_npz"
        if [ ! -z "$VAR_PREFIX" ]; then
            additional_flags="$additional_flags --var_prefix $VAR_PREFIX"
        fi
        cmd="python $T2T_HOME/tensor2tensor/utils/avg_checkpoints.py
          --prefix $TRAIN_DIR/
          --num_last_checkpoints $NUM_CKPT
          --output_path $OUTPUT_DIR/averaged.ckpt
          $additional_flags"
	logMessage "$cmd"
        $cmd >> $LOG_DIR/log.avg_checkpoints.out 2>> $LOG_DIR/log.avg_checkpoints.err 
    fi

    logMessage "END averaging, elapsed time: $SECONDS seconds (`displaytime $SECONDS`)"
fi

logMessage "TRAINING HAS BEEN FINISHED"

# Decode test file
if [[ -f $DECODE_FILE ]]; then
    DECODE_FILE_OUT_PATH=$OUTPUT_DIR/$(basename "$DECODE_FILE").out
    run_decode=1
    # if output file already exists, check if it has the correct number of lines
    if [ -f $DECODE_FILE_OUT_PATH ]; then
        in_ln=`wc -l < $DECODE_FILE`
        out_ln=`wc -l < $DECODE_FILE_OUT_PATH`
        if [ $in_ln -eq $out_ln ]; then
            run_decode=0
        else
            logMessage "Re-decode since output file contains incorrect line count ($out_ln != $in_ln): $DECODE_FILE_OUT_PATH"
        fi
    fi
    if [ $run_decode -eq 1 ]; then
        # Need to specify a targeted vocab size if doing "translate_generic"
        if [ $PROBLEM = translate_generic ]; then
            DECODER_FLAGS="$DECODER_FLAGS --targeted_vocab_size=$VOCAB_SIZE"
        fi
        SECONDS=0
        logMessage "Start decoding... $DECODE_FILE"
        cmd="python $T2T_BIN/t2t-decoder
          --data_dir=$DATA_DIR
          --problems=$PROBLEM
          --model=$MODEL
          --hparams_set=$HPARAMS_SET
          --output_dir=$OUTPUT_DIR
          --decode_beam_size=$BEAM_SIZE
          --decode_alpha=$ALPHA
          --decode_from_file=$DECODE_FILE
          --decode_to_file=$DECODE_FILE_OUT_PATH $DECODER_FLAGS"
	logMessage "$cmd"
        $cmd >> $LOG_DIR/log.t2t-decoder.out 2>> $LOG_DIR/log.t2t-decoder.err
        logMessage "END decoding, elapsed time: $SECONDS seconds (`displaytime $SECOND`)"
        logMessage "Decoded output: $DECODE_FILE_OUT_PATH"
    else
        logMessage "SKIPPING decoding, output file already exists: $DECODE_FILE_OUT_PATH."
    fi
fi

logMessage "Done"
