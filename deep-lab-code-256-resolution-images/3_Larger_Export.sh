#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move two-level up to tensorflow/models/research directory.
echo $PYTHONPATH

# Set up the working environment.
WORK_DIR=$(pwd)
# Go to datasets folder and download POTSDAM segmentation dataset.
DATASET_DIR="datasets"

# Set up the working directories.
POTSDAM_FOLDER="potsdam"
EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/${EXP_FOLDER}/export"
POTSDAM_DATASET="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/tfrecord"

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-50000"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="mobilenet_v2" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=6 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
