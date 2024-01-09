#!/bin/bash

conda activate geometry_hints;

CHECKPOINT='/mnt/nas/personal/mohameds/geometry_hints/weights/densification_fast_mesh_hint/version_0/checkpoints/last.ckpt'
CONFIG='configs/models/baselines/densification_fast_mesh_hint.yaml'

CUDA_VISIBLE_DEVICES=$1 python -m geometryhints.test \
--config_file $CONFIG \
--load_weights_from_checkpoint $CHECKPOINT \
--data_config configs/data/scannet_default_test.yaml \
--num_workers 12 \
--batch_size 8 \
--output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet \
--depth_hint_aug 0.0 \
--depth_hint_dir /mnt/nas3/personal/mohameds/geometry_hints/outputs/hero_model_fast/scannet/default/meshes/0.04_3.0_ours/renders/  \
--name densification_fast_mesh_hint;