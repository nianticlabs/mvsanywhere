#!/bin/bash

conda activate geometryhints;

CHECKPOINT='/mnt/nas/personal/mohameds/geometry_hints/weights/hero_model_sampled_weights_single_MLP_partial/version_2/checkpoints/epoch=7-step=88141.ckpt'
CONFIG='/mnt/nas/personal/mohameds/geometry_hints/weights/hero_model_sampled_weights_single_MLP_partial/version_2/config.yaml'
RENDER_DIR='~/geometryhints/data/renders'

echo $CHECKPOINT
echo $CONFIG

CUDA_VISIBLE_DEVICES=$1 python -m geometryhints.test \
--config_file $CONFIG \
--load_weights_from_checkpoint $CHECKPOINT \
--data_config configs/data/scannet_default_test.yaml \
--num_workers 8 \
--batch_size 4 \
--output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet \
--depth_hint_aug 1.0 \
--depth_hint_dir $RENDER_DIR  \
--name hero_model_sampled_weights_single_MLP_partial_no_hint;

CUDA_VISIBLE_DEVICES=$1 python -m geometryhints.test \
--config_file $CONFIG \
--load_weights_from_checkpoint $CHECKPOINT \
--data_config configs/data/scannet_default_test.yaml \
--num_workers 8 \
--batch_size 4 \
--output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet \
--depth_hint_aug 0.0 \
--depth_hint_dir $RENDER_DIR  \
--name hero_model_sampled_weights_single_MLP_partial_with_hint;

CUDA_VISIBLE_DEVICES=$1 python -m geometryhints.test \
--config_file $CONFIG \
--load_weights_from_checkpoint $CHECKPOINT \
--data_config configs/data/scannet_default_test.yaml \
--num_workers 8 \
--batch_size 4 \
--output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet \
--depth_hint_aug 0.0 \
--depth_hint_dir $RENDER_DIR  \
--name hero_model_sampled_weights_single_MLP_partial_with_hint_null \
--null_plane_sweep;

CUDA_VISIBLE_DEVICES=$1 python -m geometryhints.test_incremental_render  \
--config_file $CONFIG \
--load_weights_from_checkpoint $CHECKPOINT \
--data_config configs/data/scannet_default_test.yaml  \
--num_workers 12  \
--batch_size 1  \
--output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/  \
--dataset_path /mnt/scannet  \
--depth_hint_aug 0.0  \
--name hero_model_sampled_weights_single_MLP_partial_incremental \
--run_fusion \
--plane_sweep_ablation_ratio 0.0  \
--load_empty_hint;