#!/bin/bash

conda activate geometryhints;

CHECKPOINT='/mnt/nas/personal/mohameds/geometry_hints/weights/cv_hint_depth_sampled_weights_better_contd_contd_contd/version_0/checkpoints/epoch=11-step=79071.ckpt'
CONFIG='/mnt/nas/personal/mohameds/geometry_hints/weights/cv_hint_depth_sampled_weights_better_contd_contd_contd/version_0/config.yaml'

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
--depth_hint_dir /mnt/nas3/personal/mohameds/geometry_hints/outputs/hero_model_fast/scannet/default/meshes/0.04_3.0_ours/renders/  \
--name cv_hint_depth_sampled_weights_no_hint;

CUDA_VISIBLE_DEVICES=$1 python -m geometryhints.test \
--config_file $CONFIG \
--load_weights_from_checkpoint $CHECKPOINT \
--data_config configs/data/scannet_default_test.yaml \
--num_workers 8 \
--batch_size 4 \
--output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet \
--depth_hint_aug 0.0 \
--depth_hint_dir /mnt/nas3/personal/mohameds/geometry_hints/outputs/hero_model_fast/scannet/default/meshes/0.04_3.0_ours/renders/  \
--name cv_hint_depth_sampled_weights_with_hint;

CUDA_VISIBLE_DEVICES=$1 python -m geometryhints.test \
--config_file $CONFIG \
--load_weights_from_checkpoint $CHECKPOINT \
--data_config configs/data/scannet_default_test.yaml \
--num_workers 8 \
--batch_size 4 \
--output_base_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/ \
--dataset_path /mnt/scannet \
--depth_hint_aug 0.0 \
--depth_hint_dir /mnt/nas3/personal/mohameds/geometry_hints/outputs/hero_model_fast/scannet/default/meshes/0.04_3.0_ours/renders/  \
--name cv_hint_depth_sampled_weights_no_sweep_with_hints \
--null_plane_sweep;