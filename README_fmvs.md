The `main` branch contains our HERO model, with high resolution, view count agnostic, normalized (except R) MLP.

## Create conda environment

```
make install-mamba

make create-mamba-env
mamba activate fmvs
python -m pip install -e .
```

## Models

HERO 480x640 MLP (No R Norm):
`/mnt/nas3/shared/projects/fmvs/fmvs/logs/ray/ablation_mlp_allsynth_hr_no_rnorm/TorchTrainer_90e3d_00000_0_2024-11-16_01-38-42/checkpoint_000014/checkpoint.ckpt`
HERO 480x640 w/o metadata:
`/mnt/nas3/shared/projects/fmvs/fmvs/logs/ray/hero_2_dot_allsynth_hr/TorchTrainer_67530_00000_0_2024-11-11_10-26-02/checkpoint_000014/checkpoint.ckpt`
Other models (follow experiment name for clue on what they are, or check code inside the experiment):
`/mnt/nas3/shared/projects/fmvs/fmvs/logs/ray/*`


## Train:

To train a model:
```
python3 ./src/doubletake/train.py \
  --log_dir logs \
  --name HERO_MODEL \
  --config_file configs/models/sr_double_vit.yaml \
  --data_config configs/data/blendedmvg/blendedmvg_default_train.yaml:configs/data/vkitti/vkitti_default_train.yaml:configs/data/dynamic_replica/dynamic_replica_default_train.yaml:configs/data/matrix_city/matrix_city_default_train.yaml:configs/data/hypersim/hypersim_default_train.yaml:configs/data/tartanair/tartanair_default_train.yaml:configs/data/sailvos3d/sailvos3d_default_train.yaml:configs/data/mvssynth/mvssynth_default_train.yaml  \
  --val_data_config configs/data/scannet/scannet_default_val.yaml \
  --gpus 2 \
  --batch_size 6 \
  --val_batch_size 6 \
  --num_workers 6 \
  --da_weights_path /mnt/nas3/shared/projects/fmvs/fmvs/weights/depth_anything_v2_vitb.pth
```

More datasets can be added by appending more `:` concatenaded. To also use real datasets, append:
```
configs/data/waymo/waymo_default_train.yaml:configs/data/arkitscenes/arkitscenes_default_train_landscape.yaml
```

To start a training in ray:
```
python3 ./src/doubletake/ray/train.py \
  --name HERO_MODEL \
  --config_file configs/models/sr_double_vit.yaml \
  --data_config configs/data/blendedmvg/blendedmvg_default_train.yaml:configs/data/vkitti/vkitti_default_train.yaml:configs/data/dynamic_replica/dynamic_replica_default_train.yaml:configs/data/matrix_city/matrix_city_default_train.yaml:configs/data/hypersim/hypersim_default_train.yaml:configs/data/tartanair/tartanair_default_train.yaml:configs/data/sailvos3d/sailvos3d_default_train.yaml:configs/data/mvssynth/mvssynth_default_train.yaml  \
  --val_data_config configs/data/scannet/scannet_default_val.yaml \
  --gpus 2 \
  --batch_size 6 \
  --val_batch_size 6 \
  --da_weights_path /mnt/nas3/shared/projects/fmvs/fmvs/weights/depth_anything_v2_vitb.pth
```


## Evaluation on the Robust Benchmark

To evaluate a model on all 5 datasets of the robust benchmark:

```
python ./eval.py \
  --model fmvs_wrapped \
  --eval_type robustmvd \
  --inputs poses intrinsics \
  --weights /path/to/code:/path/to/weights \
  --output path/to/results \
  --kitti_size 480 1280 \ 
  --dtu_size 480 640 \
  --eth3d_size 480 640 \
  --scannet_size 480 640 \
  --tanks_and_temples_size 480 640 \
  --max_source_views 7 \
  --use_refinement
```

For example to evaluate on our hero model, use:
```
--weights /mnt/nas3/shared/projects/fmvs/fmvs/logs/ray/ablation_mlp_allsynth_hr_no_rnorm/code/_ray_pkg_aeeedeac580c8789/:/mnt/nas3/shared/projects/fmvs/fmvs/logs/ray/ablation_mlp_allsynth_hr_no_rnorm/TorchTrainer_90e3d_00000_0_2024-11-16_01-38-42/checkpoint_000014/checkpoint.ckpt
```


## Improved Benchmark

Version of the Robust Benchmark with our improvements:
- Scannet with better tuples, using the selection from DMVS
- ETH3D on undistorted images.

It lies on `sergioizquierdo/improved_benchmark`

To run a model in these two improved datasets:
```
git switch sergioizquierdo/improved_benchmark
python ./eval.py \
  --model fmvs_wrapped \
  --eval_type robustmvd \
  --inputs poses intrinsics \
  --weights /path/to/code:/path/to/weights \
  --output path/to/results \
  --eth3d_size 480 640 \
  --scannet_size 480 640 \
  --max_source_views 7 \
  --use_refinement
```

To generate again the improved datasets pickel files:

```
git switch sergioizquierdo/improved_benchmark

# Scannet
# Compute the scannet tuples for the rmvd reference frames using DMVS selection
python3 scripts/data_scripts/robustmvd_scripts/generate_rmvd_scannet_better_tuples.py \
  --data_config configs/data/scannet/scannet_rmvd_better_tuples_test.yaml \
  --num_workers 8 \
  --num_images_in_tuple 8 \
# Create the pickle file
python3 scripts/data_scripts/robustmvd_scripts/create_scannet_better_tuples.py

# ETH3D
# Undistort the images and save them to disk. At the same time create the corresponding pickle file
python3 scripts/data_scripts/robustmvd_scripts/create_undistorted_eth3d.py
```


### Waymo preprocessing

First run
```
python dust3r_waymo_preprocess.py 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
```
The whole dataset is huge; the numbers after the command let you preprocess 1/16th at a time, e.g. if you want to split across different machines.
Inside that script there is a `do_extract_frames` parameter which you can set to False if you just want to do the final step of preprocessing. You can also up the number of workers but personally I found it didn't really speed things up.

```
> /mnt/nas3/personal/mfirman/fmvs/data_splits/waymo/train_scans.txt
for file in $(ls /mnt/nas3/shared/datasets/waymo/preprocessed/training/ | grep -v tmp); do
  for i in {1..5}; do
    echo "${file}^${i}" >> /mnt/nas3/personal/mfirman/fmvs/data_splits/waymo/train_scans.txt
  done
done

python -m scripts.data_scripts.generate_train_tuples_geometry \
    --data_config configs/data/waymo/waymo_default_train.yaml \
    --num_workers 16 \
    --num_images_in_tuple 8
```

### Sections of the code that are relevant?

To choose the input depth range of the network `src/doubletake/datasets/generic_mvs_dataset.py`, lines 630

How is the depth prediction calculated `src/doubletake/experiment_modules/sr_depth_model.py` at the end of the forward function

Where is the ViT double img encoder code? `src/doubletake/modules/vit_modules.py` and `src/doubletake/modules/depth_anything_blocks.py`



## Where is the stuff?

Datasets:
```
/mnt/nas3/shared/datasets/
```

Training logs
```
/mnt/nas3/shared/projects/fmvs/fmvs/logs/ray
```

## Branches

### main
Our hero branch, with all the datasets, MLP implemented.

Best configuration: `sr_double_vit.yaml`

### sergioizquierdo/improved_benchmark
Code to create the improved datasates and to evaluate a model on them

### sergioizquierdo/create_sequential_video
Two small python scripts to create T&T and KITTI sequential datasets for the robust benchmark, and a script to create a video out of the results.
Only the architecture is changed to use ViT and Depth Anything volumes. Code ready to train, validate and evaluate on scannet

### sergioizquierdo/contribution_depth_range_estimator

Branch with code to predict the depth range directly from the patchified cost volume. It doesn't work very well, but may be worth trying again.

## Evaluation on the robust benchmark

See [src/rmvd/models/README.md](src/rmvd/models/README.md).