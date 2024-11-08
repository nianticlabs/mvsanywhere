## Create conda environment

```
make install-mamba

make create-mamba-env
mamba activate fmvs
python -m pip install -e .
```

## Branches

### main
SimpleRecon branch with small changes that are common among other branches
 - Code to work with ray
 - Arbitrary matching scale and prediction scale as an option

### sr++ (and others sr++_double_img_encoder, sr++_fullvit)
Only the architecture is changed to use ViT and Depth Anything volumes. Code ready to train, validate and evaluate on scannet

### softmax_multids_double_img_encoder and regression_multids_double_img_encoder
The double img encoder implementation ready to train on multiple datasets and to evaluate on scannet.

### more_datasets
Branch where I keep implementing dataloaders, tuple generators, .... for the different datasets


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

## Train:

To train and validate on scannet (from main, or sr++) run:
```
# For the original SR arch
python3 ./src/doubletake/train.py --name HERO_MODEL --log_dir logs --config_file configs/models/sr_model.yaml --data_config configs/data/scannet/scannet_default_train.yaml --gpus 1 --batch_size 16
# For the double img encoder
python3 ./src/doubletake/train.py --name double_img_encoder --log_dir logs --config_file configs/models/sr_double_img_encoder.yaml --data_config configs/data/scannet/scannet_default_train.yaml --gpus 1 --batch_size 16
```

To train on multiple datasets:

```
python3 ./src/doubletake/train.py --name double_img_encoder --log_dir logs --config_file configs/models/sr_double_img_encoder.yaml --data_config configs/data/dynamic_replica/dynamic_replica_default_train.yaml:configs/data/matrix_city/matrix_city_default_train.yaml:configs/data/hypersim/hypersim_default_train.yaml:configs/data/blendedmvg/blendedmvg_default_val.yaml:configs/data/tartanair/tartanair_default_train.yaml:configs/data/vkitti/vkitti_default_train.yaml --val_data_config configs/data/scannet/scannet_default_val.yaml --gpus 1 --batch_size 16
```

To start a training in ray:

```
python src/doubletake/ray/train.py --name regression_multids_double_img_encoder_poses_range --config_file configs/models/sr_double_img_encoder.yaml --data_config configs/data/dynamic_replica/dynamic_replica_default_train.yaml:configs/data/matrix_city/matrix_city_default_train.yaml:configs/data/hypersim/hypersim_default_train.yaml:configs/data/blendedmvg/blendedmvg_default_val.yaml:configs/data/tartanair/tartanair_default_train.yaml:configs/data/vkitti/vkitti_default_train.yaml --val_data_config configs/data/scannet/scannet_default_val.yaml --batch_size 8 --val_batch_size 8
```

To test on scannet:
python3 ./src/doubletake/test.py --name my_model_name --config_file configs/models/sr_double_img_encoder.yaml --data_config configs/data/scannet/scannet_default_test.yaml --load_weights_from_checkpoint  /path/to/ckpt/ --fast_cost_volume --gpus 1 --num_workers 8 --batch_size 4

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


## Evaluation on the robust benchmark

See [src/rmvd/models/README.md](src/rmvd/models/README.md).