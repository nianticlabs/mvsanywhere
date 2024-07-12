# DoubleTake: Geometry Guided Depth Estimation

This is the reference PyTorch implementation for training and testing MVS depth estimation models using the method described in

> **DoubleTake: Geometry Guided Depth Estimation**
>
> [Mohamed Sayed](https://masayed.com), [Filippo Aleotti](https://filippoaleotti.github.io/website/), [Jamie Watson](https://www.linkedin.com/in/jamie-watson-544825127/), [Zawar Qureshi](https://qureshizawar.github.io/), [Guillermo Garcia-Hernando](), [Gabriel Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/), [Sara Vicente](https://scholar.google.co.uk/citations?user=7wWsNNcAAAAJ&hl=en) and  [Michael Firman](http://www.michaelfirman.co.uk).
>
> [Paper, ECCV 2024 (arXiv pdf)](), [Supplemental Material](), [Project Page](https://nianticlabs.github.io/doubletake/), [Video](https://www.youtube.com/watch?v=IklQ5AHNdI8&feature=youtu.be)

<p align="center">
  <img src="media/teaser.jpeg" alt="example output" width="720" />
</p>


This code is for non-commercial use; please see the [license file](LICENSE) for terms. If you do find any part of this codebase helpful, please cite our paper using the BibTex below and link this repo. Thanks!

## Table of Contents

  * [🗺️ Overview](#%EF%B8%8F-overview)
  * [⚙️ Setup](#%EF%B8%8F-setup)
  * [📦 Pretrained Models](#-pretrained-models)
  * [🏃 Running out of the box!](#-running-out-of-the-box)
  * [💾 ScanNetv2 Dataset](#-scannetv2-dataset)
  * [📊 Testing and Evaluation](#-testing-and-evaluation)
  * [📊 Mesh Metrics](#-mesh-metrics)
  * [⏳ Training](#-training)
    + [🎛️ Finetuning a pretrained model](#%EF%B8%8F-finetuning-a-pretrained-model)
  * [🔧 Other training and testing options](#-other-training-and-testing-options)
  * [📝🧮👩‍💻 Notation for Transformation Matrices](#-notation-for-transformation-matrices)
  * [🗺️ World Coordinate System](#%EF%B8%8F-world-coordinate-system)
  * [🙏 Acknowledgements](#-acknowledgements)
  * [📜 BibTeX](#-bibtex)
  * [👩‍⚖️ License](#%EF%B8%8F-license)

## 🗺️ Overview

DoubleTake takes as input posed RGB images, and outputs a depth map for a target image.

## ⚙️ Setup

We are going to create a new Mamba environment called `doubletake`. If you don't have Mamba, you can install it with:

```shell
make install-mamba
```

```shell
make create-mamba-env
mamba activate doubletake
```

## 📦 Pretrained Models

Download a pretrained model into the `weights/` folder.


## 🏃 Running out of the box!

We've now included two scans for people to try out immediately with the code. You can download these scans [from here](https://drive.google.com/file/d/1x-auV7vGCMdu5yZUMPcoP83p77QOuasT/view?usp=sharing).

Steps:
1. Download weights for the `hero_model` into the weights directory.
2. Download the scans and unzip them to a directory of your choosing.
3. Modify the value for the option `dataset_path` in `configs/data/vdr_dense.yaml` to the base path of the unzipped vdr folder.
4. You should be able to run it! Something like this will work:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/vdr_dense.yaml \
            --num_workers 8 \
            --batch_size 2 \
            --fast_cost_volume \
            --run_fusion \
            --depth_fuser open3d \
            --fuse_color \
            --dump_depth_visualization;
```

This will output meshes, quick depth viz, and socres when benchmarked against LiDAR depth under `OUTPUT_PATH`. 

This command uses `vdr_dense.yaml` which will generate depths for every frame and fuse them into a mesh. In the paper we report scores with fused keyframes instead, and you can run those using `vdr_default.yaml`. You can also use `dense_offline` tuples by instead using `vdr_dense_offline.yaml`.



See the section below on testing and evaluation. Make sure to use the correct config flags for datasets. 

## 💾 ScanNetv2 Dataset

This section explains how to prepare ScanNetv2 for training and testing: in fact, ScanNetv2 only provides meshes with semantic labels, but not with planes.
Following previous works, we process the dataset extracting planar information with RANSAC.

🕒 Please note that the data preparation scripts will take a few hours to run.

<details>
<summary>ScanNetv2 download (training & testing)</summary>

  Please follow instructions reported in [SimpleRecon](https://github.com/nianticlabs/simplerecon/tree/main/data_scripts/scannet_wrangling_scripts)

You should get at the end a ScanNetv2 root folder that looks like:

```shell
SCANNET_ROOT
├── scans_test (test scans)
│   ├── scene0707
│   │   ├── scene0707_00_vh_clean_2.ply (gt mesh)
│   │   ├── sensor_data
│   │   │   ├── frame-000261.pose.txt
│   │   │   ├── frame-000261.color.jpg 
│   │   │   └── frame-000261.depth.png (full res depth, stored scale *1000)
│   │   ├── scene0707.txt (scan metadata and image sizes)
│   │   └── intrinsic
│   │       ├── intrinsic_depth.txt
│   │       └── intrinsic_color.txt
│   └── ...
└── scans (val and train scans)
    ├── scene0000_00
    │   └── (see above)
    ├── scene0000_01
    └── ....
```
</details>


## 📊 Testing and Evaluation

You can use `test.py` for inferring and evaluating depth maps and fusing meshes. 

All results will be stored at a base results folder (results_path) at:

    opts.output_base_path/opts.name/opts.dataset/opts.frame_tuple_type/

where opts is the `options` class. For example, when `opts.output_base_path` is `./results`, `opts.name` is `HERO_MODEL`,
`opts.dataset` is `scannet`, and `opts.frame_tuple_type` is `default`, the output directory will be 

    ./results/HERO_MODEL/scannet/default/

Make sure to set `--opts.output_base_path` to a directory suitable for you to store results.

`--frame_tuple_type` is the type of image tuple used for MVS. A selection should 
be provided in the `data_config` file you used. 

By default `test.py` will attempt to compute depth scores for each frame and provide both frame averaged and scene averaged metrics. The script will save these scores (per scene and totals) under `results_path/scores`.

We've done our best to ensure that a torch batching bug through the matching 
encoder is fixed for (<10^-4) accurate testing by disabling image batching 
through that encoder. Run `--batch_size 4` at most if in doubt, and if 
you're looking to get as stable as possible numbers and avoid PyTorch 
gremlins, use `--batch_size 1` for comparison evaluation.

If you want to use this for speed, set `--fast_cost_volume` to True. This will
enable batching through the matching encoder and will enable an einops 
optimized feature volume.


```bash
# Example command to just compute scores 
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_test.yaml \
            --num_workers 8 \
            --batch_size 4;

# If you'd like to get a super fast version use:
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_test.yaml \
            --num_workers 8 \
            --fast_cost_volume \
            --batch_size 2;
```

This script can also be used to perform a few different auxiliary tasks, 
including:

**TSDF Fusion**

To run TSDF fusion provide the `--run_fusion` flag. You have two choices for 
fusers
1) `--depth_fuser ours` (default) will use our fuser, whose meshes are used 
    in most visualizations and for scores. This fuser does not support 
    color. We've provided a custom branch of scikit-image with our custom
    implementation of `measure.matching_cubes` that allows single walled. We use 
    single walled meshes for evaluation. If this is isn't important to you, you
    can set the export_single_mesh to `False` for call to `export_mesh` in `test.py`.
2) `--depth_fuser open3d` will use the open3d depth fuser. This fuser 
    supports color and you can enable this by using the `--fuse_color` flag. 

By default, depth maps will be clipped to 3m for fusion and a tsdf 
resolution of 0.04m<sup>3</sup> will be used, but you can change that by changing both 
`--max_fusion_depth` and `--fusion_resolution`

You can optionnally ask for predicted depths used for fusion to be masked 
when no vaiid MVS information exists using `--mask_pred_depths`. This is not 
enabled by default.

You can also fuse the best guess depths from the cost volume before the 
cost volume encoder-decoder that introduces a strong image prior. You can do this by using 
`--fusion_use_raw_lowest_cost`.

Meshes will be stored under `results_path/meshes/`.

```bash
# Example command to fuse depths to get meshes
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_test.yaml \
            --num_workers 8 \
            --run_fusion \
            --batch_size 8;
```

**Cache depths**

You can optionally store depths by providing the `--cache_depths` flag. 
They will be stored at `results_path/depths`.

```bash
# Example command to compute scores and cache depths
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_test.yaml \
            --num_workers 8 \
            --cache_depths \
            --batch_size 8;

# Example command to fuse depths to get color meshes
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_test.yaml \
            --num_workers 8 \
            --run_fusion \
            --depth_fuser open3d \
            --fuse_color \
            --batch_size 4;
```
**Quick viz**

There are other scripts for deeper visualizations of output depths and 
fusion, but for quick export of depth map visualization you can use 
`--dump_depth_visualization`. Visualizations will be stored at `results_path/viz/quick_viz/`.


```bash
# Example command to output quick depth visualizations
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_test.yaml \
            --num_workers 8 \
            --dump_depth_visualization \
            --batch_size 4;
```

## 📊 Mesh Metrics

We use TransformerFusion's [mesh evaluation](https://github.com/AljazBozic/TransformerFusion/blob/main/src/evaluation/eval.py) for our main results table but set the seed to a fixed value for consistency when randomly sampling meshes. We also report mesh metrics using NeuralRecon's [evaluation](https://github.com/zju3dv/NeuralRecon/blob/master/tools/evaluation.py) in the supplemental material.

For point cloud evaluation, we use TransformerFusion's code but load in a point cloud in place of sampling a mesh's surface.



## ⏳ Training

By default models and tensorboard event files are saved to `~/tmp/tensorboard/<model_name>`.
This can be changed with the `--log_dir` flag.

We train with a batch_size of 16 with 16-bit precision on two A100s on the default ScanNetv2 split.

Example command to train with two GPUs:
```shell
CUDA_VISIBLE_DEVICES=0,1 python train.py --name HERO_MODEL \
            --log_dir logs \
            --config_file configs/models/hero_model.yaml \
            --data_config configs/data/scannet_default_train.yaml \
            --gpus 2 \
            --batch_size 16;
```


The code supports any number of GPUs for training.
You can specify which GPUs to use with the `CUDA_VISIBLE_DEVICES` environment.

All our training runs were performed on two NVIDIA A100s.

**Different dataset**

You can train on a custom MVS dataset by writing a new dataloader class which inherits from `GenericMVSDataset` at `datasets/generic_mvs_dataset.py`. See the `ScannetDataset` class in `datasets/scannet_dataset.py` or indeed any other class in `datasets` for an example.


### 🎛️ Finetuning a pretrained model

To finetune, simple load a checkpoint (not resume!) and train from there:
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/models/hero_model.yaml
                --data_config configs/data/scannet_default_train.yaml 
                --load_weights_from_checkpoint weights/hero_model.ckpt
```

Change the data configs to whatever dataset you want to finetune to. 

## 🔧 Other training and testing options

See `options.py` for the range of other training options, such as learning rates and ablation settings, and testing options.


## 📝🧮👩‍💻 Notation for Transformation Matrices

__TL;DR:__ `world_T_cam == world_from_cam`  
This repo uses the notation "cam_T_world" to denote a transformation from world to camera points (extrinsics). The intention is to make it so that the coordinate frame names would match on either side of the variable when used in multiplication from *right to left*:

    cam_points = cam_T_world @ world_points

`world_T_cam` denotes camera pose (from cam to world coords). `ref_T_src` denotes a transformation from a source to a reference view.  
Finally this notation allows for representing both rotations and translations such as: `world_R_cam` and `world_t_cam`

## 🗺️ World Coordinate System

This repo is geared towards ScanNet, so while its functionality should allow for any coordinate system (signaled via input flags), the model weights we provide assume a ScanNet coordinate system. This is important since we include ray information as part of metadata. Other datasets used with these weights should be transformed to the ScanNet system. The dataset classes we include will perform the appropriate transforms. 

## 🙏 Acknowledgements

We thank Aljaž Božič of [TransformerFusion](https://github.com/AljazBozic/TransformerFusion), Jiaming Sun of [Neural Recon](https://zju3dv.github.io/neuralrecon/), and Arda Düzçeker of [DeepVideoMVS](https://github.com/ardaduz/deep-video-mvs) for quickly providing useful information to help with baselines and for making their codebases readily available, especially on short notice.

The tuple generation scripts make heavy use of a modified version of DeepVideoMVS's [Keyframe buffer](https://github.com/ardaduz/deep-video-mvs/blob/master/dvmvs/keyframe_buffer.py) (thanks again Arda and co!).


We'd also like to thank Niantic's infrastructure team for quick actions when we needed them. Thanks folks!

## 📜 BibTeX

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{sayed2022simplerecon,
  title={DoubleTake: Geometry Guided Depth Estimation},
  author={Sayed, Mohamed and Aleotti, Filippo and Watson, Jamie and Qureshi, Zawar and Garcia-Hernando, Guillermo and Brostow, Gabriel and Vicente, Sara and Firman, Michael},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024},
}
```

## 👩‍⚖️ License

Copyright © Niantic, Inc. 2024. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
