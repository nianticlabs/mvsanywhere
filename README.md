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
  * [💾 3RScan Dataset](#-3rscan-dataset)
  * [📊 Testing and Evaluation](#-testing-and-evaluation)
  * [📊 Mesh Metrics](#-mesh-metrics)
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

We provide three models. The standard DoubleTake model used for incremental, offline, and revisit evaluation on all datasets and figures in the paper, a slimmed down faster version of DoubleTake, and the vanilla SimpleRecon model we used for SimpleRecon scores. Use the links in the table to access the weights for each.

| `--config`  | Model  | Abs Diff↓| Sq Rel↓ | delta < 1.05↑| Chamfer↓ | F-Score↑ |
|-------------|----------|--------------------|---------|---------|--------------|----------|
| Online/Incremental using `test_incremental.py` | | | | | | |
| [`configs/models/doubletake_model.yaml`](https://storage.googleapis.com/niantic-lon-static/research/doubletake/doubletake_model.ckpt) | Ours from paper | .0754 | .0109 | 80.29 | 5.03 | .689 |
| [`configs/models/doubletake_small_model.yaml`](https://storage.googleapis.com/niantic-lon-static/research/doubletake/doubletake_small_model.ckpt) | Ours fast from paper | .0825 | .0124 | 76.75 | 5.53 | .649 |
| [`configs/models/simplerecon_model.yaml`](https://storage.googleapis.com/niantic-lon-static/research/doubletake/simplerecon_model) | SimpleRecon | .0873 | .0128 | 74.12 | 5.29 | .668 |
| Offline/Two Pass using `test_two_pass.py` | | | | | | |
| configs/models/doubletake_model.yaml | Ours from paper | .0624 | .0092 | 86.64 | 4.42 | .742 |
| configs/models/doubletake_small_model.yaml | Ours fast from paper | .0631 | .0097 | 86.36 | 4.64 | .723 |
| configs/models/simplerecon_model.yaml | SimpleRecon | .0812 | .0118 | 77.02 | 5.05 | .687 |
| No hint and online using `test_no_hint` | | | | | | |
| configs/models/doubletake_model.yaml | Ours from paper | .0863 | .0127 | 74.64 | 5.22 | .672 |
| configs/models/doubletake_small_model.yaml | Ours fast from paper | .0938 | .0148 | 72.02 | 5.50 | .650 |
| configs/models/simplerecon_model.yaml | SimpleRecon | .0873 | .0128 | 74.12 | 5.29 | .668 |


## 🚀 Speed
Please see the paper and supplemental material for details on runtime.


## 🏃 Running out of the box!

We've now included two scans for people to try out immediately with the code. You can download these scans [from here](https://drive.google.com/file/d/1x-auV7vGCMdu5yZUMPcoP83p77QOuasT/view?usp=sharing).

Steps:
1. Download weights for the `hero_model` into the weights directory.
2. Download the scans and unzip them to a directory of your choosing.
3. Modify the value for the option `dataset_path` in `configs/data/vdr_dense.yaml` to the base path of the unzipped vdr folder.
4. You should be able to run it! Something like this will work:

```bash
CUDA_VISIBLE_DEVICES=0 python test_incremental.py --name doubletake \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/doubletake_model.yaml \
            --load_weights_from_checkpoint weights/doubletake_model.ckpt \
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
We've written a quick tutorial and included modified scripts to help you with downloading and extracting ScanNetv2. You can find them at [data_scripts/scannet_wrangling_scripts/](data_scripts/scannet_wrangling_scripts)

You should change the `dataset_path` config argument for ScanNetv2 data configs at `configs/data/` to match where your dataset is.

The codebase expects ScanNetv2 to be in the following format:

    dataset_path
        scans_test (test scans)
            scene0707
                scene0707_00_vh_clean_2.ply (gt mesh)
                sensor_data
                    frame-000261.pose.txt
                    frame-000261.color.jpg 
                    frame-000261.color.512.png (optional, image at 512x384)
                    frame-000261.color.640.png (optional, image at 640x480)
                    frame-000261.depth.png (full res depth, stored scale *1000)
                    frame-000261.depth.256.png (optional, depth at 256x192 also
                                                scaled)
                scene0707.txt (scan metadata and image sizes)
                intrinsic
                    intrinsic_depth.txt
                    intrinsic_color.txt
            ...
        scans (val and train scans)
            scene0000_00
                (see above)
            scene0000_01
            ....

In this example `scene0707.txt` should contain the scan's metadata:

        colorHeight = 968
        colorToDepthExtrinsics = 0.999263 -0.010031 0.037048 ........
        colorWidth = 1296
        depthHeight = 480
        depthWidth = 640
        fx_color = 1170.187988
        fx_depth = 570.924255
        fy_color = 1170.187988
        fy_depth = 570.924316
        mx_color = 647.750000
        mx_depth = 319.500000
        my_color = 483.750000
        my_depth = 239.500000
        numColorFrames = 784
        numDepthFrames = 784
        numIMUmeasurements = 1632

`frame-000261.pose.txt` should contain pose in the form:

        -0.384739 0.271466 -0.882203 4.98152
        0.921157 0.0521417 -0.385682 1.46821
        -0.0587002 -0.961035 -0.270124 1.51837

`frame-000261.color.512.png` and `frame-000261.color.640.png` are precached resized versions of the original image to save load and compute time during training and testing. `frame-000261.depth.256.png` is also a 
precached resized version of the depth map. 

All resized precached versions of depth and images are nice to have but not 
required. If they don't exist, the full resolution versions will be loaded, and downsampled on the fly.


## 🖼️🖼️🖼️ Frame Tuples

By default, we estimate a depth map for each keyframe in a scan. We use DeepVideoMVS's heuristic for keyframe separation and construct tuples to match. We use the depth maps at these keyframes for depth fusion. For each keyframe, we associate a list of source frames that will be used to build the cost volume. We also use dense tuples, where we predict a depth map for each frame in the data, and not just at specific keyframes; these are mostly used for visualization.

We generate and export a list of tuples across all scans that act as the dataset's elements. We've precomputed these lists and they are available at `data_splits` under each dataset's split. For ScanNet's test scans they are at `data_splits/ScanNetv2/standard_split`. Our core depth numbers are computed using `data_splits/ScanNetv2/standard_split/test_eight_view_deepvmvs.txt`.



Here's a quick taxonamy of the type of tuples for test:

- `default`: a tuple for every keyframe following DeepVideoMVS where all source frames are in the past. Used for all depth and mesh evaluation unless stated otherwise. For ScanNet use `data_splits/ScanNetv2/standard_split/test_eight_view_deepvmvs.txt`.
- `offline`: a tuple for every frame in the scan where source frames can be both in the past and future relative to the current frame. These are useful when a scene is captured offline, and you want the best accuracy possible. With online tuples, the cost volume will contain empty regions as the camera moves away and all source frames lag behind; however with offline tuples, the cost volume is full on both ends, leading to a better scale (and metric) estimate.
- `dense`: an online tuple (like default) for every frame in the scan where all source frames are in the past. For ScanNet this would be `data_splits/ScanNetv2/standard_split/test_eight_view_deepvmvs_dense.txt`.
- `offline`: an offline tuple for every keyframefor every keyframe in the scan.


For the train and validation sets, we follow the same tuple augmentation strategy as in DeepVideoMVS and use the same core generation script.

If you'd like to generate these tuples yourself, you can use the scripts at `data_scripts/generate_train_tuples.py` for train tuples and `data_scripts/generate_test_tuples.py` for test tuples. These follow the same config format as `test.py` and will use whatever dataset class you build to read pose informaiton.

Example for test:

```bash
# default tuples
python ./data_scripts/generate_test_tuples.py 
    --data_config configs/data/scannet/scannet_default_test.yaml
    --num_workers 16

# dense tuples
python ./data_scripts/generate_test_tuples.py 
    --data_config configs/data/scannet_dense_test.yaml
    --num_workers 16
```

Examples for train:

```bash
# train
python ./data_scripts/generate_train_tuples.py 
    --data_config configs/data/scannet/scannet_default_train.yaml
    --num_workers 16

# val
python ./data_scripts/generate_val_tuples.py 
    --data_config configs/data/scannet/scannet_default_val.yaml
    --num_workers 16
```

These scripts will first check each frame in the dataset to make sure it has an existing RGB frame, an existing depth frame (if appropriate for the dataset), and also an existing and valid pose file. It will save these `valid_frames` in a text file in each scan's folder, but if the directory is read only, it will ignore saving a `valid_frames` file and generate tuples anyway.


## 💾 3RScan Dataset

This section explains how to prepare 3RScan for testing:

Please download and extract the dataset by following the instructions [here](https://github.com/WaldJohannaU/3RScan).

The dataset should be formatted like so:

```
<dataset_path>
  <scanId>
  |-- mesh.refined.v2.obj
      Reconstructed mesh
  |-- mesh.refined.mtl
      Corresponding material file
  |-- mesh.refined_0.png
      Corresponding mesh texture
  |-- sequence.zip
      Calibrated RGB-D sensor stream with color and depth frames, camera poses
  |-- labels.instances.annotated.v2.ply
      Visualization of semantic segmentation
  |-- mesh.refined.0.010000.segs.v2.json
      Over-segmentation of annotation mesh
  |-- semseg.v2.json
            Instance segmentation of the mesh (contains the labels)
```

Please make sure to extract each `sequence.zip` inside every scanId folder.

We provide the frame tuple files for this dataset (see for eg. `data_splits/3rscan/test_eight_view_deepvmvs.txt`) but if you need recreate them, you can do so by following the instructions [here](https://github.com/nianticlabs/simplerecon/tree/main?tab=readme-ov-file#%EF%B8%8F%EF%B8%8F%EF%B8%8F-frame-tuples).

NOTE: we only use 3RScan dataset for testing and the data split used (`data_splits/3rscan/3rscan_test.txt`) corresponds to the validation split in the original dataset repo (`splits/val.txt`). We use the val split as the transformations that align the reference scan to the rescans are readily available for the train and val splits. 


## 📊 Testing and Evaluation

### Depth Evaluation

You can evaluate our model on the depth benchmark of ScanNetv2 using the following commands:

```shell
python -m scripts.evaluation incremental
python -m scripts.evaluation two_pass
```

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

Meshes will be stored under `results_path/meshes/`.

```bash
# Example command to fuse depths to get meshes
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet/scannet_default_test.yaml \
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
            --data_config configs/data/scannet/scannet_default_test.yaml \
            --num_workers 8 \
            --cache_depths \
            --batch_size 8;

# Example command to fuse depths to get color meshes
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet/scannet_default_test.yaml \
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
            --data_config configs/data/scannet/scannet_default_test.yaml \
            --num_workers 8 \
            --dump_depth_visualization \
            --batch_size 4;
```

### Revisit Evaluation

You can evaluate our model in the revist scenario (i.e using the geometry from a previous visit as ‘hints’ for our current depth estimates) on the 3RScan dataset by running the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m doubletake.test_revisit \
            --config_file configs/models/doubletake_model.yaml \
            --load_weights_from_checkpoint ./models/doubletake_model.ckpt \
            --data_config configs/data/3rscan/3rscan_test.yaml \
            --dataset_path PATH/TO/3RScan_dataset \
            --num_workers 12 \
            --batch_size 6 \
            --output_base_path ./outputs/ \
            --depth_hint_aug 0.0 \
            --load_empty_hint \
            --name final_model_3rscan_revisit \
            --run_fusion \
            --rotate_images \
            --cache_depths
```

## 📊 Mesh Metrics

We use a mesh evaluation protocol similar to TransformerFusion's, but use occlusion masks that better fit available geometry in the ground truth.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evals/mesh_eval.py \
    --groundtruth_dir SCANNET_TEST_FOLDER_PATH  \
    --prediction_dir ROOT_PRED_DIRECTORY/SCAN_NAME.ply \
```
 


## 📝🧮👩‍💻 Notation for Transformation Matrices

__TL;DR:__ `world_T_cam == world_from_cam`  
This repo uses the notation "cam_T_world" to denote a transformation from world to camera points (extrinsics). The intention is to make it so that the coordinate frame names would match on either side of the variable when used in multiplication from *right to left*:

    cam_points = cam_T_world @ world_points

`world_T_cam` denotes camera pose (from cam to world coords). `ref_T_src` denotes a transformation from a source to a reference view.  
Finally this notation allows for representing both rotations and translations such as: `world_R_cam` and `world_t_cam`

## 🗺️ World Coordinate System

This repo is geared towards ScanNet, so while its functionality should allow for any coordinate system (signaled via input flags), the model weights we provide assume a ScanNet coordinate system. This is important since we include ray information as part of metadata. Other datasets used with these weights should be transformed to the ScanNet system. The dataset classes we include will perform the appropriate transforms. 


## 🔨💾 Training Data Preperation
To train a DoubleTake model you'll need the ScanNetv2 dataset and renders of a mesh from an SR model. We provide these
renders.

To generate mesh renders, you'll first need to run a SimpleRecon model and cache those depths to disk. You should
use `scannet_default_train_inference_style.yaml` and `scannet_default_val_inference_style.yaml` for this. These conigs run the model on test-style keyframes 
on both train and val splits. Something like this:

```bash
CUDA_VISIBLE_DEVICES=0 python -m doubletake.test_no_hint 
    --config_file configs/models/simplerecon_model.yaml
    --load_weights_from_checkpoint simplerecon_model_weights.ckpt
    --data_config configs/data/scannet_default_train_inference_style.yaml  
    --num_workers 8
    --batch_size 8
    --cache_depths 
    --run_fusion 
    --output_base_path YOUR_OUTPUT_DIR
    --dataset_path SCANNET_DIR;

CUDA_VISIBLE_DEVICES=0 python -m doubletake.test_no_hint 
    --config_file configs/models/simplerecon_model.yaml
    --load_weights_from_checkpoint simplerecon_model_weights.ckpt
    --data_config configs/data/scannet_default_val_inference_style.yaml  
    --num_workers 8
    --batch_size 8
    --cache_depths 
    --run_fusion 
    --output_base_path YOUR_OUTPUT_DIR
    --dataset_path SCANNET_DIR;
```

With these cached depths, you can generate mesh renders for training:

```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/render_scripts/render_meshes.py \
    --data_config configs/data/scannet/scannet_default_train.yaml \
    --cached_depth_path YOUR_OUTPUT_DIR/simplerecon_model/scannet/default/depths \
    --output_root renders/partial_renders \
    --dataset_path SCANNET_DIR \
    --batch_size 4 \
    --data_to_render both \
    --partial 1;

CUDA_VISIBLE_DEVICES=0 python ./scripts/render_scripts/render_meshes.py \
    --data_config configs/data/scannet/scannet_default_train.yaml \
    --cached_depth_path YOUR_OUTPUT_DIR/simplerecon_model/scannet/default/depths \
    --output_root renders/renders \
    --dataset_path /mnt/scannet/ \
    --batch_size 4 \
    --data_to_render both \
    --partial 0;

CUDA_VISIBLE_DEVICES=0 python ./scripts/render_scripts/render_meshes.py \
    --data_config configs/data/scannet/scannet_default_val.yaml \
    --cached_depth_path YOUR_OUTPUT_DIR/simplerecon_model/scannet/default/depths \
    --output_root renders/partial_renders \
    --dataset_path SCANNET_DIR \
    --batch_size 4 \
    --data_to_render both \
    --partial 1;

CUDA_VISIBLE_DEVICES=0 python ./scripts/render_scripts/render_meshes.py \
    --data_config configs/data/scannet/scannet_default_val.yaml \
    --cached_depth_path YOUR_OUTPUT_DIR/simplerecon_model/scannet/default/depths \
    --output_root renders/renders \
    --dataset_path /mnt/scannet/ \
    --batch_size 4 \
    --data_to_render both \
    --partial 0;
```

## 🙏 Acknowledgements

The tuple generation scripts make heavy use of a modified version of DeepVideoMVS's [Keyframe buffer](https://github.com/ardaduz/deep-video-mvs/blob/master/dvmvs/keyframe_buffer.py) (thanks Arda and co!).

We'd like to thank the Niantic Raptor R\&D infrastructure team - Saki Shinoda, Jakub Powierza, and Stanimir Vichev - for their valuable infrastructure support.

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
