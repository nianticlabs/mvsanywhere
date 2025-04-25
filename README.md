# MVSAnywhere: Zero Shot Multi-View Stereo

A multi-view stereo depth estimation model which works anywhere, in any scene, with any range of depths

> **MVSAnywhere: Zero Shot Multi-View Stereo**
>
> [Sergio Izquierdo](https://serizba.github.io/), [Mohamed Sayed](https://masayed.com), [Michael Firman](http://www.michaelfirman.co.uk), [Guillermo Garcia-Hernando](https://guiggh.github.io/), [Daniyar Turmukhambetov](https://dantkz.github.io/about/), [Javier Civera](http://webdiis.unizar.es/~jcivera/), [Oisin Mac Aodha](https://homepages.inf.ed.ac.uk/omacaod/), [Gabriel Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/) and [Jamie Watson](https://www.linkedin.com/in/jamie-watson-544825127/).
> [Paper, CVPR 2025 (arXiv pdf)](https://arxiv.org/abs/2503.22430), [Project Page](https://nianticlabs.github.io/mvsanywhere/)

https://github.com/user-attachments/assets/d35b93f7-5f0e-4fbd-b991-bc4e7a45f2b6

This code is for non-commercial use; please see the [license file](LICENSE) for terms. If you do find any part of this codebase helpful, please cite our paper using the BibTex below and link this repo. Thanks!

## Table of Contents

- [MVSAnywhere: Zero Shot Multi-View Stereo](#mvsanywhere-zero-shot-multi-view-stereo)
  - [Table of Contents](#table-of-contents)
  - [⚙️ Setup](#️-setup)
  - [📦 Pretrained Models](#-pretrained-models)
  - [🏃 Running out of the box!](#-running-out-of-the-box)
  - [Running on recordings from your own device!](#running-on-recordings-from-your-own-device)
    - [🍏 iOS](#-ios)
    - [📱 Android](#-android)
    - [📷 Custom data](#-custom-data)
  - [Running Gaussian splatting with MVSAnywhere regularisation!](#running-gaussian-splatting-with-mvsanywhere-regularisation)
  - [📊 Testing and Evaluation](#-testing-and-evaluation)
    - [Robust Multi-View Depth Benchmark (RMVD)](#robust-multi-view-depth-benchmark-rmvd)
  - [🔨 Training](#-training)
  - [📝🧮👩‍💻 Notation for Transformation Matrices](#-notation-for-transformation-matrices)
  - [🗺️ World Coordinate System](#️-world-coordinate-system)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [📜 BibTeX](#-bibtex)
  - [👩‍⚖️ License](#️-license)

## ⚙️ Setup

We are going to create a new Mamba environment called `mvsanywhere`. If you don't have Mamba, you can install it with:

```shell
make install-mamba
```

```shell
make create-mamba-env
mamba activate mvsanywhere
```

In the code directory, install the repo as a pip package:

```shell
pip install -e .
```
To use our Gaussian splatting regularization also install that module:

```shell
pip install -e src/regsplatfacto/
```

## 📦 Pretrained Models

We provide 2 variants of our models: [mvsanywhere_hero.ckpt](https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_hero.ckpt) and [mvsanywhere_dot.ckpt](https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_dot.ckpt). `mvsanywhere_hero` is "Ours" from the main paper, and `mvsanywhere_dot` is ours with no metadata MLP. 

## 🏃 Running out of the box!

We've now included two scans for people to try out immediately with the code. You can download these scans [from here](https://drive.google.com/file/d/1x-auV7vGCMdu5yZUMPcoP83p77QOuasT/view?usp=sharing).

Steps:
1. Download weights for the `hero_model` into the weights directory.
2. Download the scans and unzip them to a directory of your choosing.
3. You should be able to run it! Something like this will work:

```shell
CUDA_VISIBLE_DEVICES=0 python src/mvsanywhere/run_demo.py \
    --name mvsanywhere \
    --output_base_path OUTPUT_PATH \
    --config_file configs/models/mvsanywhere_model.yaml \
    --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
    --data_config_file configs/data/vdr/vdr_dense.yaml \
    --scan_parent_directory /path/to/vdr/ \
    --scan_name house \ # Scan name (house or living_room)
    --num_workers 8 \
    --batch_size 2 \
    --fast_cost_volume \
    --run_fusion \
    --depth_fuser custom_open3d \
    --fuse_color \
    --fusion_max_depth 3.5 \
    --fusion_resolution 0.02 \
    --extended_neg_truncation \
    --dump_depth_visualization
```

This will output meshes, quick depth viz, and socres when benchmarked against LiDAR depth under `OUTPUT_PATH`. 

If you run out of GPU memory, you can try removing the `--fast_cost_colume` flag.

## Running on recordings from your own device!

### 🍏 iOS
<details>
<summary>How to use NeRF Capture to record videos</summary>

1. Download the [NeRF Capture](https://github.com/jc211/NeRFCapture) app from the [App Store](https://apps.apple.com/us/app/nerfcapture/id6446518379). Capture a recording of your favourite environment and save it. 

2. Place your recordings in a directory with the following structure:

```
/path/to/recordings/
│-- recording_0/
│   │-- images/
|   |   |-- image_0.png
|   |   |-- image_1.png
|   |   ...
│   │-- transforms.json
│-- recording_1/
|   ...
|-- scans.txt # See point 3.
```

3. And run the model 🚀🚀🚀
```shell
python src/mvsanywhere/run_demo.py \
    --name mvsanywhere \
    --output_base_path OUTPUT_PATH \
    --config_file configs/models/mvsanywhere_model.yaml \
    --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
    --data_config_file configs/data/nerfstudio/nerfstudio_empty.yaml \
    --scan_parent_directory /path/to/recordings/ \
    --scan_name recording_0 \
    --fast_cost_volume \
    --num_workers 8 \
    --batch_size 2 \
    --image_height 480 \
    --image_width 640 \
    --dump_depth_visualization \
    --rotate_images # Only if you recorded in portrait
```
</details>

### 📱 Android
<details>
<summary>Coming Soon</summary>

</details>

### 📷 Custom data
<details>
<summary>Use COLMAP to obtain a sparse reconstruction</summary>

If you already have a COLMAP reconstruction skip to 4.

1. Install [nerfstudio](https://docs.nerf.studio/quickstart/installation.html)
2. Install COLMAP using `conda install -c conda-forge colmap`. 
3. Process your video/sequence using
```shell
ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
```
4. Your reconstructions should have the following structure:
```
/path/to/reconstruction/
│-- reconstruction_0/
│   │-- images/
|   |   |-- image_0.png
|   |   |-- image_1.png
|   |   ...
│   │-- colmap/
|   |   |-- database.db
|   |   |-- sparse/
|   |   |   |-- 0/
|   |   |   |   |-- cameras.bin
|   |   |   |   |-- images.bin
|   |   |   |   ...
|   |   |   |-- 1/
|   |   |   |   ...
│-- reconstruction_1/
|   ...
```
5. And run the model 🚀🚀🚀
```shell
python src/mvsanywhere/run_demo.py \
    --name mvsanywhere \
    --output_base_path OUTPUT_PATH \
    --config_file configs/models/mvsanywhere_model.yaml \
    --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
    --data_config_file configs/data/colmap/colmap_empty.yaml \
    --scan_parent_directory /path/to/reconstruction \
    --scan_name reconstruction_0:0 \ # reconstruction_name:n where n is the colmap sparse model
    --fast_cost_volume \
    --num_workers 8 \
    --batch_size 2 \
    --image_height 480 \
    --image_width 640 \
    --dump_depth_visualization
```
</details>


## Running Gaussian splatting with MVSAnywhere regularisation!

https://github.com/user-attachments/assets/12a6bb3f-fe9c-48ed-8982-e55c59dfd14d

We release code `regsplatfacto` to run splatting using MVSAnywhere depths as regularisation. This is heavily inspired by techniques such as [DN-Splatter](https://maturk.github.io/dn-splatter/) and [VCR-Gauss](https://hlinchen.github.io/projects/VCR-GauS/).

You can use any data in the nerfstudio format - e.g. existing nerfstudio data, or data from the 3 sources listed above.

If you are using data which has camera distortion, you will need to run our script `scripts/data_scripts/undistort_nerfstudio_data.py`:
```shell
python3 scripts/data_scripts/undistort_nerfstudio_data.py \
    --data-dir /path/to/input/scene \
    --output-dir /path/to/output/scene
```

Additionally, the [NeRF Capture](https://github.com/jc211/NeRFCapture) app saves frame metadata without file extension. To run splatting you will need to run our script `scripts/data_scripts/fix_nerfcapture_filenames.py`.

To train a splat, you can use 
```shell
ns-train regsplatfacto \
    --data path/to/data \
    --experiment-name mvsanywhere-splatting \
    --pipeline.datamanager.load_weights_from_checkpoint path/to/model \
    --pipeline.model.use-skybox False
```
This will first run mvsanywhere inference and save outputs to disk, and then start training your splat. 

> Tips:
> * If your data was captured with a phone in portrait mode, you can append the flag `--pipeline.datamanager.rotate_images True`.
> * If your data contains a lot of sky, you can try adding a background skybox using `--pipeline.model.use-skybox True`.

Once you have a splat, you can extract a mesh using TSDF fusion, using
```shell
ns-render-for-meshing \
    --load-config /path/to/splat/config \
    --rescale_to_world True \
    --output_path /path/to/render/outputs
ns-meshing \
    --renders-path /path/to/render/outputs \
    --max_depth 20.0  \
    --save-name mvsanywhere_mesh  \
    --voxel_size 0.04
```
If you are running on a scene reconstructed without metric scale (e.g. COLMAP), then you will need to adjust the `max_depth` and `voxel_size` to be something sensible for your scale.

Congratulations - you now have a splat and a mesh!

## 📊 Testing and Evaluation

### Robust Multi-View Depth Benchmark (RMVD)

We used the [Robust Multi-View Depth Benchmark](https://github.com/lmb-freiburg/robustmvd/) to evaluate MVSAnywhere depth estimation on a zero-shot environment with multiple datasets.

To evaluate MVSAnywhere on this benchmark, first, download the benchmark code in your system:

```shell
git clone https://github.com/lmb-freiburg/robustmvd.git
```

Now, download and preprocess the evaluation datasets following [this guide](https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/README.md). You should download:
- KITTI
- Scannet
- ETH3D
- DTU
- Tanks and Temples

Don't forget to set the path to these datasets in `rmvd/data/paths.toml`. Now you are ready to evaluate MVSAnywhere by just running:

```shell
export PYTHONPATH="/path/to/robustmvd/:$PYTHONPATH"

python src/mvsanywhere/test_rmvd.py \
    --name mvsanywhere \
    --output_base_path OUTPUT_PATH \
    --config_file configs/models/mvsanywhere_model.yaml \
    --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt
```

## 🔨 Training

To train MVSAnywhere:

1. Download all the required synthetic datasets (and val dataset):

    <details>
    <summary>Hypersim</summary>

    * Download following instructions from [here](https://github.com/apple/ml-hypersim):
    ```shell
    python code/python/tools/dataset_download_images.py \
      --downloads_dir path/to/download \
      --decompress_dir /path/to/hypersim/raw
    ```
    * Update `configs/data/hypersim/hypersim_default_train.yaml` to point to the correct location.
    * Convert distances into planar depth using the provided script in this repo:
    ```shell
    python ./data_scripts/generate_hypersim_planar_depths.py \
            --data_config configs/data/hypersim_default_train.yaml \
            --num_workers 8 
    ```
    </details>
    <details>
    <summary>TartanAir</summary>

    * Download following instructions from [here](https://github.com/apple/ml-hypersim):
    ```shell
    python download_training.py \
      --output-dir /path/to/tartan \
      --rgb \
      --depth \
      --seg \
      --only-left \
      --unzip
    ```
    * Update `configs/data/tartanair/tartanair_default_train.yaml` to point to the correct location.
    </details>

    <details>
    <summary>BlendedMVG</summary>

    * Download following instructions from [here](https://github.com/YoYo000/BlendedMVS).
    * You should download BlendedMVS, BlendedMVS+ and BlendedMVS++, all low-res. Place all on the same folder.
    * Update `configs/data/blendedmvg/blendedmvg_default_train.yaml` to point to the correct location.
    </details>
    <details>
    <summary>MatrixCity</summary>

    * Download following instructions from [here](https://github.com/city-super/MatrixCity).
    * You should download big_city, big_city_depth, big_city_depth_float32.
    * Update `configs/data/matrix_city/matrix_city_default_train.yaml` to point to the correct location.
    </details>
    <details>
    <summary>VKITTI2</summary>

    * Download following instructions from [here](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/).
    * You should download rgb, depth, classSegmentation and textgt.
    * Update `configs/data/vkitti/vkitti_default_train.yaml` to point to the correct location.
    </details>
    <details>
    <summary>Dynamic Replica</summary>

    * Download following instructions from [here](https://github.com/facebookresearch/dynamic_stereo).
    * After download you can remove unused stuff to save disk space (segmentation, optical flow and pixel trajectories.)
    * Update `configs/data/dynamic_replica/dynamic_replica_default_train.yaml` to point to the correct location.
    </details>

    <details>
    <summary>MVSSynth</summary>

    * Download following instructions from [here](https://phuang17.github.io/DeepMVS/mvs-synth.html).
    * You should download the 960x540 version.
    * Update `configs/data/mvssynth/mvssynth_default_train.yaml` to point to the correct location.
    </details>

    <details>
    <summary>SAIL-VOS 3D</summary>

    * Download following instructions from [here](https://sailvos.web.illinois.edu/_site/_site/index.html).
    * You will need to contact the authors to download the data.
    * Buy Grand Theft Auto V.
    * (optional, recommended) Play Grand Theft Auto V and relax a little bit.
    * Update `configs/data/sailvos3d/sailvos3d_default_train.yaml` to point to the correct location.
    </details>

    <details>
    <summary>ScanNet v2 (Optional, val only)</summary>

    * Follow the instructions from [here](https://github.com/nianticlabs/mvsanywhere/tree/main/scripts/data_scripts/scannet_wrangling_scripts).
    </details>

2. Download Depth Anything v2 base weights from [here](https://github.com/DepthAnything/Depth-Anything-V2).

3. Now you can train the model using:
```shell
python src/mvsanywhere/train.py \
  --log_dir logs/ \
  --name mvsanywhere_training \
  --config_file configs/models/mvsanywhere_model.yaml \
  --data_config configs/data/hypersim/hypersim_default_train.yaml:configs/data/tartanair/tartanair_default_train.yaml:configs/data/blendedmvg/blendedmvg_default_train.yaml:configs/data/matrix_city/matrix_city_default_train.yaml:configs/data/vkitti/vkitti_default_train.yaml:configs/data/dynamic_replica/dynamic_replica_default_train.yaml:configs/data/mvssynth/mvssynth_default_train.yaml:configs/data/sailvos3d/sailvos3d_default_train.yaml \
  --val_data_config configs/data/scannet/scannet_default_val.yaml \
  --batch_size 6 \
  --val_batch_size 6 \
  --da_weights_path /path/to/depth_anything_v2_vitb.pth \
  --gpus 2
```


## 📝🧮👩‍💻 Notation for Transformation Matrices

__TL;DR:__ `world_T_cam == world_from_cam`  
This repo uses the notation "cam_T_world" to denote a transformation from world to camera points (extrinsics). The intention is to make it so that the coordinate frame names would match on either side of the variable when used in multiplication from *right to left*:

    cam_points = cam_T_world @ world_points

`world_T_cam` denotes camera pose (from cam to world coords). `ref_T_src` denotes a transformation from a source to a reference view.  
Finally this notation allows for representing both rotations and translations such as: `world_R_cam` and `world_t_cam`

## 🗺️ World Coordinate System

This repo is geared towards ScanNet, so while its functionality should allow for any coordinate system (signaled via input flags), the model weights we provide assume a ScanNet coordinate system. This is important since we include ray information as part of metadata. Other datasets used with these weights should be transformed to the ScanNet system. The dataset classes we include will perform the appropriate transforms. 

## 🙏 Acknowledgements

The tuple generation scripts make heavy use of a modified version of DeepVideoMVS's [Keyframe buffer](https://github.com/ardaduz/deep-video-mvs/blob/master/dvmvs/keyframe_buffer.py) (thanks Arda and co!).

We'd like to thank the Niantic Raptor R\&D infrastructure team - Saki Shinoda, Jakub Powierza, and Stanimir Vichev - for their valuable infrastructure support.

## 📜 BibTeX

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{izquierdo2025mvsanywhere,
  title={{MVSAnywhere}: Zero Shot Multi-View Stereo},
  author={Izquierdo, Sergio and Sayed, Mohamed and Firman, Michael and Garcia-Hernando, Guillermo and Turmukhambetov, Daniyar and Civera, Javier and Mac Aodha, Oisin and Brostow, Gabriel J. and Watson, Jamie},
  booktitle={CVPR},
  year={2025}
}
```

## 👩‍⚖️ License

Copyright © Niantic, Inc. 2024. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
