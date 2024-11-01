#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Preprocessing code for the WayMo Open dataset
# dataset at https://github.com/waymo-research/waymo-open-dataset
# 1) Accept the license
# 2) download all training/*.tfrecord files from Perception Dataset, version 1.4.2
# 3) put all .tfrecord files in '/path/to/waymo_dir'
# 4) install the waymo_open_dataset package with
#    `python3 -m pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4`
# 5) execute this script as `python preprocess_waymo.py --waymo_dir /path/to/waymo_dir`
# --------------------------------------------------------
import sys
import os
import os.path as osp
import shutil
import json
from tqdm import tqdm
import PIL.Image
import torch
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

# import path_to_root  # noqa
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K

def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')



def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    """ Jointly rescale a (image, depthmap)
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(output_resolution, resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics


### END IMPORTS ####
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count


def parallel_map(function, args, workers=0, star_args=False, kw_args=False, front_num=1, Pool=ThreadPool, **tqdm_kw):
    """ tqdm but with parallel execution.

    Will essentially return
      res = [ function(arg) # default
              function(*arg) # if star_args is True
              function(**arg) # if kw_args is True
              for arg in args]

    Note:
        the <front_num> first elements of args will not be parallelized.
        This can be useful for debugging.
    """
    while workers <= 0:
        workers += cpu_count()
    if workers == 1:
        front_num = float('inf')

    # convert into an iterable
    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    # sequential execution first
    front = []
    while len(front) < front_num:
        try:
            a = next(args)
        except StopIteration:
            return front  # end of the iterable
        front.append(function(*a) if star_args else function(**a) if kw_args else function(a))

    # then parallel execution
    out = []
    with Pool(workers) as pool:
        # Pass the elements of args into function
        if star_args:
            futures = pool.imap(starcall, [(function, a) for a in args])
        elif kw_args:
            futures = pool.imap(starstarcall, [(function, a) for a in args])
        else:
            futures = pool.imap(function, args)
        # Print out the progress as tasks complete
        for f in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(f)
    return front + out


def parallel_processes(*args, **kwargs):
    """ Same as parallel_threads, with processes
    """
    import multiprocessing as mp
    kwargs['Pool'] = mp.Pool
    return parallel_threads(*args, **kwargs)



def show_raw_pointcloud(pts3d, colors, point_size=2):
    scene = trimesh.Scene()

    pct = trimesh.PointCloud(cat_3d(pts3d), colors=cat_3d(colors))
    scene.add_geometry(pct)

    scene.show(line_settings={'point_size': point_size})


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--waymo_dir', required=True)
    parser.add_argument('--output_dir', default='data/waymo_processed')
    parser.add_argument('--workers', type=int, default=1)
    return parser


def main(waymo_root, output_dir, workers=1):
    extract_frames(waymo_root, output_dir, workers=workers)
    make_crops(output_dir, workers=args.workers)
    shutil.rmtree(osp.join(output_dir, 'tmp'))
    print('Done! all data generated at', output_dir)


def _list_sequences(db_root):
    print('>> Looking for sequences in', db_root)
    res = sorted(f for f in os.listdir(db_root) if f.endswith('.tfrecord'))
    print(f'    found {len(res)} sequences')
    return res


def extract_frames(db_root, output_dir, workers=8):
    sequences = _list_sequences(db_root)
    output_dir = osp.join(output_dir, 'tmp')
    print('>> outputing result to', output_dir)
    args = [(db_root, output_dir, seq) for seq in sequences]
    parallel_map(process_one_seq, args, star_args=True, workers=workers)


def process_one_seq(db_root, output_dir, seq):
    out_dir = osp.join(output_dir, seq)
    os.makedirs(out_dir, exist_ok=True)
    calib_path = osp.join(out_dir, 'calib.json')
    if osp.isfile(calib_path):
        return

    try:
        with tf.device('/CPU:0'):
            calib, frames = extract_frames_one_seq(osp.join(db_root, seq))
    except RuntimeError:
        print(f'/!\\ Error with sequence {seq} /!\\', file=sys.stderr)
        return  # nothing is saved

    for f, (frame_name, views) in enumerate(tqdm(frames, leave=False)):
        for cam_idx, view in views.items():
            img = PIL.Image.fromarray(view.pop('img'))
            img.save(osp.join(out_dir, f'{f:05d}_{cam_idx}.jpg'))
            np.savez(osp.join(out_dir, f'{f:05d}_{cam_idx}.npz'), **view)

    with open(calib_path, 'w') as f:
        json.dump(calib, f)


def extract_frames_one_seq(filename):
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils

    print('>> Opening', filename)
    dataset = tf.data.TFRecordDataset(filename, compression_type='')

    calib = None
    frames = []

    for data in tqdm(dataset, leave=False):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        content = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _, range_image_top_pose = content

        views = {}
        frames.append((frame.context.name, views))

        # once in a sequence, read camera calibration info
        if calib is None:
            calib = []
            for cam in frame.context.camera_calibrations:
                calib.append((cam.name,
                              dict(width=cam.width,
                                   height=cam.height,
                                   intrinsics=list(cam.intrinsic),
                                   extrinsics=list(cam.extrinsic.transform))))

        # convert LIDAR to pointcloud
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)

        # The distance between lidar points and vehicle frame origin.
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        for i, image in enumerate(frame.images):
            # select relevant 3D points for this view
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            cp_points_msk_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)

            pose = np.asarray(image.pose.transform).reshape(4, 4)
            timestamp = image.pose_timestamp

            rgb = tf.image.decode_jpeg(image.image).numpy()

            pix = cp_points_msk_tensor[..., 1:3].numpy().round().astype(np.int16)
            pts3d = points_all[mask.numpy()]

            views[image.name] = dict(img=rgb, pose=pose, pixels=pix, pts3d=pts3d, timestamp=timestamp)

        if not 'show full point cloud':
            show_raw_pointcloud([v['pts3d'] for v in views.values()], [v['img'] for v in views.values()])

    return calib, frames


def make_crops(output_dir, workers=16, **kw):
    tmp_dir = osp.join(output_dir, 'tmp')
    sequences = _list_sequences(tmp_dir)
    args = [(tmp_dir, output_dir, seq) for seq in sequences]
    parallel_map(crop_one_seq, args, star_args=True, workers=workers, front_num=0)


def crop_one_seq(input_dir, output_dir, seq, resolution=512):
    seq_dir = osp.join(input_dir, seq)
    out_dir = osp.join(output_dir, seq)
    if osp.isfile(osp.join(out_dir, '00100_1.jpg')):
        return
    os.makedirs(out_dir, exist_ok=True)

    # load calibration file
    try:
        with open(osp.join(seq_dir, 'calib.json')) as f:
            calib = json.load(f)
    except IOError:
        print(f'/!\\ Error: Missing calib.json in sequence {seq} /!\\', file=sys.stderr)
        return

    axes_transformation = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]])

    cam_K = {}
    cam_distortion = {}
    cam_res = {}
    cam_to_car = {}
    for cam_idx, cam_info in calib:
        cam_idx = str(cam_idx)
        cam_res[cam_idx] = (W, H) = (cam_info['width'], cam_info['height'])
        f1, f2, cx, cy, k1, k2, p1, p2, k3 = cam_info['intrinsics']
        cam_K[cam_idx] = np.asarray([(f1, 0, cx), (0, f2, cy), (0, 0, 1)])
        cam_distortion[cam_idx] = np.asarray([k1, k2, p1, p2, k3])
        cam_to_car[cam_idx] = np.asarray(cam_info['extrinsics']).reshape(4, 4)  # cam-to-vehicle

    frames = sorted(f[:-3] for f in os.listdir(seq_dir) if f.endswith('.jpg'))

    # from dust3r.viz import SceneViz
    # viz = SceneViz()

    for frame in tqdm(frames, leave=False):
        cam_idx = frame[-2]  # cam index
        assert cam_idx in '12345', f'bad {cam_idx=} in {frame=}'
        data = np.load(osp.join(seq_dir, frame + 'npz'))
        car_to_world = data['pose']
        W, H = cam_res[cam_idx]

        # load depthmap
        pos2d = data['pixels'].round().astype(np.uint16)
        x, y = pos2d.T
        pts3d = data['pts3d']  # already in the car frame
        pts3d = geotrf(axes_transformation @ inv(cam_to_car[cam_idx]), pts3d)
        # X=LEFT_RIGHT y=ALTITUDE z=DEPTH

        # load image
        image = imread_cv2(osp.join(seq_dir, frame + 'jpg'))

        # downscale image
        output_resolution = (resolution, 1) if W > H else (1, resolution)
        image, _, intrinsics2 = rescale_image_depthmap(image, None, cam_K[cam_idx], output_resolution)
        image.save(osp.join(out_dir, frame + 'jpg'), quality=80)

        # save as an EXR file? yes it's smaller (and easier to load)
        W, H = image.size
        depthmap = np.zeros((H, W), dtype=np.float32)
        pos2d = geotrf(intrinsics2 @ inv(cam_K[cam_idx]), pos2d).round().astype(np.int16)
        x, y = pos2d.T
        depthmap[y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)] = pts3d[:, 2]
        cv2.imwrite(osp.join(out_dir, frame + 'exr'), depthmap)

        # save camera parametes
        cam2world = car_to_world @ cam_to_car[cam_idx] @ inv(axes_transformation)
        np.savez(osp.join(out_dir, frame + 'npz'), intrinsics=intrinsics2,
                 cam2world=cam2world, distortion=cam_distortion[cam_idx])

        # viz.add_rgbd(np.asarray(image), depthmap, intrinsics2, cam2world)
    # viz.show()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.waymo_dir,  args.output_dir, workers=args.workers)