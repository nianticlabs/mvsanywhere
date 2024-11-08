"""
From: https://github.com/phuang17/DeepMVS/issues/13#issuecomment-642519445
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from numpy.linalg import inv
from PIL import Image
import imageio
import json,torch
from torchvision.transforms import *
from torchvision.utils import *
import torch.nn.functional as F
import cv2

# path = "./data/MVSSynth/0000/"
path = "/mnt/nas3/shared/datasets/mvssynth/GTAV_540/0000"
image1_idx = 0
image2_idx = 4


def read_img_depth_pose(i):
    img_path = path + "/images/%04d.png" % i
    depth_path = path + "/depths/%04d.exr" % i
    pose_path = path + "/poses/%04d.json" % i

    img = np.array(Image.open(img_path))

    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    raw_depth = np.clip(raw_depth, 0.1, 1000.0)

    im_height, im_width = 540, 810

    with open(pose_path) as f:
        r_info = json.load(f)

        # Converting intrinsics into [-1, 1] coordinate system
        # See https://github.com/phuang17/DeepMVS/issues/13#issuecomment-642539491
        c_x = r_info["c_x"] / im_width * 2.0 - 1.0
        c_y = r_info["c_y"] / im_height * 2.0 - 1.0
        f_x = r_info["f_x"] * 2 * 810 / 1920
        f_x = f_x / im_width * 2.0
        f_y = r_info["f_y"] / im_height * 2.0

        extrinsic = np.array(r_info["extrinsic"])

        extrinsic = inv(extrinsic)

    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    print("After", K)

    return img, raw_depth, K, extrinsic


img1, depth1, K, T1 = read_img_depth_pose(image1_idx)
img2, depth2, K, T2 = read_img_depth_pose(image2_idx)

# Save the depth image
depth1_vis = np.clip(depth1 / 60 * 255, 0, 255).astype(np.uint8)
cv2.imwrite("depth1.png", depth1_vis)

left2right = np.dot(inv(T2), T1)
left2right_r = left2right[:3, :3]
left2right_t = left2right[:3, 3]
left2right_t = left2right_t.reshape(3,1)

H, W, _ = img1.shape

xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
xs = xs.reshape(1, H, W)
ys = ys.reshape(1, H, W)
depth_gt = depth1.reshape(1,H,W)
uv_d1 = np.vstack((xs*depth_gt,ys*depth_gt,depth_gt))
uv_d1 = uv_d1.reshape(3,-1)
uv_c1 = np.dot(inv(K),uv_d1)
uv_c2 = np.dot(left2right_r,uv_c1)+left2right_t
uv_d2 = np.dot(K,uv_c2)
uv2 = uv_d2[0:2]/uv_d2[2]
uv2 = uv2.reshape(2,H,W).astype(np.float32)

#Create img tensor
img1_tensor = ToTensor()(img1).unsqueeze(0)
img2_tensor = ToTensor()(img2).unsqueeze(0)
sampler = torch.from_numpy(uv2).permute(1,2,0).unsqueeze(0)
print(sampler.min(), sampler.max())

# Sample pixels from the reference image to the query image
img_warp = F.grid_sample(img2_tensor,sampler,align_corners=True)
print(img_warp.max(), img_warp.min())

#Save Image
print(img1_tensor.shape, img_warp.shape)
save_image(img1_tensor[0], './1.png')
save_image(img2_tensor[0], './2.png')
save_image(img_warp[0], './2_warped_to_1.png')

print('Done')