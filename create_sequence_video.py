import matplotlib.pyplot as plt
import cv2
import numpy as np

NUM_FRAMES = 70

depth_maps_1 = []
depth_maps_2 = []
rgb_images = []
for idx in range(NUM_FRAMES):
    depth_maps_1.append(np.load(
        f"results/extra_sky/tanks_and_temples_sequence/qualitative/{idx:07d}-pred_depth.npy"
    ))
    depth_maps_2.append(np.load(
        f"results/hero_3_mlp_allsynth_hr/tanks_and_temples_sequence/qualitative/{idx:07d}-pred_depth.npy"
    ))
    rgb_images.append(cv2.imread(
        f"/mnt/nas3/shared/datasets/tanks_and_temples/Barn/images/{idx+5:08d}.jpg"  
    ))

depth_maps_1 = np.concatenate(depth_maps_1, axis=0)
depth_maps_2 = np.concatenate(depth_maps_2, axis=0)
rgb_images = np.stack(rgb_images, axis=0)

H, W = depth_maps_1.shape[-2:]

video_1=cv2.VideoWriter('results/barn_video/sky_video_depth_sky.avi', cv2.VideoWriter_fourcc(*'mp4v'), 7, (W // 4, H // 4))
video_2=cv2.VideoWriter('results/barn_video/sky_video_depth_ours.avi', cv2.VideoWriter_fourcc(*'mp4v'), 7, (W // 2, H // 4))
video_3=cv2.VideoWriter('results/barn_video/sky_video_depth_rgb.avi', cv2.VideoWriter_fourcc(*'mp4v'), 7, (W // 4, H // 4))

cm = plt.get_cmap('jet')
max_depth = max(depth_maps_1.max(), depth_maps_2.max())
max_depth = depth_maps_2.max()
min_depth = min(depth_maps_1.min(), depth_maps_2.min())

for i in range(NUM_FRAMES):
    img_1 = np.uint8(np.clip(cm((depth_maps_1[i] - min_depth) / (max_depth - min_depth))[..., :3], 0, 1.0) * 255)
    img_1 = cv2.resize(img_1, (W // 4, H // 4))

    depth_2 = np.hstack([depth_maps_1[i], depth_maps_2[i]])
    img_2 = np.uint8(np.clip(cm((depth_2 - min_depth) / (max_depth - min_depth))[..., :3], 0, 1.0) * 255)
    img_2 = cv2.resize(img_2, (W // 2, H // 4))

    img_3 = cv2.resize(rgb_images[i], (W // 4, H // 4))

    video_1.write(img_1)
    video_2.write(img_2)
    video_3.write(img_3)

video_1.release()
video_2.release()
video_3.release()





# NUM_FRAMES = 200

# depth_maps_1 = []
# depth_maps_2 = []
# rgb_images = []
# for idx in range(0, NUM_FRAMES):
#     depth_maps_1.append(np.load(
#         f"results/mast3r_avg_sergio/kitti_sequence/qualitative/{idx:07d}-pred_invdepth.npy"
#     ))
#     depth_maps_2.append(np.load(
#         f"results/hero_3_mlp_allsynth_hr/kitti_sequence/qualitative/{idx:07d}-pred_invdepth.npy"
#     ))
#     rgb_images.append(cv2.imread(
#         f"/mnt/nas3/shared/datasets/kitti/raw/2011_09_30/2011_09_30_drive_0018_sync/image_02/data/{idx+2400:010d}.png"  
#     ))

# depth_maps_1 = np.concatenate(depth_maps_1, axis=0)
# depth_maps_2 = np.concatenate(depth_maps_2, axis=0)
# rgb_images = np.stack(rgb_images, axis=0)

# H, W = depth_maps_1.shape[-2:]

# video_1=cv2.VideoWriter('results/kitti_video/video_mast3r.avi', cv2.VideoWriter_fourcc(*'mp4v'), 7, (W // 2, H // 2))
# video_2=cv2.VideoWriter('results/kitti_video/video_ours.avi', cv2.VideoWriter_fourcc(*'mp4v'), 7, (W // 2, H // 2))
# video_3=cv2.VideoWriter('results/kitti_video/video_rgb.avi', cv2.VideoWriter_fourcc(*'mp4v'), 7, (W // 2, H // 2))

# cm = plt.get_cmap('jet')
# max_depth = max(depth_maps_1.max(), depth_maps_2.max())
# # max_depth = 2.0
# min_depth = min(depth_maps_1.min(), depth_maps_2.min())

# for i in range(len(depth_maps_1)):
#     img_1 = np.uint8(np.clip(cm((depth_maps_1[i] - min_depth) / (max_depth - min_depth))[..., :3], 0, 1.0) * 255)
#     img_1 = cv2.resize(img_1, (W // 2, H // 2))

#     img_2 = np.uint8(np.clip(cm((depth_maps_2[i] - min_depth) / (max_depth - min_depth))[..., :3], 0, 1.0) * 255)
#     img_2 = cv2.resize(img_2, (W // 2, H // 2))

#     img_3 = cv2.resize(rgb_images[i], (W // 2, H // 2))

#     video_1.write(img_1)
#     video_2.write(img_2)
#     video_3.write(img_3)

# video_1.release()
# video_2.release()
# video_3.release()