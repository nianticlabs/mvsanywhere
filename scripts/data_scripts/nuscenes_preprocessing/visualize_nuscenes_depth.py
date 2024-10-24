import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from PIL import Image


def load_and_plot_depth_and_image(npy_file, jpg_file, threshold=0.0):
    depth_data = np.load(npy_file)

    # Print depth data statistics
    print(f"Depth data statistics:")
    print(f"  Data type: {depth_data.dtype}")
    print(f"  Shape: {depth_data.shape}")
    print(f"  Min value: {np.nanmin(depth_data)}")
    print(f"  Max value: {np.nanmax(depth_data)}")
    print(f"  Mean value: {np.nanmean(depth_data)}")
    print(f"  Median value: {np.nanmedian(depth_data)}")
    print(f"  Standard Deviation: {np.nanstd(depth_data)}")
    print(f"  Number of zeros: {np.sum(depth_data == 0)}")
    print(f"  Number of NaNs: {np.isnan(depth_data).sum()}")
    print(f"  Number of Infinities: {np.isinf(depth_data).sum()}")

    image = Image.open(jpg_file)

    masked_depth = ma.masked_where((depth_data <= threshold) | np.isnan(depth_data), depth_data)

    valid_points = np.sum(~masked_depth.mask)

    print(f"Number of valid depth points after masking: {valid_points}")
    if valid_points == 0:
        print("No valid depth points above the threshold. Consider lowering the threshold.")
        return

    vmin = np.nanmin(depth_data[depth_data > threshold])
    vmax = np.nanmax(depth_data)
    print(f"Adjusted color scale: vmin = {vmin}, vmax = {vmax}")

    plt.figure(figsize=(8, 4))
    plt.hist(depth_data[depth_data > 0].flatten(), bins=100)
    plt.title("Histogram of non-zero depth values")
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.show()

    y_indices, x_indices = np.nonzero(~masked_depth.mask)
    depth_values = masked_depth[y_indices, x_indices]

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    scatter = plt.scatter(
        x_indices, y_indices, c=depth_values, cmap="inferno", s=10, vmin=vmin, vmax=vmax
    )
    plt.colorbar(scatter, label="Depth")
    plt.title("Overlaid sparse depth on RGB")
    plt.axis("off")
    plt.show()


npy_file_path = "/mnt/nas/shared/datasets/nuscenes/depth/samples/CAM_FRONT_LEFT/n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915284904917.npy"
jpg_file_path = "/mnt/nas/shared/datasets/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915284904917.jpg"
load_and_plot_depth_and_image(npy_file_path, jpg_file_path)
