from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import tqdm
import tyro
from nerfstudio.utils.rich_utils import CONSOLE

from regsplatfacto.utils import OPENGL_TO_OPENCV, force_to_hw3
from mvsanywhere.tools.fusers_helper import Open3DFuser


def fuse_npz_folder(renders_path: Path, fuser: Open3DFuser, key_to_fuse_colour: str = "image_hw3") -> None:
    """Fuse a folder of npz files into the TSDF volume.

    Args:
        renders_path (Path): Path to the renders directory.
        fuser (Open3DFuser): The Open3DFuser object to use for fusion.
        key_to_fuse_colour (str): The key in the npz file that contains the data to fuse into
            the RGB channels of the TSDF.
    """
    frame_paths = sorted(list(renders_path.glob("*.npz")))

    if len(frame_paths) == 0:
        raise ValueError(f"No npz files found in {renders_path}")

    for frame_path in tqdm.tqdm(frame_paths):
        frame_data = np.load(frame_path)

        image_hw3 = force_to_hw3(np.array(frame_data[key_to_fuse_colour]))
        depth_hw = np.array(frame_data["depth_hw"])
        K_44 = np.array(frame_data["intrinsics_44"])
        cam2world_44 = np.array(frame_data["cam_to_world_44"])

        # change of basis from opengl to opencv
        cam2world_44 = OPENGL_TO_OPENCV @ cam2world_44 @ OPENGL_TO_OPENCV.T
        world2cam_44 = np.linalg.inv(cam2world_44)

        image_13hw = torch.tensor(image_hw3).permute(2, 0, 1).unsqueeze(0).float()
        depth_11hw = torch.tensor(depth_hw).unsqueeze(0).unsqueeze(0).float()
        K_144 = torch.tensor(K_44).unsqueeze(0).float()
        world2cam_144 = torch.tensor(world2cam_44).unsqueeze(0).float()

        fuser.fuse_frames(
            depths_b1hw=depth_11hw,
            K_b44=K_144,
            cam_T_world_b44=world2cam_144,
            color_b3hw=image_13hw,
            apply_reverse_imagenet_normalize=False,
        )


def main(
    renders_path: Path,
    save_name: str,
    voxel_size: float = 0.04,
    max_depth: float = 100.0,
    truncation: float = 5.0,
    min_cluster_size: int = 100,
    key_to_fuse_colour: str = "image_hw3",
) -> None:
    """
    Fuses a sequence of renders into a mesh using Open3D.

    Known issues:
        - The voxel_size and max_depth parameters are hard to tune because of the unknown scale.
        - For outdoor scenes we probably need some kind of LoD meshing to be able to reconstruct
            foreground objects at a high level of detail while also being able to reconstruct a
            large area.

    Args:
        renders_path (Path): Path to the renders directory. Each render is assumed to be an
            npz file, containing the following keys:
                - image_hw3: The color image, in range [0, 1]
                - depth_hw: The depth image.
                - intrinsics_44: The camera intrinsics, in unnormalised (pixel) coordinates.
                - cam_to_world_44: The camera-to-world transformation in OpenGL coordinates.
        save_name (str): Name of the mesh file to save. The mesh will be saved to
            <renders_path>/<save_name>.ply.
        voxel_size (float): The size of each voxel in the TSDF volume.
        max_depth (float): The maximum depth for the fusion; depths beyond this value
            will not be fused into the TSDF.
        truncation (float): Truncation band for the TSDF.
        min_cluster_size (int): If will remove any clusters smaller than this value from the mesh.
            Set to 0 to avoid any cluster removal. Defaults to 100.
        key_to_fuse_colour (str): The key in the npz file that contains the data to fuse into the
            RGB channels of the mesh. Defaults to "image_hw3".

    """

    # check we have a valid save_name first to prevent finding this out at the end
    mesh_save_path = (renders_path / save_name).with_suffix(".ply")
    if not mesh_save_path.parent.exists():
        raise ValueError(f"Parent directory for save path {mesh_save_path} does not exist.")

    CONSOLE.print(f"Creating TSDF volume with voxel size {voxel_size} and max depth {max_depth}.")

    fuser = Open3DFuser(
        fusion_resolution=voxel_size,
        max_fusion_depth=max_depth,
        fuse_color=True,
        truncation=truncation,
    )

    CONSOLE.print(f"Fusing frames from {renders_path} into the TSDF.")

    fuse_npz_folder(renders_path, fuser=fuser, key_to_fuse_colour=key_to_fuse_colour)

    mesh = fuser.get_mesh()

    if min_cluster_size > 0:
        CONSOLE.print("Removing small clusters! Disable by setting --min-cluster-size 0")
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_cluster_size
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh = mesh.remove_unreferenced_vertices()

    # convert back to opengl so the mesh is aligned with the splat
    mesh_opengl = mesh.transform(OPENGL_TO_OPENCV)

    # use with_suffix so if save_name already ends with .ply nothing new will be added
    CONSOLE.print(f"Saving mesh to {mesh_save_path}")
    o3d.io.write_triangle_mesh(str(mesh_save_path), mesh_opengl)


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
