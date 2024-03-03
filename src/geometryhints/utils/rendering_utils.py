from pathlib import Path
import torch
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.utils import cameras_from_opencv_projection
import trimesh


class PyTorch3DMeshDepthRenderer:
    def __init__(self, height=192, width=256) -> None:
        self.height = height
        self.width = width

        self.raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
        )

        self.rasterizer = MeshRasterizer(
            raster_settings=self.raster_settings,
        )

    def render(self, mesh, cam_T_world_b44, K_b44):
        """Renders a mesh with a given pose and **normalized** intrinsics."""
        image_size = (
            torch.tensor((self.height, self.width)).unsqueeze(0).expand(cam_T_world_b44.shape[0], 2)
        )

        R = cam_T_world_b44[:, :3, :3]
        T = cam_T_world_b44[:, :3, 3]
        K = K_b44.clone()
        K[:, 0] *= self.width
        K[:, 1] *= self.height
        cams = cameras_from_opencv_projection(
            R=R, tvec=T, camera_matrix=K, image_size=image_size
        ).cuda()

        _mesh = mesh.extend(len(cams))

        fragments = self.rasterizer(_mesh, cameras=cams)

        depth_bhw1 = fragments.zbuf
        depth_b1hw = depth_bhw1.permute(0, 3, 1, 2)

        return depth_b1hw

def load_and_preprocess_mesh_for_rendering(
    mesh_load_path: Path,
    scan: str,
    color_with: str,
) -> trimesh.Trimesh:
    """Load a mesh and preprocess it for rendering, e.g. removing ceiling and recolouring"""


    scene_trimesh_mesh = trimesh.load(mesh_load_path, force="mesh")


    if color_with == "raw":
        scene_trimesh_mesh.visual.face_colors = None
        scene_trimesh_mesh.visual.vertex_colors = None
    elif color_with == "normals":
        normals = scene_trimesh_mesh.vertex_normals.copy()
        normals = (normals + 1) / 2
        scene_trimesh_mesh.visual.vertex_colors = normals
    else:
        raise ValueError(f"Unknown color_with option {color_with}")

    return scene_trimesh_mesh
