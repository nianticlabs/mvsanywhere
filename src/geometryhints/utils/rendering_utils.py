from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.utils import cameras_from_opencv_projection
import torch

class PyTorch3DMeshDepthRenderer:
    def __init__(self, height=192, width=256) -> None:
        self.height = height
        self.width = width

        self.raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
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
        cams = cameras_from_opencv_projection(R=R, tvec=T, camera_matrix=K, image_size=image_size)

        rasterizer = MeshRasterizer(
            cameras=cams,
            raster_settings=self.raster_settings,
        )

        mesh = mesh.cuda()
        cams = cams.cuda()
        _mesh = mesh.extend(len(cams))

        fragments = rasterizer(_mesh)

        depth_bhw1 = fragments.zbuf
        depth_b1hw = depth_bhw1.permute(0, 3, 1, 2)

        return depth_b1hw
