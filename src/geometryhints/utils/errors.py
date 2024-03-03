from typing import Optional, Union

import matplotlib as mpl
import open3d as o3d
import torch
import trimesh

from geometryhints.utils.volume_utils import SimpleVolume

T_ = Union[trimesh.Trimesh, torch.Tensor]


class ErrorVisualiser:
    def __init__(self, max_val: float):
        self.max_val = max_val

        normaliser = mpl.colors.Normalize(vmin=0, vmax=self.max_val)
        self.mapper = mpl.cm.ScalarMappable(norm=normaliser, cmap="jet")

    def compute_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, prediction: T_, target: T_) -> T_:
        raise NotImplementedError


class DepthErrorVisualiser(ErrorVisualiser):
    def __init__(self, max_val: float = 10):
        """Visualise the depth error map. The colormap is from blue (good predicted depth)
        to red (bad predicted depth).

        Params:
            max_val: worst depth discrepancy possible. Default is 10 m
        """
        super().__init__(max_val)

    def compute_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute abs-diff between predicted and target depth"""
        return torch.abs(prediction - target)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the depth error map given the predicted and the target depth. When mask is given,
        mask the depth error map according to the mask.

        Params:
            prediction: tensor (1,1,H,W) with the predicted depth
            target: tensor (1,1,H,W) with the target depth
            mask: optional tensor (1,1,H,W) with 1 for valid pixels and 0 otherwise
        Returns:
            a tensor (1,3,H,W) where the colour of each pixel represents the depth error.
        """
        assert prediction.shape == target.shape
        assert prediction.shape[0] == 1

        error = self.compute_error(prediction=prediction, target=target).squeeze().numpy()
        error_map = self.mapper.to_rgba(error)[:, :, :3]
        error_map = torch.tensor(error_map).permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            error_map = error_map * mask
        return error_map


class MeshErrorVisualiser(ErrorVisualiser):
    def __init__(self, max_val: float = 20):
        """Visualise the accuracy of each vertex in the mesh. The colormap is from blue (good accuracy)
        to red (bad accuracy).

        Params:
            max_val: worst accuracy possible. Default is 20 cm
        """
        super().__init__(max_val)

    def compute_error(self, prediction: trimesh.Trimesh, target: trimesh.Trimesh) -> torch.Tensor:
        """Compute accuracy error (i.e. from predicted to target meshes)

        Params:
            prediction: predicted mesh
            target: target mesh
        Returns:
            the accuracy score for each vertex in the predicted mesh (in cm)
        """

        prediction_vertices = prediction.vertices
        target_vertices = target.vertices

        prediction_pcd = o3d.geometry.PointCloud()
        prediction_pcd.points = o3d.utility.Vector3dVector(prediction_vertices)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_vertices)

        # compute accuracy
        distances_prediction_to_target = torch.tensor(
            prediction_pcd.compute_point_cloud_distance(target_pcd)
        )
        # convert m into cm
        return distances_prediction_to_target * 100

    def filter_mesh_by_visibility(self, mesh: trimesh.Trimesh, visibility_volume: SimpleVolume):
        points_pred = torch.tensor(mesh.vertices)

        visibility_volume.cuda()
        vis_samples_N = visibility_volume.sample_volume(points_pred)
        valid_mask_N = vis_samples_N > 0.5

        # prepare the indices
        indices_N = valid_mask_N.cpu().squeeze().bool().numpy()

        # remove vertices and faces with index = 0, ie is not visible
        mesh.update_faces(indices_N[mesh.faces].all(axis=1))

        return mesh

    def forward(
        self,
        prediction: trimesh.Trimesh,
        target: trimesh.Trimesh,
        visibility_volume: SimpleVolume,
    ) -> trimesh.Trimesh:
        """Given the vertices of the predicted and the target meshes, return a new mesh where the
        colour of each vertex represents the accuracy (in cm).

        Params:
            prediction: predicted mesh
            target: target mesh
            mask: optional tensor, representing the visibility volume
            transform: optional tensor, representing the world to grid transformation
        Returns:
            a mesh where the colour of each vertex represents the accuracy (blue means good, red means bad)
        """
        # assert (mask is None and transform is None) or (mask is not None and transform is not None)

        chamfer_distance = self.compute_error(prediction=prediction, target=target)
        error_mesh = trimesh.Trimesh(vertices=prediction.vertices, faces=prediction.faces)
        error_map = self.mapper.to_rgba(chamfer_distance)[:, :3]
        error_mesh.visual.vertex_colors = error_map
        # if mask is not None:
        error_mesh = self.filter_mesh_by_visibility(
            mesh=error_mesh,
            visibility_volume=visibility_volume,
        )
        return error_mesh
