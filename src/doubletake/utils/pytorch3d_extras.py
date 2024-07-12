from typing import List, Optional, Tuple

import torch
from pytorch3d.transforms import Translate
from torch.autograd import Function
from torch.utils.cpp_extension import load

# JIT compile the marching cubes implementation
marching_cubes_impl = load(
    name="ext",
    sources=[
        "src/doubletake/tools/marching_cubes/ext.cpp",
        "src/doubletake/tools/marching_cubes/marching_cubes_cpu.cpp",
        "src/doubletake/tools/marching_cubes/marching_cubes.cu",
    ],
    verbose=True,
)


class _marching_cubes(Function):
    """
    Torch Function wrapper for marching_cubes implementation.
    This function is not differentiable. An autograd wrapper is used
    to ensure an error if user tries to get gradients.
    """

    @staticmethod
    def forward(ctx, vol, isolevel, active_voxels, min_bounds, max_bounds):
        verts, faces, ids = marching_cubes_impl.marching_cubes_(
            vol, isolevel, active_voxels, min_bounds, max_bounds
        )
        return verts, faces, ids

    @staticmethod
    def backward(ctx, grad_verts, grad_faces):
        raise ValueError("marching_cubes backward is not supported")


def marching_cubes(
    vol_batch: torch.Tensor,
    active_voxels: torch.Tensor,
    isolevel: Optional[float] = None,
    return_local_coords: bool = True,
    min_bounds: Optional[torch.Tensor] = None,
    max_bounds: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run marching cubes over a volume scalar field with a designated isolevel.
    Returns vertices and faces of the obtained mesh.
    This operation is non-differentiable.

    Handles verts and faces flips from pytorch3d to match trimesh.

    Args:
        vol_batch: a Tensor of size (N, D, H, W) corresponding to
            a batch of 3D scalar fields
        isolevel: float used as threshold to determine if a point is inside/outside
            the volume.  If None, then the average of the maximum and minimum value
            of the scalar field is used.
        return_local_coords: bool. If True the output vertices will be in local coordinates in
            the range [-1, 1] x [-1, 1] x [-1, 1]. If False they will be in the range
            [0, W-1] x [0, H-1] x [0, D-1]

    Returns:
        verts: [{V_0}, {V_1}, ...] List of N sets of vertices of shape (|V_i|, 3) in FloatTensor
        faces: [{F_0}, {F_1}, ...] List of N sets of faces of shape (|F_i|, 3) in LongTensors
    """

    if min_bounds is None:
        min_bounds = torch.ones(3, device=vol_batch.device).int() * -10000
        max_bounds = torch.ones(3, device=vol_batch.device).int() * 10000

    batched_verts, batched_faces = [], []
    D, H, W = vol_batch.shape[1:]
    for i in range(len(vol_batch)):
        vol = vol_batch[i]
        thresh = ((vol.max() + vol.min()) / 2).item() if isolevel is None else isolevel
        verts, faces, ids = _marching_cubes.apply(
            vol, thresh, active_voxels, min_bounds, max_bounds
        )
        if len(faces) > 0 and len(verts) > 0:
            # Convert from world coordinates ([0, D-1], [0, H-1], [0, W-1]) to
            # local coordinates in the range [-1, 1]
            if return_local_coords:
                verts = (
                    Translate(x=+1.0, y=+1.0, z=+1.0, device=vol.device)
                    .scale((vol.new_tensor([W, H, D])[None] - 1) * 0.5)
                    .inverse()
                ).transform_points(verts[None])[0]
            # deduplication for cuda
            if vol.is_cuda:
                unique_ids, inverse_idx = torch.unique(ids, return_inverse=True)
                verts_ = verts.new_zeros(unique_ids.shape[0], 3)
                verts_[inverse_idx] = verts
                verts = verts_
                faces = inverse_idx[faces]

            # Flip verts and faces to come back from pytorch3d to our coord convention
            verts = verts[:, [2, 1, 0]]
            faces = faces.flip(1)

            batched_verts.append(verts)
            batched_faces.append(faces)
        else:
            batched_verts.append([])
            batched_faces.append([])
    return batched_verts, batched_faces
