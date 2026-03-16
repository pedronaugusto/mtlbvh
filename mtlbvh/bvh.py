"""Python wrapper for MtlBVH — Metal-accelerated BVH queries."""
import torch
import numpy as np
from . import _C

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

class MtlBVH:
    """Metal-accelerated BVH for distance queries and ray tracing.

    Drop-in replacement for GPU BVH implementations.
    """
    def __init__(self, vertices, triangles):
        """Build BVH from mesh.

        Args:
            vertices: [N, 3] float32 (numpy or torch)
            triangles: [M, 3] int32 (numpy or torch)
        """
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles):
            triangles = triangles.detach().cpu().numpy()

        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(triangles, dtype=np.int32)

        assert triangles.shape[0] > 8, f"BVH needs at least 8 triangles, got {triangles.shape[0]}"

        self._impl = _C.MtlBVH(
            torch.from_numpy(vertices),
            torch.from_numpy(triangles),
        )

    def unsigned_distance(self, positions, return_uvw=False):
        """Compute unsigned distance to mesh surface.

        Args:
            positions: [N, 3] float32 torch tensor
            return_uvw: if True, also return barycentric coordinates

        Returns:
            (distances [N], face_id [N], uvw [N,3] or None)
        """
        positions = positions.float().contiguous()
        return self._impl.unsigned_distance(positions, return_uvw)

    def signed_distance(self, positions, return_uvw=False, mode='watertight'):
        """Compute signed distance to mesh surface.

        Args:
            positions: [N, 3] float32 torch tensor
            return_uvw: if True, also return barycentric coordinates
            mode: 'watertight' or 'raystab'

        Returns:
            (distances [N], face_id [N], uvw [N,3] or None)
        """
        positions = positions.float().contiguous()
        return self._impl.signed_distance(positions, return_uvw, _sdf_mode_to_id[mode])

    def ray_trace(self, rays_o, rays_d):
        """Ray-triangle intersection.

        Args:
            rays_o: [N, 3] float32 torch tensor (ray origins)
            rays_d: [N, 3] float32 torch tensor (ray directions)

        Returns:
            (positions [N,3], face_id [N], depth [N])
        """
        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()
        return self._impl.ray_trace(rays_o, rays_d)
