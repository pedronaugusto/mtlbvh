"""Python wrapper for MtlBVH — Metal-accelerated BVH queries."""
import torch
from . import _C

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

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
        # Normalize to torch tensors on CPU. The C++ constructor already pulls
        # inputs to CPU for the CPU-side BVH build; accept numpy only when
        # NumPy is available (some PyTorch builds compile without it).
        if not torch.is_tensor(vertices):
            if _HAS_NUMPY:
                vertices = torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32))
            else:
                vertices = torch.tensor(vertices, dtype=torch.float32)
        if not torch.is_tensor(triangles):
            if _HAS_NUMPY:
                triangles = torch.from_numpy(np.ascontiguousarray(triangles, dtype=np.int32))
            else:
                triangles = torch.tensor(triangles, dtype=torch.int32)

        vertices = vertices.detach().to(torch.float32).contiguous()
        triangles = triangles.detach().to(torch.int32).contiguous()

        assert triangles.shape[0] > 8, f"BVH needs at least 8 triangles, got {triangles.shape[0]}"

        self._impl = _C.MtlBVH(vertices, triangles)

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
