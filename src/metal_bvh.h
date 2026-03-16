// Metal BVH host — builds BVH on CPU, dispatches queries on Metal GPU.
#pragma once

#import <Metal/Metal.h>
#import <torch/extension.h>
#include "bvh_types.h"
#include <vector>

class MtlBVHImpl {
public:
    MtlBVHImpl(const torch::Tensor& vertices, const torch::Tensor& triangles);

    // Unsigned distance: closest point on mesh.
    // Returns: (distances [N], face_id [N], optional uvw [N,3])
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    unsigned_distance(const torch::Tensor& positions, bool return_uvw);

    // Signed distance: watertight or raystab mode.
    // Returns: (distances [N], face_id [N], optional uvw [N,3])
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    signed_distance(const torch::Tensor& positions, bool return_uvw, int mode);

    // Ray trace: ray-triangle intersection.
    // Returns: (positions [N,3], face_id [N], depth [N])
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    ray_trace(const torch::Tensor& rays_o, const torch::Tensor& rays_d);

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;

    // GPU buffers for BVH data (uploaded once at construction)
    id<MTLBuffer> nodes_buf_;
    id<MTLBuffer> tris_buf_;
    uint32_t num_nodes_;
    uint32_t num_triangles_;

    // Cached pipeline state objects (created once in constructor)
    id<MTLComputePipelineState> pso_unsigned_;
    id<MTLComputePipelineState> pso_signed_watertight_;
    id<MTLComputePipelineState> pso_signed_raystab_;
    id<MTLComputePipelineState> pso_raytrace_;
};
