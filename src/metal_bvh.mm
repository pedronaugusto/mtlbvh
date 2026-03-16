// Metal BVH host — builds BVH on CPU, dispatches queries on Metal GPU.
#import "metal_bvh.h"
#import "bvh_build.h"
#include <dlfcn.h>

#if __has_include(<torch/mps.h>)
#include <torch/mps.h>
#define MTLBVH_HAS_MPS 1
#else
#define MTLBVH_HAS_MPS 0
#endif

static void _mtlbvh_anchor() {}

// ─── Construction ────────────────────────────────────────────────────────────

MtlBVHImpl::MtlBVHImpl(const torch::Tensor& vertices, const torch::Tensor& triangles) {
    device_ = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(device_ != nil, "[mtlbvh] No Metal device found");
    queue_ = [device_ newCommandQueue];

    // Load metallib
    NSError* error = nil;
    Dl_info dl_info;
    dladdr((void*)&_mtlbvh_anchor, &dl_info);
    NSString* soPath = [NSString stringWithUTF8String:dl_info.dli_fname];
    NSString* dir = [soPath stringByDeletingLastPathComponent];
    NSString* libPath = [dir stringByAppendingPathComponent:@"mtlbvh.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
        library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
    }
    if (!library_) library_ = [device_ newDefaultLibrary];
    TORCH_CHECK(library_ != nil, "[mtlbvh] Failed to load Metal library");

    // Convert input to CPU numpy-compatible layout
    auto verts_cpu = vertices.detach().cpu().contiguous().to(torch::kFloat32);
    auto tris_cpu = triangles.detach().cpu().contiguous().to(torch::kInt32);

    int n_verts = verts_cpu.size(0);
    int n_tris = tris_cpu.size(0);
    TORCH_CHECK(n_tris > 8, "[mtlbvh] BVH needs at least 8 triangles, got ", n_tris);

    float* v_ptr = verts_cpu.data_ptr<float>();
    int* t_ptr = tris_cpu.data_ptr<int>();

    // Build triangle list
    std::vector<BvhTriangle> bvh_tris(n_tris);
    for (int i = 0; i < n_tris; ++i) {
        int i0 = t_ptr[i * 3 + 0];
        int i1 = t_ptr[i * 3 + 1];
        int i2 = t_ptr[i * 3 + 2];
        bvh_tris[i].a = {v_ptr[i0 * 3 + 0], v_ptr[i0 * 3 + 1], v_ptr[i0 * 3 + 2]};
        bvh_tris[i].b = {v_ptr[i1 * 3 + 0], v_ptr[i1 * 3 + 1], v_ptr[i1 * 3 + 2]};
        bvh_tris[i].c = {v_ptr[i2 * 3 + 0], v_ptr[i2 * 3 + 1], v_ptr[i2 * 3 + 2]};
        bvh_tris[i].id = (int64_t)i;
    }

    // Build BVH tree (CPU, reorders triangles)
    auto nodes = bvh_build(bvh_tris, 8);
    num_nodes_ = (uint32_t)nodes.size();
    num_triangles_ = (uint32_t)bvh_tris.size();

    // Upload to Metal buffers (eager copy)
    nodes_buf_ = [device_ newBufferWithBytes:nodes.data()
                                      length:nodes.size() * sizeof(BvhNode)
                                     options:MTLResourceStorageModeShared];
    tris_buf_ = [device_ newBufferWithBytes:bvh_tris.data()
                                     length:bvh_tris.size() * sizeof(BvhTriangle)
                                    options:MTLResourceStorageModeShared];

    // Pre-create all pipeline state objects
    auto make_pso = [&](const char* name) -> id<MTLComputePipelineState> {
        NSError* error = nil;
        id<MTLFunction> func = [library_ newFunctionWithName:
            [NSString stringWithUTF8String:name]];
        TORCH_CHECK(func != nil, "[mtlbvh] Metal function not found: ", name);
        id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:func error:&error];
        TORCH_CHECK(pso != nil, "[mtlbvh] Failed to create pipeline for: ", name);
        return pso;
    };
    pso_unsigned_ = make_pso("unsigned_distance_kernel");
    pso_signed_watertight_ = make_pso("signed_distance_watertight_kernel");
    pso_signed_raystab_ = make_pso("signed_distance_raystab_kernel");
    pso_raytrace_ = make_pso("raytrace_kernel");
}

// ─── Helper: create Metal buffer from tensor (zero-copy on Apple Silicon) ────

static id<MTLBuffer> tensor_to_buffer(id<MTLDevice> dev, const torch::Tensor& t) {
    size_t nbytes = t.nbytes();
    TORCH_CHECK(nbytes > 0, "[mtlbvh] empty tensor");
    id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:t.data_ptr()
                                               length:nbytes
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    TORCH_CHECK(buf != nil, "[mtlbvh] Failed to create buffer");
    return buf;
}

// ─── Unsigned distance ──────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MtlBVHImpl::unsigned_distance(const torch::Tensor& positions, bool return_uvw) {
    auto pos_f = positions.detach().contiguous().to(torch::kFloat32);
    auto prefix = pos_f.sizes().vec();
    prefix.pop_back();  // remove last dim (3)
    int64_t N = pos_f.numel() / 3;

    pos_f = pos_f.view({N, 3});

    auto distances = torch::empty({N}, torch::kFloat32);
    auto face_id = torch::empty({N}, torch::kInt64);
    auto uvw = return_uvw ? torch::empty({N, 3}, torch::kFloat32) : torch::zeros({1, 3}, torch::kFloat32);

    uint32_t n_elements = (uint32_t)N;
    uint32_t ret_uvw = return_uvw ? 1 : 0;

    auto pos_buf = tensor_to_buffer(device_, pos_f);
    auto dist_buf = tensor_to_buffer(device_, distances);
    auto fid_buf = tensor_to_buffer(device_, face_id);
    auto uvw_buf = tensor_to_buffer(device_, uvw);

    auto pso = pso_unsigned_;
    id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:pos_buf    offset:0 atIndex:0];
    [enc setBuffer:dist_buf   offset:0 atIndex:1];
    [enc setBuffer:fid_buf    offset:0 atIndex:2];
    [enc setBuffer:uvw_buf    offset:0 atIndex:3];
    [enc setBuffer:nodes_buf_ offset:0 atIndex:4];
    [enc setBuffer:tris_buf_  offset:0 atIndex:5];
    [enc setBytes:&n_elements length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&ret_uvw    length:sizeof(uint32_t) atIndex:7];

    NSUInteger tw = pso.threadExecutionWidth;
    [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Reshape outputs back to original prefix shape
    distances = distances.view(prefix);
    face_id = face_id.view(prefix);
    if (return_uvw) {
        auto uvw_shape = prefix;
        uvw_shape.push_back(3);
        uvw = uvw.view(uvw_shape);
    } else {
        uvw = torch::Tensor();  // None
    }

    return {distances, face_id, uvw};
}

// ─── Signed distance ────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MtlBVHImpl::signed_distance(const torch::Tensor& positions, bool return_uvw, int mode) {
    auto pos_f = positions.detach().contiguous().to(torch::kFloat32);
    auto prefix = pos_f.sizes().vec();
    prefix.pop_back();
    int64_t N = pos_f.numel() / 3;

    pos_f = pos_f.view({N, 3});

    auto distances = torch::empty({N}, torch::kFloat32);
    auto face_id = torch::empty({N}, torch::kInt64);
    auto uvw = return_uvw ? torch::empty({N, 3}, torch::kFloat32) : torch::zeros({1, 3}, torch::kFloat32);

    uint32_t n_elements = (uint32_t)N;
    uint32_t ret_uvw = return_uvw ? 1 : 0;

    auto pos_buf = tensor_to_buffer(device_, pos_f);
    auto dist_buf = tensor_to_buffer(device_, distances);
    auto fid_buf = tensor_to_buffer(device_, face_id);
    auto uvw_buf = tensor_to_buffer(device_, uvw);

    auto pso = (mode == 0) ? pso_signed_watertight_ : pso_signed_raystab_;
    id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:pos_buf    offset:0 atIndex:0];
    [enc setBuffer:dist_buf   offset:0 atIndex:1];
    [enc setBuffer:fid_buf    offset:0 atIndex:2];
    [enc setBuffer:uvw_buf    offset:0 atIndex:3];
    [enc setBuffer:nodes_buf_ offset:0 atIndex:4];
    [enc setBuffer:tris_buf_  offset:0 atIndex:5];
    [enc setBytes:&n_elements length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&ret_uvw    length:sizeof(uint32_t) atIndex:7];

    NSUInteger tw = pso.threadExecutionWidth;
    [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    distances = distances.view(prefix);
    face_id = face_id.view(prefix);
    if (return_uvw) {
        auto uvw_shape = prefix;
        uvw_shape.push_back(3);
        uvw = uvw.view(uvw_shape);
    } else {
        uvw = torch::Tensor();
    }

    return {distances, face_id, uvw};
}

// ─── Ray trace ──────────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MtlBVHImpl::ray_trace(const torch::Tensor& rays_o, const torch::Tensor& rays_d) {
    auto ro_f = rays_o.detach().contiguous().to(torch::kFloat32);
    auto rd_f = rays_d.detach().contiguous().to(torch::kFloat32);

    auto prefix = ro_f.sizes().vec();
    prefix.pop_back();
    int64_t N = ro_f.numel() / 3;

    ro_f = ro_f.view({N, 3});
    rd_f = rd_f.view({N, 3});

    auto positions = torch::empty({N, 3}, torch::kFloat32);
    auto face_id = torch::empty({N}, torch::kInt64);
    auto depth = torch::empty({N}, torch::kFloat32);

    uint32_t n_elements = (uint32_t)N;

    auto ro_buf = tensor_to_buffer(device_, ro_f);
    auto rd_buf = tensor_to_buffer(device_, rd_f);
    auto pos_buf = tensor_to_buffer(device_, positions);
    auto fid_buf = tensor_to_buffer(device_, face_id);
    auto dep_buf = tensor_to_buffer(device_, depth);

    auto pso = pso_raytrace_;
    id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:ro_buf     offset:0 atIndex:0];
    [enc setBuffer:rd_buf     offset:0 atIndex:1];
    [enc setBuffer:pos_buf    offset:0 atIndex:2];
    [enc setBuffer:fid_buf    offset:0 atIndex:3];
    [enc setBuffer:dep_buf    offset:0 atIndex:4];
    [enc setBuffer:nodes_buf_ offset:0 atIndex:5];
    [enc setBuffer:tris_buf_  offset:0 atIndex:6];
    [enc setBytes:&n_elements length:sizeof(uint32_t) atIndex:7];

    NSUInteger tw = pso.threadExecutionWidth;
    [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    auto pos_shape = prefix;
    pos_shape.push_back(3);
    positions = positions.view(pos_shape);
    face_id = face_id.view(prefix);
    depth = depth.view(prefix);

    return {positions, face_id, depth};
}
