// gamma-rt-engine -- sprint 7.5.6.c-1
//
// Editor-driven scene kernel with Lambert shading.
//
// Inputs:
//   texture(0) -- output RGBA8
//   buffer(0)  -- primitive AS (each editor mesh is one geometry;
//                 geometry_id maps 1:1 to scene.meshes[i])
//   buffer(1)  -- CameraUniform (precomputed pinhole basis + fov)
//   buffer(2)  -- per-mesh flat color (float4 per geometry, .xyz)
//   buffer(3)  -- global per-triangle flat normal (float4, .xyz)
//   buffer(4)  -- per-geometry GeomOffset (gives first-triangle index
//                 into buffer(3) for this geometry)
//   buffer(5)  -- LightsUniform (ambient + up to 4 directional)
//
// Per-pixel: build a ray in world space from the camera basis,
// intersect against the AS via hardware RT cores, fetch the hit
// mesh's color + the hit triangle's flat normal, evaluate Lambert
// against every active directional light, add ambient, write.
//
// c-1 limitations (lifted in c-2):
//   - Flat per-triangle normals (smooth/per-vertex normals = c-2)
//   - Lambert only (Cook-Torrance + specular = c-2)
//   - No shadow rays (c-3)
//   - Point / Spot / Area lights from the editor are ignored

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct CameraUniform {
    float4 eye;       // .xyz = world-space eye
    float4 right;     // .xyz = right basis
    float4 up;        // .xyz = up basis (orthogonalized)
    float4 forward;   // .xyz = forward (look) direction
    float4 misc;      // .x = tan(fov/2), .y = aspect, .zw unused
};

struct GeomOffset {
    // 4 scalar uints (not uint + uint3 -- uint3 aligns to 16 which
    // would bump the struct to 32 bytes total, mismatching the
    // contiguous 16-byte [u32; 4] on the Rust side).
    uint triangle_offset;  // index into flat_normals[] for this geom's first triangle
    uint pad0;
    uint pad1;
    uint pad2;
};

struct LightsUniform {
    float4 ambient;     // .xyz = ambient color, .w = intensity
    uint4  meta;        // .x = number of directional lights (0..=4)
    float4 dirs[4];     // .xyz = unit direction FROM surface TOWARD light
    float4 colors[4];   // .xyz = color, .w = intensity
};

kernel void rt_scene(
    texture2d<float, access::write> outTex          [[texture(0)]],
    primitive_acceleration_structure accel          [[buffer(0)]],
    constant CameraUniform&  camera                 [[buffer(1)]],
    constant float4*         mesh_colors            [[buffer(2)]],
    constant float4*         flat_normals           [[buffer(3)]],
    constant GeomOffset*     geom_offsets           [[buffer(4)]],
    constant LightsUniform&  lights                 [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint w = outTex.get_width();
    const uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Pixel UV → world-space ray direction via precomputed basis.
    const float u = (float(gid.x) + 0.5) / float(w);
    const float v = (float(gid.y) + 0.5) / float(h);
    const float screen_x = (2.0 * u - 1.0) * camera.misc.x * camera.misc.y;
    const float screen_y = (1.0 - 2.0 * v) * camera.misc.x;
    const float3 rd = normalize(
        screen_x * camera.right.xyz
      + screen_y * camera.up.xyz
      + camera.forward.xyz
    );

    ray r;
    r.origin       = camera.eye.xyz;
    r.direction    = rd;
    r.min_distance = 0.001;
    r.max_distance = 1000.0;

    intersector<triangle_data> isr;
    isr.assume_geometry_type(geometry_type::triangle);
    intersection_result<triangle_data> hit = isr.intersect(r, accel);

    float3 color;
    if (hit.type == intersection_type::triangle) {
        // Fetch material color (per-mesh) + flat normal (per-triangle).
        const uint gid_hit  = hit.geometry_id;
        const uint prim_id  = hit.primitive_id;
        const float3 albedo = mesh_colors[gid_hit].xyz;
        const uint nrm_idx  = geom_offsets[gid_hit].triangle_offset + prim_id;
        float3 N = flat_normals[nrm_idx].xyz;

        // Flip the normal toward the camera if the hit is on the back
        // face. Per-triangle flat normals from a CCW mesh point in
        // one stable direction; double-sided shading is just easier
        // than tracking winding in the editor's mesh builders.
        if (dot(N, rd) > 0.0) N = -N;

        // Lambert: sum over the active directional lights.
        float3 diffuse = float3(0.0);
        const uint nLights = lights.meta.x;
        for (uint i = 0; i < nLights; i++) {
            const float3 L = lights.dirs[i].xyz;       // already normalized + flipped on CPU
            const float n_dot_l = max(0.0, dot(N, L));
            diffuse += lights.colors[i].xyz * lights.colors[i].w * n_dot_l;
        }

        // Ambient floor keeps shadowed areas from going pure black.
        // c-3 will replace this with traced ambient occlusion via
        // a one-bounce hemisphere sample. Until then, a flat ambient
        // is the cheap approximation everyone uses.
        const float3 ambient = lights.ambient.xyz * lights.ambient.w;

        color = albedo * (diffuse + ambient);
    } else {
        // Sky -- dark navy, same as the raster path's default clear.
        color = float3(0.05, 0.06, 0.10);
    }

    outTex.write(float4(color, 1.0), gid);
}
