// gamma-rt-engine -- sprint 7.5.6.a part 2e-1
//
// Editor-driven scene kernel. Inputs:
//   texture(0) -- output RGBA8
//   buffer(0)  -- primitive AS (BLAS containing every scene mesh as
//                 one geometry; geometry_id maps 1:1 to mesh index)
//   buffer(1)  -- CameraUniform (precomputed pinhole basis + fov)
//   buffer(2)  -- per-mesh color buffer (float4 per geometry, .xyz)
//
// Per-pixel: build a ray in world space from the camera basis,
// intersect against the AS via the hardware RT cores on Apple9+ (M3,
// M4) or the shader-emulated path on older Apple silicon, then output
// either the hit mesh's color or the clear-sky color.

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

kernel void rt_scene(
    texture2d<float, access::write> outTex     [[texture(0)]],
    primitive_acceleration_structure accel     [[buffer(0)]],
    constant CameraUniform& camera             [[buffer(1)]],
    constant float4* colors                    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint w = outTex.get_width();
    const uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Pixel UV in [0, 1].
    const float u = (float(gid.x) + 0.5) / float(w);
    const float v = (float(gid.y) + 0.5) / float(h);

    // Screen-space coordinates on the image plane at distance 1.
    // tan(fov/2) is the half-height; multiply by aspect for half-width.
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

    // intersector<triangle_data> -- this AS holds only triangles and
    // we want triangle-specific hit data (barycentrics, primitive ID,
    // geometry ID). assume_geometry_type lets the compiler skip
    // polymorphic dispatch for a slight perf win.
    intersector<triangle_data> isr;
    isr.assume_geometry_type(geometry_type::triangle);
    intersection_result<triangle_data> hit = isr.intersect(r, accel);

    float3 color;
    if (hit.type == intersection_type::triangle) {
        // colors[geometry_id] is the per-mesh flat color (option a).
        // geometry_id is the index of the mesh among the AS's
        // geometry descriptors, in the order they were appended on
        // the CPU side -- matches mesh index in the Scene struct.
        color = colors[hit.geometry_id].xyz;
    } else {
        // Sky -- dark navy, same as the raster path's default clear.
        color = float3(0.05, 0.06, 0.10);
    }

    outTex.write(float4(color, 1.0), gid);
}
