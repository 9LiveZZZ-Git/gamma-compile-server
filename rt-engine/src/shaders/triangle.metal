// gamma-rt-engine -- sprint 7.5.6.a part 2c
//
// Hardware-accelerated ray-triangle test kernel. Same scene as part
// 2b (one triangle in the XY plane, pinhole camera at z=2), but the
// per-pixel intersection is now done by Metal's intersector<> running
// against a real MTLAccelerationStructure. On Apple9+ GPUs (M3, M4)
// this lowers to the dedicated hardware RT cores; on older Apple
// silicon (M1, M2) the same source compiles to a software emulation
// path on the regular shader cores.
//
// The acceleration structure is built once at MetalRenderer construct
// time (see metal_path.rs) and bound at buffer(0). The triangle
// vertices live in a vertex buffer owned by the AS build, so the
// kernel only needs the AS handle -- no per-vertex MSL constants.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

kernel void rt_triangle(
    texture2d<float, access::write> outTex [[texture(0)]],
    primitive_acceleration_structure accel [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint w = outTex.get_width();
    const uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Screen-space UV in [0, 1].
    const float u = (float(gid.x) + 0.5) / float(w);
    const float v = (float(gid.y) + 0.5) / float(h);

    // Pinhole camera. Eye at (0, 0, 2), looking toward -Z.
    // Image plane at z=1, height 2, width 2*aspect.
    const float aspect = float(w) / float(h);
    const float3 ro = float3(0.0, 0.0, 2.0);
    const float3 rd = normalize(float3(
        (u - 0.5) * 2.0 * aspect,
        (0.5 - v) * 2.0,        // flip Y so screen-up = world-up
        -1.0
    ));

    // Hardware RT path: build a ray, hand it to the intersector,
    // let the GPU's RT cores (or software emulator on older chips)
    // traverse the BVH and return the closest hit -- or a miss.
    ray r;
    r.origin       = ro;
    r.direction    = rd;
    r.min_distance = 0.001;
    r.max_distance = 1000.0;

    // intersector<triangle_data> means "this AS only has triangles
    // and I want triangle-specific hit data (barycentrics, primitive
    // ID)" -- lets the compiler skip the polymorphic dispatch.
    intersector<triangle_data> isr;
    isr.assume_geometry_type(geometry_type::triangle);

    intersection_result<triangle_data> hit = isr.intersect(r, accel);

    float3 color;
    if (hit.type == intersection_type::triangle) {
        // Barycentric: hit.triangle_barycentric_coord is (u, v) where
        // the third coord is 1 - u - v. Map (1-u-v, u, v) to RGB so
        // the three triangle corners get the three primary colors --
        // matches editor-side DebugTriangle's vertex-color gradient.
        const float2 bary = hit.triangle_barycentric_coord;
        const float bw = 1.0 - bary.x - bary.y;
        color = float3(bw, bary.x, bary.y);
    } else {
        color = float3(0.05, 0.06, 0.10);  // dark sky
    }

    outTex.write(float4(color, 1.0), gid);
}
