// gamma-rt-engine -- sprint 7.5.6.a part 2b
//
// Hello-triangle test kernel. Per-pixel Möller-Trumbore ray-triangle
// intersection against a hard-coded triangle in the XY plane, with
// a fixed camera at (0, 0, 2) looking toward -Z. Output goes into
// a single output texture; hit pixels are colored by barycentric
// coordinates (one corner per color channel), miss pixels are a
// dark blue-grey sky.
//
// Part 2c replaces the inline triangle test with
// metal_raytracing intersector{} + a real MTLAccelerationStructure
// build, scaling beyond one primitive. For this push, the goal is
// just to prove the Metal compute pipeline + texture readback
// works on the dev M4.

#include <metal_stdlib>
using namespace metal;

// Returns (t, u, v) for a ray hit, or t<0 for miss.
// t is the ray-distance to the hit; (u, v) are the barycentric
// coordinates of the hit point on the triangle.
inline float3 intersect_triangle(
    float3 ro, float3 rd,
    float3 v0, float3 v1, float3 v2
) {
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;
    const float3 h = cross(rd, e2);
    const float a = dot(e1, h);
    if (abs(a) < 1e-7) return float3(-1.0);
    const float f = 1.0 / a;
    const float3 s = ro - v0;
    const float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return float3(-1.0);
    const float3 q = cross(s, e1);
    const float v = f * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) return float3(-1.0);
    const float t = f * dot(e2, q);
    if (t < 0.001) return float3(-1.0);
    return float3(t, u, v);
}

kernel void rt_triangle(
    texture2d<float, access::write> outTex [[texture(0)]],
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

    // Hard-coded triangle in the XY plane at z=0. Same vertices
    // as the editor-side DebugTriangle (sprint 7.5.3a) so visual
    // diffing against the raster Scene output is meaningful.
    const float3 v0 = float3(-0.866, -0.5, 0.0);
    const float3 v1 = float3( 0.866, -0.5, 0.0);
    const float3 v2 = float3( 0.0,    1.0, 0.0);

    const float3 hit = intersect_triangle(ro, rd, v0, v1, v2);

    float3 color;
    if (hit.x >= 0.0) {
        // Barycentric: hit.y = u, hit.z = v, w = 1 - u - v.
        // Map (w, u, v) to RGB so each corner is a primary color
        // (matches editor-side DebugTriangle's vertex-color
        // gradient: bottom-left red, bottom-right green, top blue).
        const float bw = 1.0 - hit.y - hit.z;
        color = float3(bw, hit.y, hit.z);
    } else {
        color = float3(0.05, 0.06, 0.10);  // dark sky
    }

    outTex.write(float4(color, 1.0), gid);
}
