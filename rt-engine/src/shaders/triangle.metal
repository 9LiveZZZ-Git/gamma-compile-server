// gamma-rt-engine -- sprint 7.5.6.c-2
//
// Editor-driven scene kernel with smooth normals + material-aware
// shading. Three BRDFs reachable from the editor's material nodes:
//
//   Unlit (UnlitMat)     -- output albedo, ignore lighting
//   Phong (PhongMat)     -- Lambert + Blinn-Phong specular
//   PBR   (PhysicalMat)  -- Cook-Torrance GGX + Schlick Fresnel +
//                           Schlick-GGX geometry + hemisphere-IBL
//                           ambient. Same BRDF as the raster path.
//
// Inputs:
//   texture(0) -- output RGBA8
//   buffer(0)  -- primitive AS
//   buffer(1)  -- CameraUniform
//   buffer(2)  -- per-mesh MaterialUniform[]
//   buffer(3)  -- global vertex_normals (float4 per vertex)
//   buffer(4)  -- global vertex_indices (u32 per index)
//   buffer(5)  -- per-mesh GeomOffset[]
//   buffer(6)  -- LightsUniform
//
// c-2 limitations (lifted later):
//   - Directional lights only. Point + Spot evaluated as no-ops.
//     Real point/spot in c-3 alongside shadow rays.
//   - IBL ambient is a procedural hemisphere fake, not a real env
//     probe. Full HDRI sampling lands in Phase 7 §5.4.
//   - No shadows. Lit side never gets occluded by other geometry.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct CameraUniform {
    float4 eye;
    float4 right;
    float4 up;
    float4 forward;
    float4 misc;        // .x = tan(fov/2), .y = aspect
};

struct MaterialUniform {
    float4 albedo;      // .xyz = base color, .w = vertex_mix (unused in c-2)
    float4 params;      // .x = metallic, .y = roughness, .z = shininess, .w = ambient
    uint4  flags;       // .x = type (0=Unlit, 1=Phong, 2=PBR)
};

struct GeomOffset {
    uint vertex_offset;
    uint index_offset;
    uint is_indexed;
    uint pad;
};

struct LightsUniform {
    float4 ambient;     // .xyz = ambient color, .w = intensity
    uint4  meta;        // .x = number of directional lights (0..=4)
    float4 dirs[4];     // .xyz = direction TO light
    float4 colors[4];   // .xyz = color, .w = intensity
};

constant float PI = 3.14159265358979;

// ── BRDF helpers (Cook-Torrance) ─────────────────────────────────────

// GGX / Trowbridge-Reitz normal distribution function.
inline float ggx_distribution(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry term (one direction).
inline float schlick_ggx_g1(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / (NdotX * (1.0 - k) + k);
}

// Schlick Fresnel approximation.
inline float3 schlick_fresnel(float cos_theta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// Cheap hemisphere IBL: gradient from a horizon ground tone up to a
// pale-blue sky based on reflection vector's Y. Same shape as the
// raster path's PBR ambient (which also uses an analytic hemisphere
// rather than an HDRI for now).
inline float3 hemisphere_ibl(float3 dir) {
    float t = saturate(0.5 + 0.5 * dir.y);
    float3 sky    = float3(0.55, 0.70, 0.95);
    float3 ground = float3(0.20, 0.18, 0.15);
    return mix(ground, sky, t);
}

// ── Main kernel ──────────────────────────────────────────────────────

kernel void rt_scene(
    texture2d<float, access::write> outTex          [[texture(0)]],
    primitive_acceleration_structure accel          [[buffer(0)]],
    constant CameraUniform&  camera                 [[buffer(1)]],
    constant MaterialUniform* materials             [[buffer(2)]],
    constant float4*          vertex_normals        [[buffer(3)]],
    constant uint*            vertex_indices        [[buffer(4)]],
    constant GeomOffset*      geom_offsets          [[buffer(5)]],
    constant LightsUniform&   lights                [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint w = outTex.get_width();
    const uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Pixel UV → world-space ray.
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

    if (hit.type != intersection_type::triangle) {
        // Sky -- dark navy.
        outTex.write(float4(0.05, 0.06, 0.10, 1.0), gid);
        return;
    }

    const uint gid_hit = hit.geometry_id;
    const uint prim    = hit.primitive_id;
    const GeomOffset offs = geom_offsets[gid_hit];

    // Fetch the 3 vertex IDs for the hit triangle. Indexed meshes
    // look up through vertex_indices; non-indexed use raw primitive*3.
    uint i0, i1, i2;
    if (offs.is_indexed != 0u) {
        const uint base = offs.index_offset + prim * 3u;
        i0 = offs.vertex_offset + vertex_indices[base + 0u];
        i1 = offs.vertex_offset + vertex_indices[base + 1u];
        i2 = offs.vertex_offset + vertex_indices[base + 2u];
    } else {
        i0 = offs.vertex_offset + prim * 3u + 0u;
        i1 = offs.vertex_offset + prim * 3u + 1u;
        i2 = offs.vertex_offset + prim * 3u + 2u;
    }

    // Smooth-shaded normal via barycentric interpolation of the
    // three vertex normals. This is the big visual upgrade from c-1
    // -- a sphere now reads as a smooth surface, not a stack of
    // discrete facets.
    const float3 n0 = vertex_normals[i0].xyz;
    const float3 n1 = vertex_normals[i1].xyz;
    const float3 n2 = vertex_normals[i2].xyz;
    const float2 bary = hit.triangle_barycentric_coord;
    const float bw = 1.0 - bary.x - bary.y;
    float3 N = normalize(bw * n0 + bary.x * n1 + bary.y * n2);

    // Double-sided shading: flip the normal toward the viewer if
    // the ray hit the back side (e.g. inside-out winding from an
    // editor mesh node). Cheaper than tracking winding on the
    // editor side; same trick c-1 used for flat normals.
    if (dot(N, rd) > 0.0) N = -N;

    const float3 V = -rd; // view direction (toward camera)
    const float n_dot_v = max(0.0, dot(N, V));

    const MaterialUniform mat = materials[gid_hit];
    const float3 albedo = mat.albedo.xyz;
    const uint   mtype  = mat.flags.x;

    float3 color = float3(0.0);

    if (mtype == 0u) {
        // ── Unlit ────────────────────────────────────────────────
        color = albedo;
    } else if (mtype == 1u) {
        // ── Phong (Lambert + Blinn-Phong specular) ───────────────
        const float shininess  = mat.params.z;
        const float ambient_mx = mat.params.w;

        float3 diffuse  = float3(0.0);
        float3 specular = float3(0.0);
        const uint nLights = lights.meta.x;
        for (uint i = 0; i < nLights; i++) {
            const float3 L = lights.dirs[i].xyz;
            const float n_dot_l = max(0.0, dot(N, L));
            if (n_dot_l <= 0.0) continue;
            const float3 lc = lights.colors[i].xyz * lights.colors[i].w;
            diffuse += lc * n_dot_l;
            // Blinn-Phong specular -- white highlight, raster path
            // matches; the editor's PhongMat is a dielectric-only model.
            const float3 H = normalize(L + V);
            const float  spec = pow(max(0.0, dot(N, H)), max(shininess, 1.0));
            specular += lc * spec;
        }
        // Ambient lifts shadowed areas. Skipping the IBL hemisphere
        // here -- Phong has its own ambient knob in the material.
        const float3 ambient = albedo * ambient_mx
                             + lights.ambient.xyz * lights.ambient.w * 0.2;
        color = albedo * diffuse + specular + ambient;
    } else {
        // ── PBR (Cook-Torrance) ──────────────────────────────────
        const float metallic  = mat.params.x;
        const float roughness = mat.params.y;

        // F0: dielectric uses 4% reflectance; metals use albedo
        // tinted specular. Standard Disney-style remap.
        const float3 F0 = mix(float3(0.04), albedo, metallic);

        float3 Lo = float3(0.0);
        const uint nLights = lights.meta.x;
        for (uint i = 0; i < nLights; i++) {
            const float3 L = lights.dirs[i].xyz;
            const float n_dot_l = max(0.0, dot(N, L));
            if (n_dot_l <= 0.0) continue;

            const float3 H = normalize(L + V);
            const float n_dot_h = max(0.0, dot(N, H));
            const float v_dot_h = max(0.0, dot(V, H));

            const float D  = ggx_distribution(n_dot_h, roughness);
            const float Gv = schlick_ggx_g1(max(n_dot_v, 1e-4), roughness);
            const float Gl = schlick_ggx_g1(n_dot_l, roughness);
            const float G  = Gv * Gl;
            const float3 F = schlick_fresnel(v_dot_h, F0);

            const float3 spec = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 1e-4);

            // Energy split: metals have no Lambertian diffuse.
            const float3 kS = F;
            const float3 kD = (1.0 - kS) * (1.0 - metallic);

            const float3 lc = lights.colors[i].xyz * lights.colors[i].w;
            Lo += (kD * albedo / PI + spec) * lc * n_dot_l;
        }

        // Hemisphere-IBL ambient -- cheap fake env that gives metals
        // somewhere to reflect even with no real probe. Sample the
        // reflection vector against the procedural sky/ground gradient.
        const float3 R = reflect(-V, N);
        const float3 sky_refl = hemisphere_ibl(R);
        const float3 sky_diff = hemisphere_ibl(N);

        const float3 amb_F  = schlick_fresnel(n_dot_v, F0);
        const float3 amb_kS = amb_F * (1.0 - roughness); // rough surfaces blur out
        const float3 amb_kD = (1.0 - amb_F) * (1.0 - metallic);

        const float3 ibl = amb_kD * albedo * sky_diff
                         + amb_kS * sky_refl;

        color = Lo + ibl * lights.ambient.w;
    }

    outTex.write(float4(color, 1.0), gid);
}
