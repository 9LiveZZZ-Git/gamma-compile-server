// gamma-rt-engine -- sprint 7.5.6.c-3
//
// Hardware ray-traced scene with shadows + multi-type lights.
//
// Adds to c-2:
//   - Shadow rays. Every contributing light fires a cheap
//     "occlusion intersector" toward itself; if anything in the AS
//     blocks the path, the light contributes nothing for that
//     pixel. Cost: one extra ray per pixel per visible light, so
//     N*L rays per pixel total. The hardware RT cores eat this for
//     breakfast on an Apple M4.
//   - Point lights. Distance attenuation = (1 - dist/range)^2,
//     clamped. range from the editor's PointLight node.
//   - Spot lights. Same distance attenuation + smoothstep cone
//     falloff between inner_angle and outer_angle.
//
// Inputs (unchanged from c-2 except LightsUniform layout):
//   texture(0) -- output RGBA8
//   buffer(0)  -- primitive AS
//   buffer(1)  -- CameraUniform
//   buffer(2)  -- per-mesh MaterialUniform[]
//   buffer(3)  -- global vertex_normals (float4 per vertex)
//   buffer(4)  -- global vertex_indices (u32 per index)
//   buffer(5)  -- per-mesh GeomOffset[]
//   buffer(6)  -- LightsUniform (now with per-slot type tag)

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct CameraUniform {
    float4 eye;
    float4 right;
    float4 up;
    float4 forward;
    float4 misc;
};

struct MaterialUniform {
    float4 albedo;
    float4 params;     // .x = metallic, .y = roughness, .z = shininess, .w = ambient
    uint4  flags;
};

struct GeomOffset {
    uint vertex_offset;
    uint index_offset;
    uint is_indexed;
    uint pad;
};

// One light slot. Discriminator: flags.x = 0 (dir) / 1 (point) / 2 (spot).
struct LightSlot {
    float4 pos_or_dir;   // .xyz = pos (point/spot) or dir-to-light (dir)
    float4 color;        // .xyz = color, .w = intensity
    float4 spot_dir;     // .xyz = spot cone axis
    float4 range_cones;  // .x = range, .y = cos_inner, .z = cos_outer
    uint4  flags;        // .x = type
};

struct LightsUniform {
    float4    ambient;   // .xyz = color, .w = intensity
    uint4     meta;      // .x = light count
    LightSlot slots[4];
};

constant float PI = 3.14159265358979;
constant float SHADOW_BIAS = 0.001;

// ── BRDF helpers (Cook-Torrance) ─────────────────────────────────────

inline float ggx_distribution(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

inline float schlick_ggx_g1(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / (NdotX * (1.0 - k) + k);
}

inline float3 schlick_fresnel(float cos_theta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cos_theta), 5.0);
}

inline float3 hemisphere_ibl(float3 dir) {
    float t = saturate(0.5 + 0.5 * dir.y);
    float3 sky    = float3(0.55, 0.70, 0.95);
    float3 ground = float3(0.20, 0.18, 0.15);
    return mix(ground, sky, t);
}

// ── Light evaluation ─────────────────────────────────────────────────

// Resolve the per-light geometry: returns L (unit direction FROM
// hit point TO light), the distance to the light (used as
// shadow-ray max), and the geometric attenuation (1 for directional,
// distance + optional cone falloff for point/spot). attenuation = 0
// signals "this light contributes nothing here" -- caller should
// short-circuit before casting the shadow ray.
struct LightSample {
    float3 L;
    float  distance;
    float  attenuation;
};

inline LightSample resolve_light(constant LightSlot& slot, float3 hit_point) {
    LightSample s;
    const uint type = slot.flags.x;
    if (type == 0u) {
        // Directional
        s.L = slot.pos_or_dir.xyz;
        s.distance = 1e6;       // effectively infinite for shadow rays
        s.attenuation = 1.0;
    } else {
        // Point or Spot: light at a position
        float3 to_light = slot.pos_or_dir.xyz - hit_point;
        s.distance = length(to_light);
        s.L = (s.distance > 0.0) ? to_light / s.distance : float3(0, 1, 0);
        // Quadratic falloff to range -- matches the raster path's
        // attenuation curve. Range = 0 effectively means "no light".
        float range = slot.range_cones.x;
        float fade = saturate(1.0 - s.distance / max(range, 0.0001));
        s.attenuation = fade * fade;
        if (type == 2u) {
            // Spot cone falloff. spot_dir is "where the light is shining"
            // (the axis of the cone), so the angle to the surface point
            // is between -L and spot_dir.
            float cos_angle = dot(-s.L, slot.spot_dir.xyz);
            float spot = smoothstep(slot.range_cones.z, slot.range_cones.y, cos_angle);
            s.attenuation *= spot;
        }
    }
    return s;
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
        outTex.write(float4(0.05, 0.06, 0.10, 1.0), gid);
        return;
    }

    // Hit point in world space + smooth normal.
    const float3 hit_point = r.origin + r.direction * hit.distance;
    const uint gid_hit = hit.geometry_id;
    const uint prim    = hit.primitive_id;
    const GeomOffset offs = geom_offsets[gid_hit];

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
    const float3 n0 = vertex_normals[i0].xyz;
    const float3 n1 = vertex_normals[i1].xyz;
    const float3 n2 = vertex_normals[i2].xyz;
    const float2 bary = hit.triangle_barycentric_coord;
    const float bw = 1.0 - bary.x - bary.y;
    float3 N = normalize(bw * n0 + bary.x * n1 + bary.y * n2);
    if (dot(N, rd) > 0.0) N = -N;

    const float3 V = -rd;
    const float n_dot_v = max(0.0, dot(N, V));

    const MaterialUniform mat = materials[gid_hit];
    const float3 albedo = mat.albedo.xyz;
    const uint   mtype  = mat.flags.x;

    // Shadow-ray intersector. accept_any_intersection lets the BVH
    // traversal early-out on the first hit; we don't care which
    // primitive shadowed us, just that something did.
    intersector<triangle_data> shadow_isr;
    shadow_isr.assume_geometry_type(geometry_type::triangle);
    shadow_isr.accept_any_intersection(true);

    // Slightly-biased shadow-ray origin to avoid self-shadowing
    // from the same triangle we just hit. Using the surface normal
    // (rather than the ray direction) means the bias also handles
    // grazing-angle ray hits correctly.
    const float3 shadow_origin = hit_point + N * SHADOW_BIAS;

    float3 color = float3(0.0);
    const uint nLights = lights.meta.x;

    if (mtype == 0u) {
        // Unlit: ignore lights entirely.
        color = albedo;
    } else if (mtype == 1u) {
        // Phong (Lambert + Blinn-Phong specular).
        const float shininess  = mat.params.z;
        const float ambient_mx = mat.params.w;

        float3 diffuse  = float3(0.0);
        float3 specular = float3(0.0);
        for (uint i = 0; i < nLights; i++) {
            const LightSample ls = resolve_light(lights.slots[i], hit_point);
            if (ls.attenuation <= 0.0) continue;
            const float n_dot_l = max(0.0, dot(N, ls.L));
            if (n_dot_l <= 0.0) continue;

            // Shadow test
            ray sr;
            sr.origin       = shadow_origin;
            sr.direction    = ls.L;
            sr.min_distance = 0.0;
            sr.max_distance = ls.distance - SHADOW_BIAS;
            auto sh = shadow_isr.intersect(sr, accel);
            if (sh.type == intersection_type::triangle) continue;

            const float3 lc = lights.slots[i].color.xyz
                            * lights.slots[i].color.w
                            * ls.attenuation;
            diffuse += lc * n_dot_l;
            const float3 H = normalize(ls.L + V);
            const float  spec = pow(max(0.0, dot(N, H)), max(shininess, 1.0));
            specular += lc * spec;
        }
        const float3 ambient = albedo * ambient_mx
                             + lights.ambient.xyz * lights.ambient.w * 0.2;
        color = albedo * diffuse + specular + ambient;
    } else {
        // PBR (Cook-Torrance).
        const float metallic  = mat.params.x;
        const float roughness = mat.params.y;
        const float3 F0 = mix(float3(0.04), albedo, metallic);

        float3 Lo = float3(0.0);
        for (uint i = 0; i < nLights; i++) {
            const LightSample ls = resolve_light(lights.slots[i], hit_point);
            if (ls.attenuation <= 0.0) continue;
            const float n_dot_l = max(0.0, dot(N, ls.L));
            if (n_dot_l <= 0.0) continue;

            ray sr;
            sr.origin       = shadow_origin;
            sr.direction    = ls.L;
            sr.min_distance = 0.0;
            sr.max_distance = ls.distance - SHADOW_BIAS;
            auto sh = shadow_isr.intersect(sr, accel);
            if (sh.type == intersection_type::triangle) continue;

            const float3 H = normalize(ls.L + V);
            const float n_dot_h = max(0.0, dot(N, H));
            const float v_dot_h = max(0.0, dot(V, H));

            const float D  = ggx_distribution(n_dot_h, roughness);
            const float Gv = schlick_ggx_g1(max(n_dot_v, 1e-4), roughness);
            const float Gl = schlick_ggx_g1(n_dot_l, roughness);
            const float G  = Gv * Gl;
            const float3 F = schlick_fresnel(v_dot_h, F0);

            const float3 spec = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
            const float3 kS = F;
            const float3 kD = (1.0 - kS) * (1.0 - metallic);

            const float3 lc = lights.slots[i].color.xyz
                            * lights.slots[i].color.w
                            * ls.attenuation;
            Lo += (kD * albedo / PI + spec) * lc * n_dot_l;
        }

        // Hemisphere-IBL ambient -- not shadow-tested for c-3
        // (would require a hemisphere sample fan; saves for the
        // path-tracing sprint 7.5.6.e). Surfaces in shadow still
        // get the IBL contribution, which "looks right" visually.
        const float3 R = reflect(-V, N);
        const float3 sky_refl = hemisphere_ibl(R);
        const float3 sky_diff = hemisphere_ibl(N);
        const float3 amb_F  = schlick_fresnel(n_dot_v, F0);
        const float3 amb_kS = amb_F * (1.0 - roughness);
        const float3 amb_kD = (1.0 - amb_F) * (1.0 - metallic);
        const float3 ibl = amb_kD * albedo * sky_diff + amb_kS * sky_refl;

        color = Lo + ibl * lights.ambient.w;
    }

    outTex.write(float4(color, 1.0), gid);
}
