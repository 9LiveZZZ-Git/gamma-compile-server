// gamma-rt-engine -- sprint 7.5.6.d (refraction + reflection)
//
// Bounce-loop kernel. Mirrors reflect, glass refracts, opaques shade.
// Up to MAX_BOUNCES (=6) ray bounces per pixel. Each bounce intersects
// the AS once via the hardware RT cores on Apple9+ silicon.
//
// Material dispatch (MaterialUniform.flags.x):
//   0  Unlit   -- just output albedo; terminate path
//   1  Phong   -- Lambert + Blinn-Phong; terminate path
//   2  PBR     -- Cook-Torrance + hemisphere IBL; terminate path
//   3  Mirror  -- perfect reflection ray, multiplicative tint, continue
//   4  Glass   -- refraction via Snell's law (with TIR fallback to
//                 reflection), multiplicative absorption tint, continue
//
// Inputs (unchanged from c-3):
//   texture(0)/buffer(0..6) as before. LightsUniform unchanged.

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
    float4 albedo;     // .xyz = base color / mirror tint / glass absorption
    float4 params;     // type-dependent: see kernel
    uint4  flags;      // .x = type (0=Unlit, 1=Phong, 2=PBR, 3=Mirror, 4=Glass)
};

struct GeomOffset {
    uint vertex_offset;
    uint index_offset;
    uint is_indexed;
    uint pad;
};

struct LightSlot {
    float4 pos_or_dir;
    float4 color;
    float4 spot_dir;
    float4 range_cones;
    uint4  flags;
};

struct LightsUniform {
    float4    ambient;
    uint4     meta;
    LightSlot slots[4];
};

constant float PI = 3.14159265358979;
constant float SHADOW_BIAS = 0.001;
constant float BOUNCE_BIAS = 0.001;
constant int   MAX_BOUNCES = 6;
constant float3 SKY_COLOR = float3(0.05, 0.06, 0.10);

// ── BRDF helpers ─────────────────────────────────────────────────────

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
    return mix(float3(0.20, 0.18, 0.15), float3(0.55, 0.70, 0.95), t);
}

// ── Light evaluation ─────────────────────────────────────────────────

struct LightSample {
    float3 L;
    float  distance;
    float  attenuation;
};

inline LightSample resolve_light(constant LightSlot& slot, float3 hit_point) {
    LightSample s;
    const uint type = slot.flags.x;
    if (type == 0u) {
        s.L = slot.pos_or_dir.xyz;
        s.distance = 1e6;
        s.attenuation = 1.0;
    } else {
        float3 to_light = slot.pos_or_dir.xyz - hit_point;
        s.distance = length(to_light);
        s.L = (s.distance > 0.0) ? to_light / s.distance : float3(0, 1, 0);
        float range = slot.range_cones.x;
        float fade = saturate(1.0 - s.distance / max(range, 0.0001));
        s.attenuation = fade * fade;
        if (type == 2u) {
            float cos_angle = dot(-s.L, slot.spot_dir.xyz);
            float spot = smoothstep(slot.range_cones.z, slot.range_cones.y, cos_angle);
            s.attenuation *= spot;
        }
    }
    return s;
}

// Shade an opaque surface (Unlit / Phong / PBR). Returns the
// outgoing radiance toward the view direction. Internally fires
// shadow rays through the same AS via the supplied shadow intersector.
inline float3 shade_opaque(
    uint mtype,
    float3 albedo, float metallic, float roughness, float shininess, float ambient_mx,
    float3 N, float3 V, float n_dot_v,
    float3 hit_point,
    primitive_acceleration_structure accel,
    thread intersector<triangle_data>& shadow_isr,
    constant LightsUniform& lights
) {
    if (mtype == 0u) return albedo;

    const float3 shadow_origin = hit_point + N * SHADOW_BIAS;
    const uint nLights = lights.meta.x;
    float3 color = float3(0.0);

    if (mtype == 1u) {
        // Phong
        float3 diffuse = float3(0.0), specular = float3(0.0);
        for (uint i = 0; i < nLights; i++) {
            const LightSample ls = resolve_light(lights.slots[i], hit_point);
            if (ls.attenuation <= 0.0) continue;
            const float n_dot_l = max(0.0, dot(N, ls.L));
            if (n_dot_l <= 0.0) continue;
            ray sr; sr.origin = shadow_origin; sr.direction = ls.L;
            sr.min_distance = 0.0; sr.max_distance = ls.distance - SHADOW_BIAS;
            if (shadow_isr.intersect(sr, accel).type == intersection_type::triangle) continue;
            const float3 lc = lights.slots[i].color.xyz * lights.slots[i].color.w * ls.attenuation;
            diffuse += lc * n_dot_l;
            const float3 H = normalize(ls.L + V);
            specular += lc * pow(max(0.0, dot(N, H)), max(shininess, 1.0));
        }
        const float3 ambient = albedo * ambient_mx
                             + lights.ambient.xyz * lights.ambient.w * 0.2;
        color = albedo * diffuse + specular + ambient;
    } else {
        // PBR
        const float3 F0 = mix(float3(0.04), albedo, metallic);
        float3 Lo = float3(0.0);
        for (uint i = 0; i < nLights; i++) {
            const LightSample ls = resolve_light(lights.slots[i], hit_point);
            if (ls.attenuation <= 0.0) continue;
            const float n_dot_l = max(0.0, dot(N, ls.L));
            if (n_dot_l <= 0.0) continue;
            ray sr; sr.origin = shadow_origin; sr.direction = ls.L;
            sr.min_distance = 0.0; sr.max_distance = ls.distance - SHADOW_BIAS;
            if (shadow_isr.intersect(sr, accel).type == intersection_type::triangle) continue;
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
            const float3 lc = lights.slots[i].color.xyz * lights.slots[i].color.w * ls.attenuation;
            Lo += (kD * albedo / PI + spec) * lc * n_dot_l;
        }
        const float3 R = reflect(-V, N);
        const float3 sky_refl = hemisphere_ibl(R);
        const float3 sky_diff = hemisphere_ibl(N);
        const float3 amb_F  = schlick_fresnel(n_dot_v, F0);
        const float3 amb_kS = amb_F * (1.0 - roughness);
        const float3 amb_kD = (1.0 - amb_F) * (1.0 - metallic);
        const float3 ibl = amb_kD * albedo * sky_diff + amb_kS * sky_refl;
        color = Lo + ibl * lights.ambient.w;
    }
    return color;
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

    // Initial pixel ray.
    const float u = (float(gid.x) + 0.5) / float(w);
    const float v = (float(gid.y) + 0.5) / float(h);
    const float screen_x = (2.0 * u - 1.0) * camera.misc.x * camera.misc.y;
    const float screen_y = (1.0 - 2.0 * v) * camera.misc.x;
    ray r;
    r.origin       = camera.eye.xyz;
    r.direction    = normalize(
        screen_x * camera.right.xyz
      + screen_y * camera.up.xyz
      + camera.forward.xyz
    );
    r.min_distance = 0.001;
    r.max_distance = 1000.0;

    intersector<triangle_data> isr;
    isr.assume_geometry_type(geometry_type::triangle);
    intersector<triangle_data> shadow_isr;
    shadow_isr.assume_geometry_type(geometry_type::triangle);
    shadow_isr.accept_any_intersection(true);

    float3 throughput = float3(1.0);
    float3 accum      = float3(0.0);
    bool   terminated = false;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        intersection_result<triangle_data> hit = isr.intersect(r, accel);
        if (hit.type != intersection_type::triangle) {
            accum += throughput * SKY_COLOR;
            terminated = true;
            break;
        }

        // Hit geometry + smooth normal.
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
        float3 N_geom = normalize(bw * n0 + bary.x * n1 + bary.y * n2);
        // For Glass: we need to know which side of the surface we hit
        // (entering vs. exiting the medium). dot(N_geom, rd) < 0 means
        // we're hitting from the front (entering); > 0 means back-face
        // (exiting). For Mirror / Phong / PBR, flip to face the viewer.
        const bool entering = dot(N_geom, r.direction) < 0.0;
        float3 N = entering ? N_geom : -N_geom;

        const MaterialUniform mat = materials[gid_hit];
        const uint mtype = mat.flags.x;

        if (mtype == 3u) {
            // ── Mirror ────────────────────────────────────────────
            throughput *= mat.albedo.xyz; // tint
            r.origin    = hit_point + N * BOUNCE_BIAS;
            r.direction = reflect(r.direction, N);
            r.min_distance = 0.0;
            r.max_distance = 1000.0;
            continue;
        }
        if (mtype == 4u) {
            // ── Glass ─────────────────────────────────────────────
            const float ior = max(mat.params.x, 1.001);
            const float ior_ratio = entering ? (1.0 / ior) : ior;
            float3 t = refract(r.direction, N, ior_ratio);
            if (dot(t, t) < 1e-4) {
                // Total internal reflection.
                r.origin    = hit_point + N * BOUNCE_BIAS;
                r.direction = reflect(r.direction, N);
            } else {
                // Refraction. Origin biased on the "outgoing" side of
                // the surface so the next intersect doesn't re-hit
                // this triangle from the inside.
                r.origin    = hit_point - N * BOUNCE_BIAS;
                r.direction = t;
            }
            r.min_distance = 0.0;
            r.max_distance = 1000.0;
            // Absorption: tint each bounce. A future c-d.2 could
            // Beer-Lambert this by distance traveled in the medium.
            throughput *= mat.albedo.xyz;
            continue;
        }

        // ── Opaque (Unlit / Phong / PBR) ─────────────────────────
        const float3 V = -r.direction;
        const float n_dot_v = max(0.0, dot(N, V));
        const float3 albedo    = mat.albedo.xyz;
        const float metallic   = mat.params.x;
        const float roughness  = mat.params.y;
        const float shininess  = mat.params.z;
        const float ambient_mx = mat.params.w;

        const float3 shade = shade_opaque(
            mtype, albedo, metallic, roughness, shininess, ambient_mx,
            N, V, n_dot_v, hit_point, accel, shadow_isr, lights
        );
        accum += throughput * shade;
        terminated = true;
        break;
    }

    if (!terminated) {
        // Bounce budget exhausted -- assume the ray escapes to sky.
        // Prevents the pixel from going pure black on a long mirror
        // hall of mirrors / inside-glass scenario.
        accum += throughput * SKY_COLOR;
    }

    outTex.write(float4(accum, 1.0), gid);
}
