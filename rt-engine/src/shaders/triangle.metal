// gamma-rt-engine -- sprint 7.5.6.e (path tracing + GI)
//
// Progressive path tracer with:
//   - Per-pixel sub-sample jittered primary rays (free AA)
//   - One bounce + indirect path sample per pixel per frame
//   - Cosine-weighted hemisphere sampling for diffuse indirect light
//   - Fresnel-weighted random reflect-or-refract at glass surfaces
//   - Persistent accumulation texture, output = accum / frame_count
//
// Convergence: static scenes refine over many frames as noise
// averages out (target ~64-256 frames for clean GI). Scenes with
// camera/light/material changes reset the accumulation (engine-side
// in update_* methods). Denoising lands in sprint f and lets the
// real-time path look clean at low frame counts.
//
// Buffers (unchanged from c-d except (7) added):
//   texture(0) RGBA8       -- output (display)
//   texture(1) RGBA32F     -- accumulation (persistent; private storage)
//   buffer(0)              -- primitive AS
//   buffer(1) CameraUniform
//   buffer(2) MaterialUniform[]
//   buffer(3) vertex_normals (float4[])
//   buffer(4) vertex_indices (u32[])
//   buffer(5) GeomOffset[]
//   buffer(6) LightsUniform
//   buffer(7) PathState      -- new; { frame_count, pad×3 }

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
    float4 params;
    uint4  flags;
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

struct PathState {
    uint     frame_count;
    uint     pad0;
    uint     pad1;
    uint     pad2;
    // f.3.b -- previous frame's view*projection matrix.
    float4x4 prev_view_proj;
    // f.3.d -- per-frame sub-pixel jitter (Halton(2,3) sequence
    // value in [0, 1]). Used as the SAME offset for every pixel in
    // this frame, so the kernel + MetalFX can agree on where the
    // ray hit within each pixel. Without this they disagree and
    // MetalFX blurs disparate sub-pixel content together.
    float4   jitter;
};

constant float PI = 3.14159265358979;
constant float SHADOW_BIAS = 0.001;
constant float BOUNCE_BIAS = 0.001;
constant int   MAX_BOUNCES = 4;
constant float NORMAL_FLIP_THR = 0.05;

// ── RNG (PCG hash + scrambler) ───────────────────────────────────────

inline uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (w >> 22u) ^ w;
}

inline uint init_rng(uint2 gid, uint frame, uint salt) {
    // Combine gid + frame + a per-bounce salt so different bounces
    // get decorrelated samples from the same pixel. PCG-hash twice
    // for diffusion.
    uint h = gid.x * 1973u + gid.y * 9277u + frame * 26699u + salt * 71093u;
    return pcg_hash(pcg_hash(h));
}

inline float rand01(thread uint& state) {
    state = pcg_hash(state);
    return float(state) * (1.0 / 4294967296.0);
}

// Cosine-weighted hemisphere sample aligned with normal N. Returns a
// unit-length world-space direction. Cosine weight means samples
// cluster toward the normal (peaks of Lambert N·L), which matches
// the BRDF's PDF and cancels out the cos / (1/pi) factors in the
// throughput multiplier (just multiply by albedo per bounce).
inline float3 cosine_hemisphere(thread uint& rng, float3 N) {
    const float u1 = rand01(rng);
    const float u2 = rand01(rng);
    const float r = sqrt(u1);
    const float theta = 2.0 * PI * u2;
    const float3 local_dir = float3(r * cos(theta), r * sin(theta), sqrt(max(0.0, 1.0 - u1)));
    // Orthonormal basis around N. Pick T not parallel to N.
    const float3 T0 = (abs(N.x) > 0.9) ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 B = normalize(cross(N, T0));
    const float3 T = cross(B, N);
    return local_dir.x * T + local_dir.y * B + local_dir.z * N;
}

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

inline float schlick_fresnel_scalar(float cos_theta, float F0) {
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

// Direct-only opaque shading: Lambert / Phong / PBR contribution from
// all lights at this hit point, with shadow rays. Returns the
// outgoing radiance toward V (the inverse ray direction). The path
// tracer also fires an indirect cosine-weighted bounce afterward
// for GI; this function handles just the *direct* term.
inline float3 shade_direct(
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
        // Phong direct.
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
        // Ambient is the analytic fallback for "everything other than
        // direct + indirect bounce" -- with GI accumulating from the
        // bounce path, we could shrink this, but it's still useful at
        // low frame counts for stable shadow-side color.
        const float3 ambient = albedo * ambient_mx
                             + lights.ambient.xyz * lights.ambient.w * 0.1;
        color = albedo * diffuse + specular + ambient;
    } else {
        // PBR direct.
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
        // Dim hemisphere-IBL ambient -- path tracing's indirect bounce
        // gives the "real" indirect light. The IBL term here is a
        // fallback that gets dialed down once we have proper GI.
        const float3 R = reflect(-V, N);
        const float3 sky_refl = hemisphere_ibl(R);
        const float3 sky_diff = hemisphere_ibl(N);
        const float3 amb_F  = schlick_fresnel(n_dot_v, F0);
        const float3 amb_kS = amb_F * (1.0 - roughness);
        const float3 amb_kD = (1.0 - amb_F) * (1.0 - metallic);
        const float3 ibl = amb_kD * albedo * sky_diff + amb_kS * sky_refl;
        color = Lo + ibl * lights.ambient.w * 0.5;
    }
    return color;
}

// ── Main path-tracing kernel ─────────────────────────────────────────
//
// Writes: accumTex (RGBA32F, sum of samples) and normalTex (RGBA32F,
// primary-hit world normal in .xyz, .w = 1 on hit / 0 on miss for the
// denoise kernel's edge-stopping). The display texture is left for
// rt_denoise to populate.

kernel void rt_scene(
    texture2d<float, access::read_write> accumTex   [[texture(1)]],
    texture2d<float, access::write>     normalTex   [[texture(2)]],
    // f.3.b G-buffers for MetalFX:
    texture2d<float, access::write>     depthTex    [[texture(3)]],
    texture2d<float, access::write>     motionTex   [[texture(4)]],
    texture2d<float, access::write>     albedoTex   [[texture(5)]],
    // f.3.c -- noisy single-sample color (RGBA16F). MetalFX's color
    // input. The kernel's `sample` value goes here un-divided; MetalFX
    // does its own temporal accumulation internally using motion
    // vectors. The engine-side accumulation (accumTex) is kept for
    // the spatial-denoiser fallback path.
    texture2d<float, access::write>     noisyColorTex [[texture(6)]],
    // f.3.d-fix4 -- roughness + specular-albedo G-buffers REQUIRED by
    // MTLFXTemporalDenoisedScaler. Without these the scaler doesn't
    // know which pixels need aggressive denoising vs which are sharp,
    // and temporal blending refuses to converge on static scenes.
    texture2d<float, access::write>     roughnessTex    [[texture(7)]],
    texture2d<float, access::write>     specAlbedoTex   [[texture(8)]],
    primitive_acceleration_structure    accel       [[buffer(0)]],
    constant CameraUniform&  camera                 [[buffer(1)]],
    constant MaterialUniform* materials             [[buffer(2)]],
    constant float4*          vertex_normals        [[buffer(3)]],
    constant uint*            vertex_indices        [[buffer(4)]],
    constant GeomOffset*      geom_offsets          [[buffer(5)]],
    constant LightsUniform&   lights                [[buffer(6)]],
    constant PathState&       path                  [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint w = accumTex.get_width();
    const uint h = accumTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    const uint frame = path.frame_count;
    thread uint rng  = init_rng(gid, frame, 0u);

    // Sub-pixel jittered primary ray. f.3.d -- one global offset
    // for the entire frame from a Halton(2, 3) sequence (path.jitter
    // .xy in [0, 1]); over many frames this covers the sub-pixel
    // grid uniformly. MetalFX gets the same value (centered to
    // [-0.5, 0.5]) so it can correlate sub-pixel samples across
    // frames. The kernel's RNG is still used for everything else
    // (random bounce direction, Fresnel-mixed glass branch).
    const float u = (float(gid.x) + path.jitter.x) / float(w);
    const float v = (float(gid.y) + path.jitter.y) / float(h);
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
    float3 sample     = float3(0.0);
    bool   terminated = false;

    // Sprint 7.5.6.f -- track the primary G-buffer values. Recorded
    // at the FIRST OPAQUE hit (mirror/glass surfaces don't write
    // their own G-buffer; whatever they ultimately reflect / refract
    // INTO is the "perceived" content for that pixel). This is
    // important for temporal denoising in f.3: motion vectors for
    // mirror reflections track the underlying scene, not the mirror.
    float3 primary_normal     = float3(0.0);
    float3 primary_hit_point  = float3(0.0);
    float3 primary_albedo     = float3(0.0);
    float  primary_depth      = 0.0;  // total path length to opaque hit
    // f.3.d-fix4 -- additional G-buffer values required by MetalFX's
    // denoised scaler. Roughness drives the denoiser's sharpness vs
    // smoothness decision per pixel; specular albedo (F0) drives the
    // separation between diffuse and specular paths. Recorded at the
    // primary OPAQUE hit, same as the rest.
    float  primary_roughness    = 1.0;  // fully rough default = treat as diffuse
    float3 primary_specular     = float3(0.04);  // dielectric F0 default
    bool   primary_recorded   = false;
    float  accumulated_dist   = 0.0;  // virtual depth through mirrors / glass

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        intersection_result<triangle_data> hit = isr.intersect(r, accel);
        if (hit.type != intersection_type::triangle) {
            sample += throughput * hemisphere_ibl(r.direction);
            terminated = true;
            break;
        }

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
        const float ndotrd = dot(N_geom, r.direction);
        const bool entering = ndotrd < NORMAL_FLIP_THR;
        float3 N = entering ? N_geom : -N_geom;

        // Accumulate the virtual distance from the camera along the
        // bounce path. For straight-through hits this is just the
        // primary ray distance; for mirror/glass refractions it's
        // the sum across all bounces ("virtual depth").
        accumulated_dist += hit.distance;

        const MaterialUniform mat = materials[gid_hit];
        const uint mtype = mat.flags.x;

        // Record the primary-hit G-buffer at the FIRST hit that's
        // "denoising-stable" -- meaning its world point and surface
        // properties don't change between frames for a given camera.
        // That includes:
        //   - opaque surfaces (mtypes 0, 1, 2): obviously stable
        //   - glass (mtype 4): the GLASS SURFACE itself is stable;
        //     it's the random Fresnel-branched path that flickers,
        //     and we want MetalFX to denoise that flickering color
        //     against a stable G-buffer.
        // Mirrors (mtype 3) are NOT denoising-stable at their own
        // surface (perfectly reflective, no own appearance) so we
        // keep falling through to whatever they reflect into.
        const bool is_opaque = (mtype == 0u || mtype == 1u || mtype == 2u);
        const bool is_glass  = (mtype == 4u);
        if (!primary_recorded && (is_opaque || is_glass)) {
            primary_normal    = N;
            primary_hit_point = hit_point;
            primary_depth     = accumulated_dist;
            if (is_glass) {
                // f.3.d-fix5 -- record G-buffer AT the glass surface
                // (not through it). Before this, glass pixels'
                // G-buffer flickered between whatever the reflect /
                // refract branch ended up hitting, and MetalFX's
                // history rejection killed every glass pixel forever.
                //
                // Mark glass as "smooth dielectric": diffuse albedo
                // mostly white (light passes through with little
                // chromatic shift for clear glass), roughness 0
                // (sharp specular response), F0 ≈ 0.04 (IOR ≈ 1.5
                // dielectric). The noisy COLOR signal still encodes
                // the Fresnel-mixed reflect/refract content; MetalFX
                // averages it temporally against this stable surface.
                primary_albedo    = float3(0.9);
                primary_roughness = 0.0;
                primary_specular  = float3(0.04);
            } else {
                primary_albedo    = mat.albedo.xyz;
                // f.3.d-fix4 -- mat.params layout: x=metallic,
                // y=roughness, z=shininess, w=ambient. For Unlit
                // (mtype 0) treat roughness as 1.0 (no specular
                // concentration) since Unlit has no glossiness.
                const float met = (mtype == 0u) ? 0.0 : mat.params.x;
                const float rgh = (mtype == 0u) ? 1.0 : mat.params.y;
                primary_roughness = rgh;
                // F0 (Fresnel at 0 deg): dielectric ~0.04, metallic
                // = albedo. WWDC25 calls out that for metallic
                // surfaces the COLORED component lives in specular
                // albedo and diffuse should be darker. We use the
                // standard lerp -- diffuse will be slightly bright
                // for metals but the denoiser tolerates it.
                primary_specular  = mix(float3(0.04), mat.albedo.xyz, met);
            }
            primary_recorded  = true;
        }

        if (mtype == 3u) {
            // Mirror: continue with reflected ray. No direct lighting
            // contribution at this surface (mirrors don't have a
            // diffuse term).
            throughput *= mat.albedo.xyz;
            r.origin    = hit_point + N * BOUNCE_BIAS;
            r.direction = reflect(r.direction, N);
            r.min_distance = 0.0;
            r.max_distance = 1000.0;
            continue;
        }
        if (mtype == 4u) {
            // Glass: Fresnel-weighted random branch (reflect or
            // refract). Path traces correct over many samples.
            const float ior = max(mat.params.x, 1.001);
            const float ior_ratio = entering ? (1.0 / ior) : ior;
            const float cos_inc = abs(dot(r.direction, N));
            // F0 for glass dielectric: ((n1-n2)/(n1+n2))^2
            const float f0_dielectric = (1.0 - ior) / (1.0 + ior);
            const float F0 = f0_dielectric * f0_dielectric;
            const float fresnel = schlick_fresnel_scalar(cos_inc, F0);

            float3 t = refract(r.direction, N, ior_ratio);
            const bool tir = dot(t, t) < 1e-4;
            const bool do_reflect = tir || (rand01(rng) < fresnel);

            if (do_reflect) {
                r.origin    = hit_point + N * BOUNCE_BIAS;
                r.direction = reflect(r.direction, N);
                // TIR is energy-preserving; refraction-sampled reflect
                // already counted by the Fresnel branch -- no extra
                // multiplier needed because the random pick is
                // unbiased in expectation.
            } else {
                r.origin    = hit_point - N * BOUNCE_BIAS;
                r.direction = t;
                // Absorption tint applies on refraction (light enters
                // / leaves the medium); not on a Fresnel-reflected
                // bounce (which never entered).
                throughput *= mat.albedo.xyz;
            }
            r.min_distance = 0.0;
            r.max_distance = 1000.0;
            continue;
        }

        // ── Opaque (Unlit / Phong / PBR) ──────────────────────────
        const float3 V = -r.direction;
        const float n_dot_v = max(0.0, dot(N, V));
        const float3 albedo    = mat.albedo.xyz;
        const float metallic   = mat.params.x;
        const float roughness  = mat.params.y;
        const float shininess  = mat.params.z;
        const float ambient_mx = mat.params.w;

        // Direct lighting at this hit.
        const float3 direct = shade_direct(
            mtype, albedo, metallic, roughness, shininess, ambient_mx,
            N, V, n_dot_v, hit_point, accel, shadow_isr, lights
        );
        sample += throughput * direct;

        // Indirect bounce (Lambertian sample), unless we've used up
        // the bounce budget. The throughput multiplier for a
        // cosine-weighted Lambert sample is just `albedo` (cos/pi PDF
        // cancels with the BRDF normalizer). For PBR materials,
        // approximating the indirect as diffuse-only is the c-e v1
        // simplification; the full energy-conserving PBR indirect
        // sample uses GGX importance sampling, which lands in c-e.2.
        if (bounce == MAX_BOUNCES - 1) {
            terminated = true;
            break;
        }
        // Unlit terminates -- no indirect from an emissive flat surface.
        if (mtype == 0u) {
            terminated = true;
            break;
        }
        float3 bounce_dir = cosine_hemisphere(rng, N);
        // Sanity: if cosine sample is degenerate (rare), terminate.
        if (dot(bounce_dir, N) <= 0.0) {
            terminated = true;
            break;
        }
        r.origin       = hit_point + N * BOUNCE_BIAS;
        r.direction    = bounce_dir;
        r.min_distance = 0.0;
        r.max_distance = 1000.0;
        throughput    *= albedo;
        // (PBR throughput should also account for metallic / specular
        // split here; c-e v1 approximates by using albedo directly,
        // which slightly under-bounces metals but stays unbiased for
        // dielectrics.)
    }

    if (!terminated) {
        sample += throughput * hemisphere_ibl(r.direction);
    }

    // ── Accumulate (kept for spatial-denoise fallback) ───────────────
    // First frame: write sample as initial value. Subsequent frames:
    // read prior accum, add, write back. RGBA32F so we don't lose
    // precision summing thousands of samples. MetalFX-path doesn't
    // need this -- it consumes the noisy single-sample texture below.
    float3 new_accum;
    if (frame == 0u) {
        new_accum = sample;
    } else {
        new_accum = accumTex.read(gid).rgb + sample;
    }
    accumTex.write(float4(new_accum, 0.0), gid);

    // f.3.c -- noisy single-sample color for MetalFX.
    // f.3.d -- firefly clamp before output. Glass / mirror paths
    // occasionally produce single super-bright samples (caustics,
    // edge-of-TIR cases). Without clamping, these stick around in
    // MetalFX's temporal blend for many frames and look like
    // sparkly artifacts. min() to a reasonable HDR ceiling keeps
    // the unbiased average correct in expectation while killing
    // the worst outliers.
    const float3 clamped_sample = min(sample, float3(8.0));
    noisyColorTex.write(float4(clamped_sample, 1.0), gid);

    // ── G-buffer writes ───────────────────────────────────────────
    // Normal: .xyz = world-space normal at primary opaque hit, .w =
    // 1 if recorded / 0 if this pixel saw only sky (or only
    // mirror/glass that escaped to sky).
    normalTex.write(
        float4(primary_normal, primary_recorded ? 1.0 : 0.0),
        gid
    );

    // f.3.b -- depth, motion vector, albedo for MetalFX.
    // Depth: linear "virtual depth" through any mirrors/glass to
    // the first opaque hit. 0 for sky pixels (could also use a
    // very-large sentinel; MetalFX seems happy with 0 for sky).
    depthTex.write(float4(primary_depth, 0.0, 0.0, 0.0), gid);

    // Albedo: raw surface color before lighting (= material's base
    // color). 0 for sky pixels.
    albedoTex.write(float4(primary_albedo, 1.0), gid);

    // f.3.d-fix4 -- roughness + specular albedo G-buffers for
    // MetalFX's denoised scaler. Sky / pure-glass pixels keep the
    // defaults (roughness=1, F0=0.04) which read as "fully diffuse
    // dielectric" -- correct for the sky envelope, conservative for
    // glass-through pixels.
    roughnessTex.write(float4(primary_roughness, 0.0, 0.0, 0.0), gid);
    specAlbedoTex.write(float4(primary_specular, 1.0), gid);

    // Motion vector: vector pointing FROM the current pixel TO where
    // this pixel's content was in the PREVIOUS frame, in UV space.
    // (Apple's convention: "vector that indicates where in the
    // previous frame the pixel had been" -> prev - current.)
    // Zero for sky pixels or the first frame after a reset.
    float2 motion = float2(0.0);
    if (primary_recorded && frame > 0u) {
        const float4 clip_prev = path.prev_view_proj
                               * float4(primary_hit_point, 1.0);
        if (abs(clip_prev.w) > 1e-5) {
            float2 ndc_prev = clip_prev.xy / clip_prev.w;
            // NDC -> UV. Flip Y: Metal texture origin is top-left
            // (v=0 at top), NDC has Y up (y=1 at top).
            float2 uv_prev  = float2(ndc_prev.x * 0.5 + 0.5,
                                      1.0 - (ndc_prev.y * 0.5 + 0.5));
            float2 uv_cur   = (float2(gid) + 0.5) / float2(float(w), float(h));
            // f.3.d-fix -- sign was reversed in f.3.b. Apple's docs
            // are explicit: the motion vector points FROM current TO
            // previous, so motion = uv_prev - uv_cur. With the wrong
            // sign MetalFX was reusing history from mirror-image
            // pixels during orbits.
            motion = uv_prev - uv_cur;
        }
    }
    motionTex.write(float4(motion.x, motion.y, 0.0, 0.0), gid);

    // No display write here -- rt_denoise reads accumTex + normalTex
    // and produces the final display output in its own kernel pass.
}

// ── Denoise kernel ───────────────────────────────────────────────────
//
// 5x5 edge-aware spatial blur. For each pixel, samples 25 neighbors
// (clamped at image edges), weights each by (1) the neighbor's
// validity (.w from normalTex), (2) cosine similarity of normals,
// (3) a Gaussian-ish spatial falloff. The weighted average is
// written to the display.
//
// Convergence-aware: at high frame counts (= clean accum), the blur
// becomes almost transparent because the input is already smooth.
// At low frame counts (orbit case), the blur kicks in hardest.
// Strength taper based on frame_count is a simple implementation
// of the "history confidence" trick.

constant float DENOISE_SIGMA_N = 0.15;  // higher = more cross-edge blur

kernel void rt_denoise(
    texture2d<float, access::write>      outTex      [[texture(0)]],
    texture2d<float, access::read>       accumTex    [[texture(1)]],
    texture2d<float, access::read>       normalTex   [[texture(2)]],
    constant PathState&                  path        [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint w = outTex.get_width();
    const uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    const uint frame = path.frame_count;
    const float inv_samples = 1.0 / float(frame + 1u);

    const float4 self_normal_v = normalTex.read(gid);
    const float3 self_normal = self_normal_v.xyz;
    const bool   self_valid  = self_normal_v.w > 0.5;

    // For sky pixels (primary miss), no neighbor blending -- they
    // tend to be uniformly-colored gradient already.
    if (!self_valid) {
        const float3 c = accumTex.read(gid).rgb * inv_samples;
        outTex.write(float4(c, 1.0), gid);
        return;
    }

    // Per-pixel "noise budget" tapers with sample count. Many samples
    // = converged image = mostly pass-through. Few samples = noisy =
    // full blur. Clamps the high-spp case toward identity so static
    // scenes look as crisp as the path tracer produces them.
    const float taper = 1.0 / (1.0 + float(frame) * 0.05);

    float3 sum  = float3(0.0);
    float  wsum = 0.0;
    const int R = 2;  // 5x5 footprint
    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            const int sx = clamp(int(gid.x) + dx, 0, int(w) - 1);
            const int sy = clamp(int(gid.y) + dy, 0, int(h) - 1);
            const uint2 sgid = uint2(uint(sx), uint(sy));

            const float4 nv = normalTex.read(sgid);
            const float3 N_n = nv.xyz;
            const bool  valid = nv.w > 0.5;
            if (!valid) continue;

            // Cosine similarity (1 = identical normal, 0 = perpendicular,
            // -1 = opposite). Convert to a Gaussian-ish weight.
            const float cos_sim = max(0.0, dot(self_normal, N_n));
            const float wN = exp(-(1.0 - cos_sim) * (1.0 - cos_sim) /
                                 (2.0 * DENOISE_SIGMA_N * DENOISE_SIGMA_N));

            // Gaussian spatial falloff (sigma ≈ R/2 = 1 pixel).
            const float r2 = float(dx * dx + dy * dy);
            const float wS = exp(-r2 * 0.5);

            const float w = wN * wS;
            const float3 c = accumTex.read(sgid).rgb * inv_samples;
            sum  += c * w;
            wsum += w;
        }
    }
    const float3 filtered = (wsum > 0.0) ? sum / wsum : accumTex.read(gid).rgb * inv_samples;
    const float3 unfiltered = accumTex.read(gid).rgb * inv_samples;

    // Blend filtered ← unfiltered by `taper`. High frame = mostly
    // unfiltered (= accum already clean). Low frame = mostly filtered
    // (= cleaning up the 1-spp noise).
    const float3 out_color = mix(unfiltered, filtered, taper);
    outTex.write(float4(out_color, 1.0), gid);
}

// ── Tonemap kernel (f.3.c, ACES upgrade in f.3.d) ────────────────────
//
// Copies the MetalFX output (RGBA16Float HDR) into the display
// texture (RGBA8Unorm). f.3.d switches from a basic Reinhard
// (which dimmed mid-tones noticeably -- 1.0 in becomes 0.5 out)
// to the Narkowicz approximation of the ACES filmic curve. ACES
// preserves bright mid-tones much better, gives more contrast,
// and is the industry-standard answer for tonemapping HDR
// path-traced output.
//
// EXPOSURE knob multiplies the linear color before the curve --
// 1.0 is "as rendered"; 1.5 lifts the image slightly to compensate
// for the curve's natural dimming of dark mids.

constant float EXPOSURE = 1.2;

inline float3 aces_filmic(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

kernel void rt_tonemap(
    texture2d<float, access::write> outTex  [[texture(0)]],
    texture2d<float, access::read>  hdrTex  [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint w = outTex.get_width();
    const uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float3 c = max(hdrTex.read(gid).rgb, float3(0.0));
    c *= EXPOSURE;
    c = aces_filmic(c);
    outTex.write(float4(c, 1.0), gid);
}
