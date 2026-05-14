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
    // f.3.d-fix6 -- samples-per-pixel-per-frame. Mapped from the
    // editor's `quality` preset (1 / 4 / 16) with an optional user
    // override on the `samples` param. Higher values target edge /
    // disocclusion noise that TDS history validation can't clear on
    // its own. CPU clamps to [1, 16] before upload; kernel re-clamps.
    uint     spp;
    // f.3.d-fix6 -- max bounce depth, replacing the file-scope
    // `constant int MAX_BOUNCES = 4` the kernel used pre-fix6.
    // Mapped from the editor's `quality` preset (2 / 4 / 8) with an
    // optional user override on the `bounces` param. CPU clamps to
    // [1, 8] before upload; kernel re-clamps.
    uint     max_bounces;
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
// f.3.d-fix6 -- MAX_BOUNCES moved from file-scope constant into the
// kernel as a uniform-driven local (see rt_scene below). Editor's
// `quality` preset drives the value; CPU clamps to [1, 8] on set.
constant float NORMAL_FLIP_THR = 0.05;
// f.3.h-fix1 -- per-primary-ray Monte-Carlo sample count for area
// lights. Each primary fires this many shadow rays into each area
// light slot and averages. 1 = old behavior (relies entirely on TDS
// for penumbra smoothing -> grainy in motion); 4 = solid trade-off,
// the penumbra reads as a smooth gradient even in motion; 8+ =
// render-grade. Cost scales linearly per area light per bounce, so
// keep this tight at preview presets. Non-area lights ignore this
// (loop runs once for them).
constant uint AREA_SAMPLES_PER_LIGHT = 4u;

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

// ── 5.4-rt environment sampling (ported from editor WGSL) ────────────
// Modes mirror the editor's sample_env mode dispatch:
//   0 = hemisphere fallback (matches pre-5.4-rt look exactly)
//   1 = GradientSky: 3-stop top/horizon/ground gradient
//   2 = ProceduralSky: physics-based Rayleigh+Mie scattering
//   3 = HDRI: not yet supported on RT (needs texture binding;
//             separate sprint). Falls back to mode 0.

constant float3 SKY_BETA_R     = float3(0.030, 0.070, 0.170);
constant float  SKY_PATH_SCALE = 4.0;

inline float _sky_optical_depth(float cos_zenith) {
    return 1.0 / max(cos_zenith + 0.05, 0.08);
}

inline float3 _sky_sun_extinction(float sunCosZenith, float turbidity) {
    float depth = _sky_optical_depth(sunCosZenith);
    float3 betaM = float3(0.030) * turbidity;
    return exp(-(SKY_BETA_R + betaM) * depth * SKY_PATH_SCALE);
}

// Smooth physics-based atmosphere (no point features). Suitable for
// IBL ambient + secondary bounces; high-frequency disk/moon/stars
// land in _features only.
inline float3 sample_procedural_sky_smooth_rt(float3 dir, constant PathState& path) {
    const float3 sunDir    = path.env_sun.xyz;
    const float  sunVis    = path.env_sun.w;
    const float  turbidity = path.env_params.z;
    const float  mieG      = path.env_params.w;
    const float  intensity = path.env_params.y;
    const float  view_y    = clamp(dir.y, -1.0f, 1.0f);
    const float  sun_elev  = clamp(sunDir.y, -1.0f, 1.0f);
    const float  day       = smoothstep(-0.08f, 0.18f, sun_elev);

    const float3 betaR = SKY_BETA_R;
    const float3 betaM = float3(0.030) * turbidity;
    const float cos_theta = clamp(dot(dir, sunDir), -1.0f, 1.0f);
    const float cos2 = cos_theta * cos_theta;

    const float phaseR = 0.0596f * (1.0f + cos2);
    const float g  = clamp(mieG, 0.0f, 0.95f);
    const float g2 = g * g;
    const float phaseM_denom = pow(max(1.0f + g2 - 2.0f * g * cos_theta, 1e-4f), 1.5f);
    const float phaseM = 0.119f * (1.0f - g2) * (1.0f + cos2) / ((2.0f + g2) * phaseM_denom);

    const float sunDepth  = _sky_optical_depth(max(sun_elev, -0.05f));
    const float viewDepth = _sky_optical_depth(max(view_y, 0.0f));
    const float3 extinction = exp(-(betaR + betaM) * sunDepth * SKY_PATH_SCALE);
    const float3 sunRadiance = float3(1.5f, 1.4f, 1.3f) * extinction;

    const float3 viewExt = exp(-(betaR + betaM) * viewDepth * SKY_PATH_SCALE * 0.6f);
    const float3 inR = sunRadiance * betaR * phaseR;
    const float3 inM = sunRadiance * betaM * phaseM;
    float3 sky_day = (inR + inM) * (float3(1.0f) - viewExt) * 80.0f;

    // Multi-scatter ambient (mimics 2+ bounces among air molecules
    // that single-scattering misses, so the sky stays blue-everywhere
    // at noon instead of dimming away from the sun).
    const float3 multiScatter = float3(0.18f, 0.42f, 0.92f) *
                                smoothstep(-0.05f, 0.35f, sun_elev) *
                                (float3(1.0f) - viewExt);
    sky_day = sky_day + multiScatter * 1.4f;

    const float3 zen_night = float3(0.015f, 0.022f, 0.060f);
    const float3 hor_night = float3(0.045f, 0.050f, 0.095f);
    const float3 sky_night = mix(hor_night, zen_night, smoothstep(0.0f, 1.0f, max(view_y, 0.0f)));

    const float3 ground_day   = float3(0.18f, 0.13f, 0.10f);
    const float3 ground_night = float3(0.010f, 0.012f, 0.025f);
    const float3 ground = mix(ground_night, ground_day, day);

    float3 col = mix(sky_night, sky_day, day);
    if (view_y < 0.0f) col = ground;

    const float halo = phaseM * sunVis * 1.0f;
    col = col + halo * sunRadiance;

    return col * intensity;
}

// High-frequency features: sun disk, moon (with phase), star field,
// Milky Way band. Sky-pass-only -- aliases through smooth normals
// if used in IBL.
inline float _hash21_rt(float2 p) {
    return fract(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453f);
}
inline float3 sample_procedural_sky_features_rt(float3 dir, constant PathState& path) {
    const float3 sunDir    = path.env_sun.xyz;
    const float  sunVis    = path.env_sun.w;
    const float  turbidity = path.env_params.z;
    const float  intensity = path.env_params.y;
    const float  moonPhase = path.env_sky.w;
    const float  view_y    = clamp(dir.y, -1.0f, 1.0f);
    const float  sun_elev  = clamp(sunDir.y, -1.0f, 1.0f);
    const float  day       = smoothstep(-0.08f, 0.18f, sun_elev);
    const float  night     = 1.0f - day;

    float3 col = float3(0.0f);

    // Sun disk + atmospheric-extinction color
    const float cos_theta = clamp(dot(dir, sunDir), -1.0f, 1.0f);
    const float disk = smoothstep(0.99988f, 0.99996f, cos_theta) * sunVis * 14.0f;
    const float3 sunExt = _sky_sun_extinction(max(sun_elev, -0.05f), turbidity);
    const float3 sunDiskColor = sunExt * float3(2.2f, 2.0f, 1.8f);
    col = col + disk * sunDiskColor;

    // Moon (opposite the sun; phase modulates brightness)
    const float3 moonDir = -sunDir;
    const float cos_moon = clamp(dot(dir, moonDir), -1.0f, 1.0f);
    const float moon_vis = clamp(night, 0.0f, 1.0f);
    const float phaseLit = sin(clamp(moonPhase, 0.0f, 1.0f) * 3.14159265f);
    const float moon_disk = smoothstep(0.99988f, 0.99996f, cos_moon) * moon_vis * 5.0f * phaseLit;
    const float moon_halo = smoothstep(0.985f, 0.9999f, cos_moon)    * moon_vis * 0.05f * phaseLit;
    col = col + (moon_disk + moon_halo) * float3(0.92f, 0.95f, 1.00f);

    // Stars + Milky Way band (night only, above horizon)
    if (view_y > 0.0f && moon_vis > 0.001f) {
        const float2 p = floor(dir * 360.0f).xy * 1.0f + floor(dir.z * 360.0f) * float2(0.0f, 7.13f);
        // Simpler: project on a hashed integer grid using all 3 dims
        const float3 p3 = floor(dir * 360.0f);
        const float h = fract(sin(dot(p3, float3(127.1f, 311.7f, 74.7f))) * 43758.5453f);
        const float star_amt = smoothstep(0.989f, 1.0f, h) * moon_vis;
        const float temp_h = fract(sin(dot(p3, float3(89.3f, 47.7f, 31.1f))) * 76123.4f);
        float3 starColor;
        if (temp_h < 0.5f) {
            starColor = mix(float3(0.62f, 0.78f, 1.00f),
                            float3(1.00f, 0.98f, 0.95f),
                            smoothstep(0.0f, 0.5f, temp_h));
        } else {
            starColor = mix(float3(1.00f, 0.98f, 0.95f),
                            float3(1.00f, 0.72f, 0.45f),
                            smoothstep(0.5f, 1.0f, temp_h));
        }
        const float twinkle = 0.55f + 0.45f * fract(sin(dot(p3, float3(19.7f, 41.3f, 7.1f))) * 12345.6f);
        col = col + starColor * star_amt * twinkle * 1.4f;

        const float3 galacticAxis = normalize(float3(0.45f, 0.55f, 0.70f));
        const float bandDist = fabs(dot(dir, galacticAxis));
        const float bandIntensity = smoothstep(0.35f, 0.05f, bandDist);
        const float nhash = fract(sin(dot(floor(dir * 24.0f), float3(17.3f, 91.2f, 53.5f))) * 8194.7f);
        const float bandTex = 0.45f + 0.55f * nhash;
        col = col + float3(0.22f, 0.20f, 0.32f) * bandIntensity * bandTex * moon_vis * 0.65f;
    }

    return col * intensity;
}

// 2D noise + fbm for clouds (ported from WGSL).
inline float _cloud_noise2d(float2 p) {
    const float2 i = floor(p);
    const float2 f = fract(p);
    const float2 u = f * f * (3.0f - 2.0f * f);
    const float a = _hash21_rt(i);
    const float b = _hash21_rt(i + float2(1.0f, 0.0f));
    const float c = _hash21_rt(i + float2(0.0f, 1.0f));
    const float d = _hash21_rt(i + float2(1.0f, 1.0f));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
inline float _cloud_fbm(float2 p) {
    float v = 0.0f;
    float amp = 0.5f;
    float freq = 1.0f;
    for (int i = 0; i < 5; i++) {
        v += amp * _cloud_noise2d(p * freq);
        freq *= 2.07f;
        amp  *= 0.5f;
    }
    return v;
}

inline float4 sample_clouds_rt(float3 dir, constant PathState& path) {
    const float coverage = path.env_cloud_params.x;
    const float density  = path.env_cloud_params.y;
    if (coverage <= 0.0f || density <= 0.0f) return float4(0.0f);
    const float yMask = smoothstep(0.0f, 0.12f, dir.y);
    if (yMask <= 0.0f) return float4(0.0f);
    const float H = 250.0f;
    const float t = H / max(dir.y, 0.02f);
    const float3 pos = dir * t;
    const float2 wind = path.env_cloud_params.zw;
    const float2 xz = (pos.xz - wind) * 0.0025f;
    const float raw = _cloud_fbm(xz);
    const float alpha = smoothstep(coverage * 0.85f, coverage * 0.85f + 0.22f, raw) * density * yMask;
    const float cos_theta = clamp(dot(normalize(dir), path.env_sun.xyz), -1.0f, 1.0f);
    const float scatter = pow(max(cos_theta * 0.5f + 0.5f, 0.0f), 2.0f);
    const float sun_elev = clamp(path.env_sun.y, -1.0f, 1.0f);
    const float3 lit = mix(float3(1.10f, 0.85f, 0.65f),
                           float3(1.05f, 1.02f, 0.98f),
                           smoothstep(0.0f, 0.35f, sun_elev));
    const float3 shadow = float3(0.35f, 0.40f, 0.50f);
    const float day = smoothstep(-0.10f, 0.20f, sun_elev);
    const float3 cloudCol = mix(shadow, lit, scatter * path.env_sun.w) * day;
    return float4(cloudCol, clamp(alpha, 0.0f, 0.95f));
}

// Mode-dispatched env sample. Smooth-only (no features) -- suitable
// for IBL ambient + secondary bounces. RT engine doesn't yet support
// HDRI texture (mode 3); falls back to hemisphere.
inline float3 sample_env_smooth_rt(float3 dir, constant PathState& path) {
    const uint mode = (uint)path.env_params.x;
    if (mode == 1u) {
        const float t = clamp(dir.y, -1.0f, 1.0f);
        const float intensity = path.env_params.y;
        if (t > 0.0f) {
            return mix(path.env_horizon.rgb, path.env_sky.rgb,    smoothstep(0.0f, 1.0f, t)) * intensity;
        } else {
            return mix(path.env_horizon.rgb, path.env_ground.rgb, smoothstep(0.0f, 1.0f, -t)) * intensity;
        }
    }
    if (mode == 2u) {
        return sample_procedural_sky_smooth_rt(dir, path);
    }
    return hemisphere_ibl(dir);
}

// Full env sample (smooth + features + clouds). Use for primary-ray
// miss (the sky background visible to the camera); secondary
// bounces stick with sample_env_smooth_rt to avoid feature aliasing.
inline float3 sample_env_full_rt(float3 dir, constant PathState& path) {
    const uint mode = (uint)path.env_params.x;
    float3 col = sample_env_smooth_rt(dir, path);
    if (mode == 2u) {
        col = col + sample_procedural_sky_features_rt(dir, path);
        const float4 cloud = sample_clouds_rt(dir, path);
        col = mix(col, cloud.rgb, cloud.a);
    }
    return col;
}

// Distance fog. Mirrors the editor's apply_fog: Beer-Lambert over
// (dist - start) with optional ground-fog height falloff. Fog color
// is the env-horizon (autoEnv=1, projects camera ray to y=0) or a
// manual rgb.
inline float3 apply_fog_rt(float3 color, float3 worldPos, float3 eye, float3 camForward, constant PathState& path) {
    const float density = path.env_fog_params.x;
    if (density <= 0.0f) return color;
    const float start = path.env_fog_params.y;
    const float hf    = path.env_fog_params.z;
    const float dist  = length(eye - worldPos);
    const float beyond = max(dist - start, 0.0f);
    float heightFactor = 1.0f;
    if (hf > 0.0f) {
        heightFactor = smoothstep(hf * 2.0f, 0.0f, worldPos.y);
    }
    const float factor = 1.0f - exp(-density * beyond * heightFactor);
    float3 fogCol;
    if (path.env_fog_params.w > 0.5f) {
        const float3 horiz = normalize(float3(camForward.x, 0.0f, camForward.z) + float3(1e-4f, 0.0f, 0.0f));
        fogCol = sample_env_smooth_rt(horiz, path);
    } else {
        fogCol = path.env_fog_color.rgb;
    }
    return mix(color, fogCol, clamp(factor, 0.0f, 1.0f));
}

// ── Light evaluation ─────────────────────────────────────────────────

struct LightSample {
    float3 L;
    float  distance;
    float  attenuation;
};

inline LightSample resolve_light(constant LightSlot& slot, float3 hit_point,
                                thread uint& rng) {
    LightSample s;
    const uint type = slot.flags.x;
    if (type == 0u) {
        // Directional.
        s.L = slot.pos_or_dir.xyz;
        s.distance = 1e6;
        s.attenuation = 1.0;
    } else if (type == 3u) {
        // f.3.h -- Area light. LightSlot packing:
        //   pos_or_dir = (center.xyz, width)
        //   spot_dir   = (normal.xyz, height)
        //   color      = (rgb, intensity)
        //   flags.x    = 3
        // Sample one random point uniformly on the rectangle per
        // shadow-ray. Each primary ray lands on a different sample,
        // and TDS averages across primaries -> soft penumbra naturally.
        const float3 center = slot.pos_or_dir.xyz;
        const float  hw     = slot.pos_or_dir.w * 0.5;
        const float3 n      = slot.spot_dir.xyz;
        const float  hh     = slot.spot_dir.w * 0.5;
        // Build a tangent basis aligned with `n`. Pick an up-reference
        // that's not parallel to n (the y-axis works unless the light
        // faces near-straight-up, in which case fall back to x).
        const float3 up_ref = (abs(n.y) < 0.9) ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 t = normalize(cross(up_ref, n));
        const float3 b = cross(n, t);
        const float u1 = rand01(rng) * 2.0 - 1.0;  // [-1, +1]
        const float u2 = rand01(rng) * 2.0 - 1.0;
        const float3 sample_pos = center + t * (u1 * hw) + b * (u2 * hh);
        const float3 to_light = sample_pos - hit_point;
        s.distance = length(to_light);
        s.L = (s.distance > 0.0) ? to_light / s.distance : float3(0, 1, 0);
        // Monte Carlo weight for uniform sampling over the rect:
        //   contribution = Li * cos_surface * cos_light * area / r^2
        // shade_direct multiplies by lights.slots[i].color.xyz *
        // intensity (= Li) and n_dot_l (= cos_surface) separately, so
        // fold the remaining factors (cos_light, area, 1/r^2) into
        // attenuation.
        const float cos_light = max(0.0, dot(n, -s.L));
        const float area = slot.pos_or_dir.w * slot.spot_dir.w; // width * height
        const float r2 = max(s.distance * s.distance, 1e-4);
        s.attenuation = cos_light * area / r2;
    } else {
        // Point (type 1) or Spot (type 2).
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
    constant LightsUniform& lights,
    thread uint& rng                    // f.3.h -- needed for area-light sampling
) {
    if (mtype == 0u) return albedo;

    const float3 shadow_origin = hit_point + N * SHADOW_BIAS;
    const uint nLights = lights.meta.x;
    float3 color = float3(0.0);

    if (mtype == 1u) {
        // Phong direct.
        float3 diffuse = float3(0.0), specular = float3(0.0);
        for (uint i = 0; i < nLights; i++) {
            // f.3.h-fix1 -- area lights take N samples per primary
            // ray to flatten penumbra MC noise. Other light types
            // are deterministic; the inner loop runs once for them.
            const uint ltype = lights.slots[i].flags.x;
            const uint samples = (ltype == 3u) ? AREA_SAMPLES_PER_LIGHT : 1u;
            float3 li_diffuse = float3(0.0);
            float3 li_specular = float3(0.0);
            for (uint a = 0u; a < samples; a++) {
                const LightSample ls = resolve_light(lights.slots[i], hit_point, rng);
                if (ls.attenuation <= 0.0) continue;
                const float n_dot_l = max(0.0, dot(N, ls.L));
                if (n_dot_l <= 0.0) continue;
                ray sr; sr.origin = shadow_origin; sr.direction = ls.L;
                sr.min_distance = 0.0; sr.max_distance = ls.distance - SHADOW_BIAS;
                if (shadow_isr.intersect(sr, accel).type == intersection_type::triangle) continue;
                const float3 lc = lights.slots[i].color.xyz * lights.slots[i].color.w * ls.attenuation;
                li_diffuse += lc * n_dot_l;
                const float3 H = normalize(ls.L + V);
                li_specular += lc * pow(max(0.0, dot(N, H)), max(shininess, 1.0));
            }
            const float inv_s = 1.0 / float(samples);
            diffuse  += li_diffuse  * inv_s;
            specular += li_specular * inv_s;
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
            // f.3.h-fix1 -- area lights take N samples per primary
            // ray to flatten penumbra MC noise. Other light types
            // are deterministic; the inner loop runs once for them.
            const uint ltype = lights.slots[i].flags.x;
            const uint samples = (ltype == 3u) ? AREA_SAMPLES_PER_LIGHT : 1u;
            float3 li_Lo = float3(0.0);
            for (uint a = 0u; a < samples; a++) {
                const LightSample ls = resolve_light(lights.slots[i], hit_point, rng);
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
                li_Lo += (kD * albedo / PI + spec) * lc * n_dot_l;
            }
            Lo += li_Lo * (1.0 / float(samples));
        }
        // Dim hemisphere-IBL ambient -- path tracing's indirect bounce
        // gives the "real" indirect light. The IBL term here is a
        // fallback that gets dialed down once we have proper GI.
        const float3 R = reflect(-V, N);
        // 5.4-rt -- IBL ambient via wired env (mode 0 = hemisphere
        // fallback, matches pre-5.4-rt look exactly). Smooth-only
        // sampler -- features (sun disk / stars / etc) would alias
        // through smooth surface normals.
        const float3 sky_refl = sample_env_smooth_rt(R, path);
        const float3 sky_diff = sample_env_smooth_rt(N, path);
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
    // f.3.d-fix6 -- samples-per-pixel-per-frame, clamped to a safe
    // upper bound here so a garbage upload can't lock up the GPU.
    // 16 is the editor's `final` preset; CPU also clamps to 16.
    const uint SPP = clamp(path.spp, 1u, 16u);
    // f.3.d-fix6 -- max bounce depth, replacing the file-scope
    // `constant int MAX_BOUNCES = 4` the kernel used pre-fix6.
    // Clamped here too; 8 is the editor's `final` preset ceiling.
    const uint MAX_BOUNCES = clamp(path.max_bounces, 1u, 8u);

    // f.3.h -- ray origin is per-pixel; ray direction is now
    // per-SAMPLE (computed inside the SPP loop). Pre-fix h the
    // direction was computed once per pixel using path.jitter only,
    // so SPP cleaned up bounce/specular variance but never
    // contributed to edge AA. Now sample 0 still uses path.jitter
    // (TDS history reprojection finds the canonical sub-pixel
    // position) and samples 1..SPP-1 use rand01 for fresh sub-pixel
    // positions. Averaged into sample_accum / SPP, this gives true
    // in-frame anti-aliasing of silhouettes that TDS's cross-frame
    // accumulation couldn't deliver while the camera was in motion.
    const float3 ray_origin = camera.eye.xyz;

    intersector<triangle_data> isr;
    isr.assume_geometry_type(geometry_type::triangle);
    intersector<triangle_data> shadow_isr;
    shadow_isr.assume_geometry_type(geometry_type::triangle);
    shadow_isr.accept_any_intersection(true);

    // G-buffer locals -- written by sample 0 only (because of the
    // `!primary_recorded` guard inside the bounce loop; once set,
    // subsequent samples skip the assignment).
    float3 primary_normal     = float3(0.0);
    float3 primary_hit_point  = float3(0.0);
    float3 primary_albedo     = float3(0.0);
    float  primary_depth      = 0.0;
    float  primary_roughness  = 1.0;
    float3 primary_specular   = float3(0.04);
    bool   primary_recorded   = false;

    // Across-samples color accumulator. Each sample's contribution
    // gets firefly-clamped individually (so one outlier doesn't
    // dominate the average) then added here. Final color = sum / SPP.
    float3 sample_accum = float3(0.0);

    for (uint s = 0u; s < SPP; s++) {
        // Per-sample RNG seed -- decorrelated across both pixels and
        // samples within a frame, plus across frames.
        thread uint rng  = init_rng(gid, frame * 16u + s, 0u);

        // f.3.h -- per-sample sub-pixel offset. Sample 0 uses the
        // per-frame Halton jitter (so TDS can reproject history at
        // the canonical sub-pixel position). Samples 1..SPP-1
        // randomize over the pixel via the RNG -- THIS is what
        // delivers in-frame edge AA. RNG advance is intentional;
        // bounce sampling later in the loop continues from here.
        float sub_x, sub_y;
        if (s == 0u) {
            sub_x = path.jitter.x;
            sub_y = path.jitter.y;
        } else {
            sub_x = rand01(rng);
            sub_y = rand01(rng);
        }
        const float u = (float(gid.x) + sub_x) / float(w);
        const float v = (float(gid.y) + sub_y) / float(h);
        const float screen_x = (2.0 * u - 1.0) * camera.misc.x * camera.misc.y;
        const float screen_y = (1.0 - 2.0 * v) * camera.misc.x;
        const float3 ray_direction = normalize(
            screen_x * camera.right.xyz
          + screen_y * camera.up.xyz
          + camera.forward.xyz
        );

        // Re-init the ray each sample because the bounce loop
        // mutates r.origin / r.direction.
        ray r;
        r.origin       = ray_origin;
        r.direction    = ray_direction;
        r.min_distance = 0.001;
        r.max_distance = 1000.0;

        float3 throughput = float3(1.0);
        float3 sample     = float3(0.0);
        bool   terminated = false;
        float  accumulated_dist   = 0.0;  // virtual depth through mirrors / glass

    for (uint bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        intersection_result<triangle_data> hit = isr.intersect(r, accel);
        if (hit.type != intersection_type::triangle) {
            // 5.4-rt -- env-aware sky color for miss rays. Primary
            // miss (bounce == 0) uses the full env including sun disk,
            // moon, stars, Milky Way + clouds (= what the camera sees
            // as the visible sky background). Secondary miss uses
            // smooth-only so features don't alias through reflected /
            // refracted ray directions.
            if (bounce == 0u) {
                sample += throughput * sample_env_full_rt(r.direction, path);
            } else {
                sample += throughput * sample_env_smooth_rt(r.direction, path);
            }
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
            N, V, n_dot_v, hit_point, accel, shadow_isr, lights, rng
        );
        sample += throughput * direct;

        // Indirect bounce (Lambertian sample), unless we've used up
        // the bounce budget. The throughput multiplier for a
        // cosine-weighted Lambert sample is just `albedo` (cos/pi PDF
        // cancels with the BRDF normalizer). For PBR materials,
        // approximating the indirect as diffuse-only is the c-e v1
        // simplification; the full energy-conserving PBR indirect
        // sample uses GGX importance sampling, which lands in c-e.2.
        if (bounce == MAX_BOUNCES - 1u) {
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
            // 5.4-rt -- post-bounce-budget miss. Always smooth-only
            // (the ray has bounced at least once; features would
            // alias).
            sample += throughput * sample_env_smooth_rt(r.direction, path);
        }

        // 5.4-rt -- distance fog (atmospheric perspective). Applied
        // only when the primary ray hit geometry; sky pixels are
        // already env-horizon-tinted so no extra fog mix needed.
        // Per-pixel ray_direction projected to the horizon plane
        // (inside apply_fog_rt) gives each pixel its OWN horizon
        // color -- actually more realistic atmospheric perspective
        // than the editor's uniform camera-forward sample.
        if (primary_recorded) {
            sample = apply_fog_rt(sample, primary_hit_point,
                                  camera.eye.xyz, ray_direction, path);
        }
        // f.3.d -- firefly clamp PER SAMPLE before averaging into the
        // cross-sample accumulator. Doing it per-sample (rather than
        // on the average) means one wild outlier from a glass / mirror
        // path doesn't get smeared across the other samples.
        sample_accum += min(sample, float3(8.0));
    }  // end SPP loop

    // f.3.d-fix6 -- mean across SPP samples. With SPP=1 this is just
    // the single sample; with SPP=2 we've halved per-frame variance.
    const float3 final_sample = sample_accum / float(SPP);

    // ── Accumulate (kept for spatial-denoise fallback) ───────────────
    // First frame: write sample as initial value. Subsequent frames:
    // read prior accum, add, write back. RGBA32F so we don't lose
    // precision summing thousands of samples. MetalFX-path doesn't
    // need this -- it consumes the noisy single-sample texture below.
    float3 new_accum;
    if (frame == 0u) {
        new_accum = final_sample;
    } else {
        new_accum = accumTex.read(gid).rgb + final_sample;
    }
    accumTex.write(float4(new_accum, 0.0), gid);

    // f.3.c -- noisy single-sample color for MetalFX. (Clamping was
    // done per-sample inside the SPP loop, so we just write the mean
    // directly here.)
    noisyColorTex.write(float4(final_sample, 1.0), gid);

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
