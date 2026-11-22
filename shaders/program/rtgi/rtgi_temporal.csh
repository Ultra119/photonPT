// Reproject the previous statistical history into the current raster receiver.

#include "/include/global.glsl"

#if RTGI_RESOLUTION == 0
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif RTGI_RESOLUTION == 2
const vec2 workGroupsRender = vec2(1.0, 1.0);
#else
const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif
layout(local_size_x = 8, local_size_y = 8) in;

// Each invocation reads only the previous bank and writes only the current
// bank. Use image loads here so neither image also needs a sampler binding.
layout(rgba32f) uniform image2D rtgi_accum_img;
layout(rgba32f) uniform image2D rtgi_history_img;
layout(rgba32f) uniform image2D rtgi_stats_even_img;
layout(rgba32f) uniform image2D rtgi_stats_odd_img;

uniform sampler2D rtgi_raw_sampler;
uniform usampler2D rtgi_frame_state_sampler;
uniform usampler2D rtgi_guide_even_sampler;
uniform usampler2D rtgi_guide_odd_sampler;

#if GI_DEBUG == GI_DEBUG_RESET
layout(rgba16f) writeonly uniform image2D rtgi_state_img;
#elif GI_DEBUG == GI_DEBUG_REJECT
layout(rgba8) writeonly uniform image2D rtgi_reject_img;
#endif

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform int frameCounter;

#include "/include/utility/camera_motion.glsl"
#include "/include/utility/color.glsl"
#include "/include/lighting/rt/rtgi_surface.glsl"
#include "/include/lighting/rt/pt_state.glsl"

struct RtgiHistory {
    vec4 mean;
    vec4 stats;
    float weight;
};

uvec4 load_guide(ivec2 texel, bool even) {
    return even ? texelFetch(rtgi_guide_even_sampler, texel, 0)
                : texelFetch(rtgi_guide_odd_sampler, texel, 0);
}

void store_history(
    ivec2 texel, bool even, vec4 mean, vec4 stats,
    float reason, float weight, bool valid_raw, float alpha
) {
    if (even) {
        imageStore(rtgi_accum_img, texel, mean);
        imageStore(rtgi_stats_even_img, texel, stats);
    } else {
        imageStore(rtgi_history_img, texel, mean);
        imageStore(rtgi_stats_odd_img, texel, stats);
    }
#if GI_DEBUG == GI_DEBUG_RESET
    imageStore(rtgi_state_img, texel, vec4(reason, weight, float(valid_raw), alpha));
#elif GI_DEBUG == GI_DEBUG_REJECT
    float visible = float(stats.z > 0.0);
    imageStore(rtgi_reject_img, texel, vec4(visible * (1.0 - weight), weight, 0.0, 1.0));
#endif
}

RtgiHistory gather_history(
    vec2 previous_pixel, RtgiGrid grid, vec2 previous_jitter, bool previous_even,
    vec3 predicted_view, vec3 world_normal, uint material
) {
    ivec2 base;
    vec2 fraction;
    rtgi_footprint(previous_pixel, grid, base, fraction);

    RtgiHistory history = RtgiHistory(vec4(0.0), vec4(0.0), 0.0);
    float best_weight = 0.0;
    vec3 normal = mat3(gbufferPreviousModelView) * world_normal;
    float footprint = rtgi_view_footprint(-predicted_view.z, grid.gi_active,
                                          gbufferPreviousProjection);
    float grazing = max(abs(dot(normal, normalize(predicted_view))), 0.02);
    float max_chord = min(6.0, 0.1 + 2.5 * footprint / grazing);

    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            ivec2 offset = ivec2(x, y);
            ivec2 tap = base + offset;
            float weight = rtgi_bilinear_weight(offset, fraction);
            if (weight <= 0.0 || !rtgi_in_bounds(tap, grid.gi_active)) {
                continue;
            }

            uvec4 guide = load_guide(tap, previous_even);
            vec3 view_pos;
            if (!rtgi_guide_matches(guide, material)
                || !rtgi_view_from_guide(guide, grid.full_size, previous_jitter,
                                         gbufferPreviousProjection, view_pos)) {
                continue;
            }
            vec3 tap_normal = mat3(gbufferPreviousModelView) * rtgi_guide_normal(guide);
            weight *= rtgi_geometry_weight(view_pos - predicted_view, normal,
                                           tap_normal, footprint, max_chord);
            if (weight <= 0.0) {
                continue;
            }

            vec4 mean = previous_even ? imageLoad(rtgi_accum_img, tap)
                                      : imageLoad(rtgi_history_img, tap);
            vec4 stats = previous_even ? imageLoad(rtgi_stats_even_img, tap)
                                       : imageLoad(rtgi_stats_odd_img, tap);
            if (!rtgi_finite(mean) || any(lessThan(mean.rgb, vec3(0.0)))
                || !rtgi_finite(stats) || stats.x < 0.0 || stats.y <= 0.0
                || stats.y > 1.0 || stats.z < 1.0) {
                continue;
            }

            history.weight += weight;
            float a = weight / history.weight;
            float delta = dot(mean.rgb - history.mean.rgb, luminance_weights);
            // Pool mixture variance in centered form and use the conservative
            // bound on Q when taps can share temporal ancestors.
            history.stats.x = mix(history.stats.x, stats.x, a) + a * (1.0 - a) * sqr(delta);
            history.stats.yz = mix(history.stats.yz, stats.yz, a);
            history.mean.rgb = mix(history.mean.rgb, mean.rgb, a);
            if (weight > best_weight) {
                best_weight = weight;
                history.stats.w = stats.w;
                history.mean.a = mean.a;
            }
        }
    }

    return history;
}

void rtgi_neighborhood_change(
    ivec2 texel, ivec2 prev_shift, RtgiGrid grid, vec2 jitter, bool even, uint material,
    vec3 center_view, vec3 normal, float center_raw, float center_prev,
    out float mean_raw, out float mean_prev, out int sample_count
) {
    float footprint = rtgi_view_footprint(-center_view.z, grid.gi_active, gbufferProjection);
    vec3 view_normal = transpose(mat3(gbufferModelViewInverse)) * normal;
    bool previous_even = !even;
    float sum_raw = center_raw;
    float sum_prev = center_prev;
    sample_count = 1;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 tap = texel + ivec2(x, y);
            if (tap == texel || !rtgi_in_bounds(tap, grid.gi_active)) {
                continue;
            }
            ivec2 prev_tap = tap + prev_shift;
            if (!rtgi_in_bounds(prev_tap, grid.gi_active)) {
                continue;
            }
            uvec4 guide = load_guide(tap, even);
            if ((guide.w & rtgi_guide_raw_valid) == 0u || !rtgi_guide_matches(guide, material)) {
                continue;
            }
            uvec4 p_guide = load_guide(prev_tap, previous_even);
            if (!rtgi_guide_matches(p_guide, material)) {
                continue;
            }
            vec4 raw = texelFetch(rtgi_raw_sampler, tap, 0);
            vec4 prev_mean = previous_even ? imageLoad(rtgi_accum_img, prev_tap)
                                           : imageLoad(rtgi_history_img, prev_tap);
            vec3 tap_view;
            if (!rtgi_finite(raw) || any(lessThan(raw.rgb, vec3(0.0)))
                || !rtgi_finite(prev_mean) || any(lessThan(prev_mean.rgb, vec3(0.0)))
                || !rtgi_view_from_guide(guide, grid.full_size, jitter, gbufferProjection, tap_view)) {
                continue;
            }
            vec3 tap_normal = transpose(mat3(gbufferModelViewInverse)) * rtgi_guide_normal(guide);
            if (rtgi_geometry_weight(tap_view - center_view, view_normal, tap_normal,
                                     footprint, 0.25 + 3.0 * footprint) > 0.5) {
                sum_raw += dot(raw.rgb, luminance_weights);
                sum_prev += dot(prev_mean.rgb, luminance_weights);
                ++sample_count;
            }
        }
    }
    float inv_count = 1.0 / float(sample_count);
    mean_raw = sum_raw * inv_count;
    mean_prev = sum_prev * inv_count;
}

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 target_res = imageSize(rtgi_accum_img);
    if (!rtgi_in_bounds(texel, target_res)) {
        return;
    }

    int bank = frameCounter & 1;
    bool even = bank == 0;
    uvec4 guide = load_guide(texel, even);
    if ((guide.w & rtgi_guide_covered) == 0u) {
        uint kind = rtgi_guide_kind(guide);
        float reason = kind == rtgi_sky ? rtgi_reset_sky
                     : kind == rtgi_hand ? rtgi_reset_hand : rtgi_reset_coverage;
        store_history(texel, even, vec4(0.0), vec4(0.0), reason, 0.0, false, 0.0);
        return;
    }

    uvec4 frame = texelFetch(rtgi_frame_state_sampler, ivec2(0, bank), 0);
    uvec4 full_layout = texelFetch(rtgi_frame_state_sampler, ivec2(1, bank), 0);
    uvec4 gi_layout = texelFetch(rtgi_frame_state_sampler, ivec2(2, bank), 0);
    RtgiGrid grid = RtgiGrid(ivec2(full_layout.xy), ivec2(full_layout.zw), ivec2(gi_layout.zw));
    vec2 jitter = uintBitsToFloat(frame.zw);
    vec3 view_pos;
    if (frame.y != rtgi_frame_tag || frame.x != uint(frameCounter)
        || !rtgi_view_from_guide(guide, grid.full_size, jitter, gbufferProjection, view_pos)) {
        store_history(texel, even, vec4(0.0), vec4(0.0), rtgi_reset_first, 0.0, false, 0.0);
        return;
    }

    vec4 raw = texelFetch(rtgi_raw_sampler, texel, 0);
    bool valid_raw = (guide.w & rtgi_guide_raw_valid) != 0u
        && rtgi_finite(raw) && all(greaterThanEqual(raw.rgb, vec3(0.0)));
    vec3 world_normal = rtgi_guide_normal(guide);
    uint material = rtgi_guide_material(guide);

    uvec4 previous = texelFetch(rtgi_frame_state_sampler, ivec2(0, 1 - bank), 0);
    vec3 displacement = camera_delta();
    bool compatible = previous.y == rtgi_frame_tag
        && rtgi_previous_frame(previous.x, frame.x) && dot(displacement, displacement) < 256.0
        && full_layout == texelFetch(rtgi_frame_state_sampler, ivec2(1, 1 - bank), 0)
        && gi_layout == texelFetch(rtgi_frame_state_sampler, ivec2(2, 1 - bank), 0);

    RtgiHistory history = RtgiHistory(vec4(0.0), vec4(0.0), 0.0);
    ivec2 prev_shift = ivec2(0);
    float reason = rtgi_reset_first;
    if (compatible) {
        vec3 scene_pos = transform(gbufferModelViewInverse, view_pos);
        vec3 previous_view = transform(gbufferPreviousModelView, scene_pos + displacement);
        vec4 clip = gbufferPreviousProjection * vec4(previous_view, 1.0);
        reason = rtgi_reset_offscreen;
        if (rtgi_finite(clip) && clip.w > 0.0 && abs(clip.z) <= clip.w && previous_view.z < 0.0) {
            vec2 previous_jitter = uintBitsToFloat(previous.zw);
            vec2 previous_pixel = rtgi_raster_position(clip.xy / clip.w, grid.full_size, previous_jitter);
            if (all(greaterThanEqual(previous_pixel, vec2(0.0)))
                && all(lessThan(previous_pixel, vec2(grid.full_active)))) {
                ivec2 p_base;
                vec2 p_frac;
                rtgi_footprint(previous_pixel, grid, p_base, p_frac);
                prev_shift = ivec2(round(vec2(p_base) + p_frac)) - texel;
                history = gather_history(previous_pixel, grid, previous_jitter,
                                         !even, previous_view, world_normal, material);
                reason = rtgi_reset_geometry;
            }
        }
    }

    float support = clamp01(history.weight);
    if (!valid_raw) {
        // Keep well-supported history on a dropped observation without aging it.
        bool keep = support >= 0.5;
        history.stats.w = 0.0;
        store_history(texel, even, keep ? history.mean : vec4(0.0),
                      keep ? history.stats : vec4(0.0), rtgi_reset_invalid, support, false, 0.0);
        return;
    }
    if (support <= 1e-4) {
        store_history(texel, even, raw, vec4(0.0, 1.0, 1.0, 0.0), reason, support, true, 1.0);
        return;
    }

    const float age_cap = float(RTGI_SPP_CAP) / float(RTGI_RAY_COUNT);
    float age = min(history.stats.z * support + 1.0, age_cap);
    float alpha = max(1.0 / age, 0.01 * RTGI_MIN_ACCUM);
    float previous_luminance = dot(history.mean.rgb, luminance_weights);
    float luminance = dot(raw.rgb, luminance_weights);
    float delta = luminance - previous_luminance;

    float evidence = 0.0;
    reason = rtgi_reset_none;
    if (RTGI_ADAPT_STRENGTH > 0.0 && support >= 0.65 && history.stats.z >= 6.0) {
        float nbr_raw, nbr_prev;
        int nbr_count;
        rtgi_neighborhood_change(
            texel, prev_shift, grid, jitter, even, material, view_pos, world_normal,
            luminance, previous_luminance, nbr_raw, nbr_prev, nbr_count
        );
        if (nbr_count >= 4) {
            // Compare matched 3x3 same-surface raw vs previous means so steep
            // 1/r^2 peaks have zero curvature bias while independent pixel noise
            // drops by sqrt(N), cleanly separating lighting steps from sparse noise.
            float nbr_delta = nbr_raw - nbr_prev;
            float frame_sigma = sqrt(history.stats.x / max(1.0 - history.stats.y, 1e-5));
            float nbr_sigma = frame_sigma * inversesqrt(float(nbr_count));
            float threshold = max(0.008, max(0.2 * max(nbr_raw, nbr_prev), 3.0 * nbr_sigma));
            if (abs(nbr_delta) > threshold) {
                float direction = sign(nbr_delta);
                evidence = clamp(history.stats.w * direction + 1.0, 1.0, 3.0) * direction;
                if (abs(evidence) >= 3.0) {
                    alpha = max(alpha, 0.55 * RTGI_ADAPT_STRENGTH);
                    age = min(age, max(1.0 / alpha, 4.0));
                    reason = rtgi_reset_adaptive;
                }
            }
        }
    }

    vec3 mean = mix(history.mean.rgb, raw.rgb, alpha);
    // Do not poison stationary frame variance with a confirmed lighting-step
    // residual; otherwise frame_sigma explodes on frame 3 and locks adaptation shut.
    float variance = reason == rtgi_reset_adaptive
        ? (1.0 - alpha) * history.stats.x
        : (1.0 - alpha) * (history.stats.x + alpha * sqr(delta));
    float q = sqr(1.0 - alpha) * history.stats.y + sqr(alpha);
    store_history(texel, even, vec4(mean, raw.a), vec4(variance, q, age, evidence),
                  reason, support, true, alpha);
}
