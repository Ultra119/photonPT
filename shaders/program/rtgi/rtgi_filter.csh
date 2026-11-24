// Display-only RTGI reconstruction via 4-pass variance-guided multi-scale
// à-trous wavelet filtering. Temporal history is read-only here.

#include "/include/global.glsl"

#if RTGI_RESOLUTION == 0
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif RTGI_RESOLUTION == 2
const vec2 workGroupsRender = vec2(1.0, 1.0);
#else
const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif
layout(local_size_x = 8, local_size_y = 8) in;

#ifndef RTGI_FILTER_PASS
#define RTGI_FILTER_PASS 1
#endif

#if RTGI_FILTER_PASS == 1 || RTGI_FILTER_PASS == 3
#if GI_DEBUG == GI_DEBUG_RAW
layout(rgba16f) writeonly uniform image2D rtgi_filter_inter_img;
#define filter_output rtgi_filter_inter_img
#else
layout(rgba16f) writeonly uniform image2D rtgi_raw_img;
#define filter_output rtgi_raw_img
#endif
#else
layout(rgba16f) writeonly uniform image2D rtgi_filtered_img;
#define filter_output rtgi_filtered_img
#endif

#if RTGI_FILTER_PASS == 1
uniform sampler2D rtgi_accum_sampler;
uniform sampler2D rtgi_history_sampler;
uniform sampler2D rtgi_stats_even_sampler;
uniform sampler2D rtgi_stats_odd_sampler;
#elif RTGI_FILTER_PASS == 3
uniform sampler2D rtgi_filtered_sampler;
#else
#if GI_DEBUG == GI_DEBUG_RAW
uniform sampler2D rtgi_filter_inter_sampler;
#define filter_inter_sampler rtgi_filter_inter_sampler
#else
uniform sampler2D rtgi_raw_sampler;
#define filter_inter_sampler rtgi_raw_sampler
#endif
#endif

uniform usampler2D rtgi_frame_state_sampler;
uniform usampler2D rtgi_guide_even_sampler;
uniform usampler2D rtgi_guide_odd_sampler;

uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;
uniform int frameCounter;

#include "/include/lighting/rt/rtgi_surface.glsl"
#include "/include/lighting/rt/pt_state.glsl"
#include "/include/utility/color.glsl"

const vec4 no_filter_sample = vec4(0.0, 0.0, 0.0, -1.0);
const float gauss3_kernel[3] = float[3](0.25, 0.5, 0.25);

vec4 read_filter_input(ivec2 texel, bool even, out float age) {
#if RTGI_FILTER_PASS == 1
    vec4 stats = even ? texelFetch(rtgi_stats_even_sampler, texel, 0)
                      : texelFetch(rtgi_stats_odd_sampler, texel, 0);
    age = stats.z;
    if (age < 1.0) {
        return no_filter_sample;
    }
    vec3 mean = even ? texelFetch(rtgi_accum_sampler, texel, 0).rgb
                     : texelFetch(rtgi_history_sampler, texel, 0).rgb;
    return vec4(mean, sqrt(rtgi_mean_variance(stats)));
#else
    age = 0.0;
#if RTGI_FILTER_PASS == 3
    return texelFetch(rtgi_filtered_sampler, texel, 0);
#else
    return texelFetch(filter_inter_sampler, texel, 0);
#endif
#endif
}

int rtgi_atrous_step() {
#if RTGI_FILTER_PASS == 1
    return 1;
#elif RTGI_FILTER_PASS == 2
    return 2;
#elif RTGI_FILTER_PASS == 3
    return 4;
#else
    #if RTGI_RESOLUTION == 0
    return 4;
    #else
    return 8;
    #endif
#endif
}

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(filter_output);
    if (!rtgi_in_bounds(texel, size)) {
        return;
    }

    int bank = frameCounter & 1;
    bool even = bank == 0;
    ivec2 active_size = rtgi_active_size(size);
    if (!rtgi_in_bounds(texel, active_size)) {
        imageStore(filter_output, texel, no_filter_sample);
        return;
    }

    float age;
    vec4 center = read_filter_input(texel, even, age);
    bool missing_center = center.a < 0.0;
    float center_y = missing_center ? 0.0 : dot(center.rgb, luminance_weights);
#if RTGI_FILTER_PASS == 1
    if (missing_center && RTGI_SPATIAL_FALLBACK == 0) {
        imageStore(filter_output, texel, no_filter_sample);
        return;
    }
#else
    if (missing_center) {
        imageStore(filter_output, texel, no_filter_sample);
        return;
    }
#endif

    uvec4 guide = even ? texelFetch(rtgi_guide_even_sampler, texel, 0)
                       : texelFetch(rtgi_guide_odd_sampler, texel, 0);
    uvec4 frame = texelFetch(rtgi_frame_state_sampler, ivec2(0, bank), 0);
    uvec4 full_layout = texelFetch(rtgi_frame_state_sampler, ivec2(1, bank), 0);
    RtgiGrid grid = RtgiGrid(ivec2(full_layout.xy), ivec2(full_layout.zw), active_size);
    vec2 jitter = uintBitsToFloat(frame.zw);
    vec3 center_view;
    if (!rtgi_view_from_guide(guide, grid.full_size, jitter, gbufferProjection, center_view)) {
        imageStore(filter_output, texel, no_filter_sample);
        return;
    }
    uint material = rtgi_guide_material(guide);
    bool is_foliage = rtgi_is_foliage(material);
    vec3 normal = mat3(gbufferModelView) * rtgi_guide_normal(guide);
    float footprint = rtgi_view_footprint(-center_view.z, grid.gi_active, gbufferProjection);
    float grazing = max(abs(dot(normal, normalize(center_view))), 0.02);

    int step_px = rtgi_atrous_step();
    // Bound solid-surface à-trous chord so extreme grazing views at 10+ blocks
    // keep 1-texel grazing support without smearing across 4-6 blocks into blobs.
    float max_chord = min(2.0, 0.25 + float(step_px) * 1.5 * footprint / grazing);
    float foliage_max_dist = min(1.5, 0.40 + 0.35 * float(step_px) * footprint);

#if RTGI_FILTER_PASS == 1
    // Pass 1 estimates same-surface 3x3 spatial variance for ALL pixels (including
    // dark/zero-variance and freshly disoccluded texels) before filtering so
    // passes 2-4 never stall on underestimated temporal variance.
    vec4 tap_val[8];
    float tap_geom[8];
    float tap_y[8];
    float mom_w = missing_center ? 0.0 : 0.25;
    float mom_y = missing_center ? 0.0 : 0.25 * center_y;
    float mom_y2 = missing_center ? 0.0 : 0.25 * sqr(center_y);
    float nbr_se_sum = missing_center ? 0.0 : 0.25 * center.a;
    float nbr_w_sum = 0.0;
    float nbr_y_sum = 0.0;
    float nbr_max_y = 0.0;

    for (int k = 0; k < 8; ++k) {
        tap_geom[k] = 0.0;
        tap_val[k] = no_filter_sample;
        tap_y[k] = 0.0;
        int index = k + int(k >= 4);
        ivec2 offset = ivec2(index % 3, index / 3);
        ivec2 tap = texel + offset - 1;
        if (!rtgi_in_bounds(tap, active_size)) {
            continue;
        }
        uvec4 tap_guide = even ? texelFetch(rtgi_guide_even_sampler, tap, 0)
                               : texelFetch(rtgi_guide_odd_sampler, tap, 0);
        vec3 tap_view;
        if (!rtgi_guide_matches(tap_guide, material)
            || !rtgi_view_from_guide(tap_guide, grid.full_size, jitter, gbufferProjection, tap_view)) {
            continue;
        }
        vec3 tap_normal = mat3(gbufferModelView) * rtgi_guide_normal(tap_guide);
        float geom_w = is_foliage
            ? rtgi_foliage_weight(tap_view - center_view, normal, tap_normal, material, foliage_max_dist)
            : rtgi_geometry_weight(tap_view - center_view, normal, tap_normal, footprint, max_chord);
        float gw = gauss3_kernel[offset.x] * gauss3_kernel[offset.y] * geom_w;
        if (gw <= 0.0) {
            continue;
        }
        float tap_age;
        vec4 sv = read_filter_input(tap, even, tap_age);
        if (sv.a < 0.0) {
            continue;
        }
        float sy = dot(sv.rgb, luminance_weights);
        tap_val[k] = sv;
        tap_geom[k] = gw;
        tap_y[k] = sy;
        nbr_w_sum += gw;
        nbr_y_sum += gw * sy;
        nbr_max_y = max(nbr_max_y, sy);
        mom_w += gw;
        mom_y += gw * sy;
        mom_y2 += gw * sqr(sy);
        nbr_se_sum += gw * sv.a;
    }

    if (mom_w <= 0.0) {
        imageStore(filter_output, texel, no_filter_sample);
        return;
    }

    // 3x3 same-surface firefly despeckle: an isolated 1-pixel outlier would
    // otherwise reject all dark neighbors via exp(-delta / luma_scale) and
    // survive Passes 1-4 as a bright glowworm or smear into a blob on edges.
    // Real lighting edges have bright neighbors so center_y <= nbr_max_y.
    float nbr_mean_y = nbr_w_sum > 1e-4 ? (nbr_y_sum / nbr_w_sum) : center_y;
    if (!missing_center && nbr_w_sum > 0.10) {
        float ff_cap = max(nbr_max_y * 1.35, nbr_mean_y * 2.2) + 0.06;
        if (center_y > ff_cap) {
            float old_y = center_y;
            center.rgb *= ff_cap / max(old_y, 1e-5);
            center_y = ff_cap;
            mom_y += 0.25 * (center_y - old_y);
            mom_y2 += 0.25 * (sqr(center_y) - sqr(old_y));
        }
    }

    float local_mean = mom_y / mom_w;
    float local_var = max(0.0, mom_y2 / mom_w - sqr(local_mean));
    float spatial_se = sqrt(local_var / max(age, 1.0));
    float young_factor = missing_center ? 1.0 : (1.0 - smoothstep(2.0, 12.0, age));
    float young_prior = (0.22 * local_mean + 0.003) * inversesqrt(max(age, 1.0)) * young_factor;
    float smooth_center_se = missing_center
        ? max(nbr_se_sum / mom_w, 0.25 * local_mean + 0.05)
        : max(max(center.a, max(0.55 * nbr_se_sum / mom_w, young_prior)),
              mix(0.60, 1.0, young_factor) * spatial_se);

    vec3 sum_rgb = missing_center ? vec3(0.0) : 0.25 * center.rgb;
    float sum_w = missing_center ? 0.0 : 0.25;
    float sum_w2_var = missing_center ? 0.0 : sqr(0.25) * sqr(smooth_center_se);
    float sum_w_var = missing_center ? 0.0 : 0.25 * sqr(smooth_center_se);

    float tap_ff_cap = max(max(center_y, nbr_mean_y) * 2.5, 0.15) + 0.12;
    for (int k = 0; k < 8; ++k) {
        float gw = tap_geom[k];
        if (gw <= 0.0) {
            continue;
        }
        vec4 sv = tap_val[k];
        float sy = tap_y[k];
        if (!missing_center && nbr_w_sum > 0.20 && sy > tap_ff_cap) {
            sv.rgb *= tap_ff_cap / max(sy, 1e-5);
            sy = tap_ff_cap;
        }
        float w = gw;
        if (!missing_center) {
            float delta = abs(sy - center_y);
            float sym_se = 0.5 * (smooth_center_se + max(sv.a, 0.5 * spatial_se));
            float luma_scale = RTGI_FILTER_PHI * sym_se
                + 0.002 + 0.05 * min(center_y, sy)
                + 1.5 * young_factor * spatial_se;
            w *= exp(-delta / max(luma_scale, 1e-5));
        }
        float sv_se = max(sv.a, mix(0.5, 1.0, young_factor) * spatial_se);
        sum_rgb += w * sv.rgb;
        sum_w += w;
        sum_w2_var += sqr(w) * sqr(sv_se);
        sum_w_var += w * sqr(sv_se);
    }

    if (sum_w <= 0.0) {
        imageStore(filter_output, texel, no_filter_sample);
        return;
    }
    vec3 out_rgb = sum_rgb / sum_w;
    // Conservative variance propagation accounting for correlated ReSTIR taps,
    // floored by the young/spatial uncertainty bootstrap for passes 2-4.
    float prop_var = mix(sum_w2_var / sqr(sum_w), sum_w_var / sum_w, 0.70);
    float out_se = max(sqrt(max(prop_var, 0.0)), max(young_prior, mix(0.60, 0.95, young_factor) * spatial_se));
    vec4 result = vec4(out_rgb, out_se);
#else
    const float center_k = 0.25;
    vec3 sum_rgb = center_k * center.rgb;
    float sum_w = center_k;
    float center_var = sqr(center.a);
    float sum_w2_var = sqr(center_k) * center_var;
    float sum_w_var = center_k * center_var;

    for (int k = 0; k < 8; ++k) {
        int index = k + int(k >= 4); // Skip the center of the 3x3 à-trous kernel.
        ivec2 offset = ivec2(index % 3, index / 3);
        ivec2 tap = texel + (offset - 1) * step_px;
        if (!rtgi_in_bounds(tap, active_size)) {
            continue;
        }
        uvec4 tap_guide = even ? texelFetch(rtgi_guide_even_sampler, tap, 0)
                               : texelFetch(rtgi_guide_odd_sampler, tap, 0);
        vec3 tap_view;
        if (!rtgi_guide_matches(tap_guide, material)
            || !rtgi_view_from_guide(tap_guide, grid.full_size, jitter, gbufferProjection, tap_view)) {
            continue;
        }
        vec3 tap_normal = mat3(gbufferModelView) * rtgi_guide_normal(tap_guide);
        float kernel = gauss3_kernel[offset.x] * gauss3_kernel[offset.y];
        float geom_w = is_foliage
            ? rtgi_foliage_weight(tap_view - center_view, normal, tap_normal, material, foliage_max_dist)
            : rtgi_geometry_weight(tap_view - center_view, normal, tap_normal, footprint, max_chord);
        float weight = kernel * geom_w;
        if (weight <= 0.0) {
            continue;
        }

        float tap_age;
        vec4 sample_value = read_filter_input(tap, even, tap_age);
        if (sample_value.a < 0.0) {
            continue;
        }
        float sample_y = dot(sample_value.rgb, luminance_weights);
        float delta = abs(sample_y - center_y);
        float sym_se = 0.5 * (center.a + sample_value.a);
        float luminance_scale = RTGI_FILTER_PHI * sym_se
            + 0.002 + 0.05 * min(center_y, sample_y);
        weight *= exp(-delta / max(luminance_scale, 1e-5));
        float sample_var = sqr(sample_value.a);
        sum_rgb += weight * sample_value.rgb;
        sum_w += weight;
        sum_w2_var += sqr(weight) * sample_var;
        sum_w_var += weight * sample_var;
    }

    if (sum_w <= 0.0) {
        imageStore(filter_output, texel, no_filter_sample);
        return;
    }
    float prop_var = mix(sum_w2_var / sqr(sum_w), sum_w_var / sum_w, 0.70);
    vec4 result = vec4(sum_rgb / sum_w, sqrt(max(prop_var, 0.0)));
#endif

    if (!rtgi_finite(result)) {
        result = no_filter_sample;
    } else if (result.a >= 0.0) {
        result = vec4(min(max0(result.rgb), vec3(65504.0)), min(result.a, 65504.0));
    }
    imageStore(filter_output, texel, result);
}
