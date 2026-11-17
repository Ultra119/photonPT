// Display-only RTGI reconstruction. Temporal history is read-only here.

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
layout(rgba16f) writeonly uniform image2D rtgi_filter_inter_img;
#define filter_output rtgi_filter_inter_img
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
uniform sampler2D rtgi_filter_inter_sampler;
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
#include "/include/utility/geometry.glsl"

const vec4 no_filter_sample = vec4(0.0, 0.0, 0.0, -1.0);

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
    return texelFetch(rtgi_filter_inter_sampler, texel, 0);
#endif
#endif
}

#if RTGI_FILTER_PASS == 2 || RTGI_FILTER_PASS == 3
const vec2 vogel_disk[8] = vec2[8](
    vec2(0.250000, 0.000000), vec2(-0.319290, 0.292496),
    vec2(0.048872, -0.556877), vec2(0.402444, 0.524918),
    vec2(-0.738535, -0.130636), vec2(0.699605, -0.445028),
    vec2(-0.233999, 0.870485), vec2(-0.446271, -0.859268)
);
#else
const float gauss3_kernel[3] = float[3](0.25, 0.5, 0.25);
#endif

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
    float center_y = dot(center.rgb, luminance_weights);
#if RTGI_FILTER_PASS == 1
    bool copy_through = RTGI_SPATIAL_FALLBACK == 0 || age >= 3.0;
#else
    bool copy_through = center.a <= 0.001 + 0.02 * center_y;
#endif
    if (center.a < 0.0 || copy_through) {
        imageStore(filter_output, texel, center);
        return;
    }

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
    vec3 normal = mat3(gbufferModelView) * rtgi_guide_normal(guide);
    float footprint = rtgi_view_footprint(-center_view.z, grid.gi_active, gbufferProjection);
    float luma_floor = 0.004 + 0.06 * center_y;

#if RTGI_FILTER_PASS == 2 || RTGI_FILTER_PASS == 3
    #if RTGI_FILTER_PASS == 2
    const float world_cap = 1.5;
    const float texel_cap = 6.0;
    const mat2 rotation = mat2(1.0, 0.0, 0.0, 1.0);
    #else
    const float world_cap = 1.0;
    const float texel_cap = 4.0;
    const mat2 rotation = mat2(-0.5, 0.8660254, -0.8660254, -0.5);
    #endif
    float relative_se = center.a / (center_y + 0.04);
    float radius = min(world_cap, texel_cap * footprint)
        * mix(0.35, 1.0, smoothstep(0.03, 0.35, relative_se));
    float max_chord = min(world_cap, radius + footprint);
    mat3 basis = build_orthonormal_basis(normal);
    ivec2 visited[8];
    int visited_count = 0;
#else
    float grazing = max(abs(dot(normal, normalize(center_view))), 0.15);
    float max_chord = min(1.0, 0.1 + 3.0 * footprint / grazing);
#endif

    vec4 sum = 0.25 * center;
    float sum_weight = 0.25;
#if RTGI_FILTER_PASS == 1
    float local_error = 0.0;
    float local_weight = 0.25;
#endif

    for (int k = 0; k < 8; ++k) {
#if RTGI_FILTER_PASS == 2 || RTGI_FILTER_PASS == 3
        vec2 offset = rotation * vogel_disk[k];
        vec3 expected = center_view + radius * (offset.x * basis[0] + offset.y * basis[1]);
        vec4 clip = gbufferProjection * vec4(expected, 1.0);
        if (!rtgi_finite(clip) || clip.w <= 0.0 || abs(clip.z) > clip.w) {
            continue;
        }
        vec2 pixel = rtgi_raster_position(clip.xy / clip.w, grid.full_size, jitter);
        if (any(lessThan(pixel, vec2(0.0))) || any(greaterThanEqual(pixel, vec2(grid.full_active)))) {
            continue;
        }
        ivec2 tap = rtgi_cell(ivec2(pixel), grid);
        bool duplicate = tap == texel;
        for (int i = 0; i < visited_count; ++i) {
            duplicate = duplicate || visited[i] == tap;
        }
        if (duplicate) {
            continue;
        }
        visited[visited_count++] = tap;
        float kernel = 0.25 * exp(-dot(offset, offset));
#else
        int index = k + int(k >= 4); // Skip the center of the 3x3 kernel.
        ivec2 offset = ivec2(index % 3, index / 3);
        ivec2 tap = texel + offset - 1;
        if (!rtgi_in_bounds(tap, active_size)) {
            continue;
        }
        float kernel = gauss3_kernel[offset.x] * gauss3_kernel[offset.y];
#endif
        uvec4 tap_guide = even ? texelFetch(rtgi_guide_even_sampler, tap, 0)
                               : texelFetch(rtgi_guide_odd_sampler, tap, 0);
        vec3 tap_view;
        if (!rtgi_guide_matches(tap_guide, material)
            || !rtgi_view_from_guide(tap_guide, grid.full_size, jitter, gbufferProjection, tap_view)) {
            continue;
        }
        vec3 tap_normal = mat3(gbufferModelView) * rtgi_guide_normal(tap_guide);
        float weight = kernel * rtgi_geometry_weight(tap_view - center_view, normal,
                                                     tap_normal, footprint, max_chord);
        if (weight <= 0.0) {
            continue;
        }

        float tap_age;
        vec4 sample_value = read_filter_input(tap, even, tap_age);
        if (sample_value.a < 0.0) {
            continue;
        }
        float delta = dot(sample_value.rgb, luminance_weights) - center_y;
        float luminance_scale = RTGI_FILTER_PHI * (center.a + sample_value.a + luma_floor);
#if RTGI_FILTER_PASS == 1
        local_error += weight * sqr(delta);
        local_weight += weight;
        luminance_scale += sqrt(sqr(delta) / age) * 0.5;
#endif
        weight *= exp(-abs(delta) / luminance_scale);
        sum += weight * sample_value;
        sum_weight += weight;
    }

    vec4 result = sum / sum_weight;
#if RTGI_FILTER_PASS == 1
    // Bootstrap display uncertainty from distinct same-surface neighbors.
    // Subsequent passes propagate SE conservatively for shared support.
    result.a = max(result.a, sqrt(local_error / (local_weight * age)));
#endif
    imageStore(filter_output, texel, result);
}
