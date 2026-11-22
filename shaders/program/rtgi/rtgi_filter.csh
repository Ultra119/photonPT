// ReBLUR-Style Self-Stabilizing World-Space Tangent-Plane GI Denoiser
//   Pass 1 (deferred3_a): Dense 3x3 bilateral pre-stabilizer + Karis outlier dampening + pooled SE
//   Pass 2 (deferred3_b): Self-stabilizing recurrent writeback + wide tangent-plane rotated Poisson disk
//   Pass 3 (deferred3_c): Medium tangent-plane rotated Poisson disk (decorrelated angle)
//   Pass 4 (deferred3_d): Dense 5x5 stride-1 bilateral polish (erases disk rotation grain)

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

#if RTGI_FILTER_PASS == 1
layout(rgba16f) writeonly uniform image2D rtgi_filter_inter_img;
uniform sampler2D rtgi_accum_sampler;
uniform sampler2D rtgi_history_sampler;
#elif RTGI_FILTER_PASS == 2
layout(rgba16f) writeonly uniform image2D rtgi_filtered_img;
layout(rgba16f) uniform image2D rtgi_accum_img;
layout(rgba16f) uniform image2D rtgi_history_img;
uniform sampler2D rtgi_filter_inter_sampler;
#elif RTGI_FILTER_PASS == 3
layout(rgba16f) writeonly uniform image2D rtgi_filter_inter_img;
uniform sampler2D rtgi_filtered_sampler;
#else
layout(rgba16f) writeonly uniform image2D rtgi_filtered_img;
uniform sampler2D rtgi_filter_inter_sampler;
#endif

uniform sampler2D rtgi_sky_accum_sampler;
uniform sampler2D rtgi_sky_history_sampler;
uniform sampler2D depthtex1;
uniform sampler2D colortex1; // gbuffer0
uniform sampler2D noisetex;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform vec3 cameraPosition;
uniform int frameCounter;

uniform vec2 view_res;
uniform vec2 view_pixel_size;

uniform float near;
uniform float far;
uniform vec2 taa_offset;

#include "/include/utility/encoding.glsl"
#include "/include/utility/geometry.glsl"
#include "/include/utility/space_conversion.glsl"
#include "/include/utility/color.glsl"
#include "/include/lighting/rt/pt_state.glsl"

#if RTGI_FILTER_PASS == 1
vec4 sample_accum_data(ivec2 t, bool ping) {
    return ping ? texelFetch(rtgi_accum_sampler, t, 0)
                : texelFetch(rtgi_history_sampler, t, 0);
}
#endif

vec4 sample_aux_data(ivec2 t, bool ping) {
    return ping ? texelFetch(rtgi_sky_accum_sampler, t, 0)
                : texelFetch(rtgi_sky_history_sampler, t, 0);
}

const float gauss3_kernel[3] = float[3](0.25, 0.5, 0.25);

// 12-tap isotropic Vogel spiral disk inside the unit circle
const vec2 vogel_disk_12[12] = vec2[12](
    vec2( 0.204124,  0.000000),
    vec2(-0.260662,  0.238509),
    vec2( 0.048048, -0.454032),
    vec2( 0.354729,  0.379799),
    vec2(-0.569820, -0.118824),
    vec2( 0.230927, -0.609515),
    vec2( 0.335224,  0.630519),
    vec2(-0.752572,  0.145020),
    vec2( 0.525620, -0.604259),
    vec2( 0.052882,  0.846926),
    vec2(-0.781962, -0.440894),
    vec2( 0.858847, -0.312272)
);

#define PHI_COLOUR RTGI_FILTER_PHI

void store_filter_output(ivec2 texel, vec4 val) {
#if RTGI_FILTER_PASS == 1 || RTGI_FILTER_PASS == 3
    imageStore(rtgi_filter_inter_img, texel, val);
#else
    imageStore(rtgi_filtered_img, texel, val);
#endif
}

void main() {
#if RTGI_RESOLUTION == 0
    const float render_scale = 0.25;
#elif RTGI_RESOLUTION == 2
    const float render_scale = 1.0;
#else
    const float render_scale = 0.5;
#endif

    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 target_res = ivec2(view_res * render_scale);

    if (texel.x >= target_res.x || texel.y >= target_res.y) {
        return;
    }

    vec2 target_pixel_size = 1.0 / vec2(target_res);
    vec2 center_uv = (vec2(texel) + 0.5) * target_pixel_size;
    ivec2 active_res = max(ivec2(1), ivec2(vec2(target_res) * taau_render_scale));

    if (taau_uv_outside_viewport(center_uv)) {
        store_filter_output(texel, vec4(0.0));
        return;
    }

    ivec2 center_full = ivec2(center_uv * view_res);
    float center_depth = texelFetch(depthtex1, center_full, 0).x;
    if (center_depth >= 1.0) {
        store_filter_output(texel, vec4(0.0));
        return;
    }

    bool ping = (frameCounter & 1) == 0;

    vec3 center_screen_pos = vec3(taau_projection_uv(center_uv), center_depth);
    vec3 center_view_pos = screen_to_view_space(center_screen_pos, true);
    float center_dist = -center_view_pos.z;

    vec4 center_aux = sample_aux_data(texel, ping);
    vec3 center_world_normal = unpack_oct_normal_f16(center_aux.b);
    vec3 center_view_normal = mat3(gbufferModelView) * center_world_normal;
    float cos_view = max(abs(center_view_normal.z), 0.15);

    vec2 center_hc = unpack_hit_count(center_aux.g);
    float center_hit_norm = clamp01(center_hc.x);
    float accum_count = max(center_hc.y, 1.0);

    float texel_world_size = (2.0 * center_dist)
        / max(gbufferProjection[1][1] * float(active_res.y), 1.0);
    float sigma_z = max(center_dist * 0.025, 0.06) + 0.5 * texel_world_size;

#if RTGI_FILTER_PASS == 1
    // -------------------------------------------------------------------------
    // Pass 1: Dense 3x3 Bilateral Pre-Stabilizer + Pooled Standard Error (.a)
    // -------------------------------------------------------------------------
    vec4 center_sample = sample_accum_data(texel, ping);
    vec3 center_radiance = center_sample.rgb;
    float center_m1 = dot(center_radiance, luminance_weights);
    float center_var = max(0.0, center_aux.r - center_m1 * center_m1);
    float center_se = sqrt(center_var / accum_count);

    float relative_se = center_se / (center_m1 + 0.04);
    float sigma_n = mix(128.0, 12.0, clamp01(relative_se));
    float sigma_l = PHI_COLOUR * (center_se + max(0.08 * center_m1, 0.03) + 1e-4);
    float karis_strength = 1.0 - smoothstep(4.0, 24.0, accum_count);
    float max_chord_sq = sqr(3.0 * texel_world_size / cos_view + 0.25);

    vec3 sum_rad = vec3(0.0);
    float sum_se2 = 0.0;
    float sum_w = 0.0;

    for (int dy = -1; dy <= 1; ++dy) {
        float wy = gauss3_kernel[dy + 1];
        for (int dx = -1; dx <= 1; ++dx) {
            float w_spatial = gauss3_kernel[dx + 1] * wy;
            ivec2 n_texel = clamp(texel + ivec2(dx, dy), ivec2(0), active_res - 1);
            vec2 n_uv = (vec2(n_texel) + 0.5) * target_pixel_size;
            if (taau_uv_outside_viewport(n_uv)) {
                continue;
            }
            vec4 n_sample = sample_accum_data(n_texel, ping);
            float n_dist = n_sample.a;
            if (n_dist <= 0.05) {
                continue;
            }
            vec2 n_ndc = taau_projection_uv(n_uv) * 2.0 - 1.0;
            vec3 n_view_pos = vec3(
                (n_ndc.x + gbufferProjection[2][0]) * n_dist / max(gbufferProjection[0][0], 1e-4),
                (n_ndc.y + gbufferProjection[2][1]) * n_dist / max(gbufferProjection[1][1], 1e-4),
                -n_dist
            );
            vec4 n_aux = sample_aux_data(n_texel, ping);
            vec3 n_world_norm = unpack_oct_normal_f16(n_aux.b);
            vec3 n_view_norm = mat3(gbufferModelView) * n_world_norm;

            vec3 pos_diff = n_view_pos - center_view_pos;
            if (dot(pos_diff, pos_diff) > max_chord_sq) {
                continue;
            }
            float plane_dist = max(
                abs(dot(pos_diff, center_view_normal)),
                abs(dot(pos_diff, n_view_norm))
            );
            float w_depth = exp(-sqr(plane_dist / sigma_z));
            float w_norm = pow(max(0.0, dot(center_world_normal, n_world_norm)), sigma_n);

            float n_m1 = dot(n_sample.rgb, luminance_weights);
            float n_cnt = max(unpack_hit_count(n_aux.g).y, 1.0);
            float n_se2 = max(0.0, n_aux.r - n_m1 * n_m1) / n_cnt;

            float w_luma = exp(-abs(n_m1 - center_m1) / sigma_l);
            float w_karis = mix(1.0, 1.0 / (1.0 + n_m1 / (3.0 * center_m1 + 0.5)), karis_strength);

            float w = w_spatial * w_depth * w_norm * w_luma * w_karis;
            sum_rad += n_sample.rgb * w;
            sum_se2 += n_se2 * w;
            sum_w += w;
        }
    }

    vec3 out_rad = (sum_w > 1e-4) ? (sum_rad / sum_w) : center_radiance;
    float out_se = (sum_w > 1e-4) ? sqrt(max(sum_se2 / sum_w, 0.0)) : center_se;
    store_filter_output(texel, vec4(out_rad, out_se));

#elif RTGI_FILTER_PASS == 2 || RTGI_FILTER_PASS == 3
    // -------------------------------------------------------------------------
    // Pass 2 & 3: World-Space Tangent-Plane Rotated Vogel Disk (ReBLUR)
    // -------------------------------------------------------------------------
    #if RTGI_FILTER_PASS == 2
    vec4 center_sample = texelFetch(rtgi_filter_inter_sampler, texel, 0);
    // Self-stabilizing recurrent writeback for low-SPP pixels (0 weight once count >= 12)
    float recurrent_w = 0.55 * (1.0 - smoothstep(1.0, 12.0, accum_count));
    if (recurrent_w > 1e-3) {
        if (ping) {
            vec4 prev_out = imageLoad(rtgi_accum_img, texel);
            imageStore(rtgi_accum_img, texel, vec4(mix(prev_out.rgb, center_sample.rgb, recurrent_w), prev_out.a));
        } else {
            vec4 prev_out = imageLoad(rtgi_history_img, texel);
            imageStore(rtgi_history_img, texel, vec4(mix(prev_out.rgb, center_sample.rgb, recurrent_w), prev_out.a));
        }
    }
    const float base_max_world = 2.6;
    const float max_texel_span = 24.0;
    const float pass_angle_offset = 0.0;
    #else
    vec4 center_sample = texelFetch(rtgi_filtered_sampler, texel, 0);
    const float base_max_world = 1.15;
    const float max_texel_span = 11.0;
    const float pass_angle_offset = 2.0943951;
    #endif

    vec3 center_radiance = center_sample.rgb;
    float center_se = max(center_sample.a, 0.0);
    float center_m1 = dot(center_radiance, luminance_weights);
    float relative_se = center_se / (center_m1 + 0.04);

    float sigma_n = mix(128.0, 10.0, clamp01(relative_se));
    float sigma_l = PHI_COLOUR * (center_se + max(0.06 * center_m1, 0.025) + 1e-4);

    // Hit-distance contact sharpening + convergence radius tightening
    float contact_scale = mix(0.12, 1.0, smoothstep(0.02, 0.75, center_hit_norm));
    float conv_t = smoothstep(4.0, float(RTGI_SPP_CAP) * 0.75, accum_count)
        * (1.0 - smoothstep(0.06, 0.25, relative_se));
    float desired_world_radius = base_max_world * contact_scale * mix(1.0, 0.30, conv_t);
    float radius_world = clamp(
        desired_world_radius,
        1.25 * texel_world_size,
        max_texel_span * texel_world_size
    );
    float max_chord_sq = sqr(1.35 * radius_world + texel_world_size / cos_view + 0.15);

    mat3 plane_tbn = build_orthonormal_basis(center_view_normal);
    float blue = texelFetch(noisetex, texel & 511, 0).b;
    float rot_angle = fract(blue + float(frameCounter & 4095) * 0.6180339887) * 6.28318530718
        + pass_angle_offset;
    vec2 cs = vec2(cos(rot_angle), sin(rot_angle));
    mat2 rot = mat2(cs.x, -cs.y, cs.y, cs.x);

    vec3 sum_rad = center_radiance * 0.25;
    float sum_se = center_se * 0.25;
    float sum_w = 0.25;

    for (int k = 0; k < 12; ++k) {
        vec2 d_local = rot * vogel_disk_12[k];
        vec3 expected_view_pos = center_view_pos
            + radius_world * (d_local.x * plane_tbn[0] + d_local.y * plane_tbn[1]);
        if (expected_view_pos.z >= -0.05) {
            continue;
        }

        vec2 tap_proj_uv = view_to_screen_space(expected_view_pos, true).xy;
        vec2 tap_fb_uv = taau_framebuffer_uv(tap_proj_uv);
        if (clamp01(tap_proj_uv) != tap_proj_uv || taau_uv_outside_viewport(tap_fb_uv)) {
            continue;
        }

        ivec2 n_texel = clamp(ivec2(tap_fb_uv * vec2(target_res)), ivec2(0), active_res - 1);
        if (n_texel == texel) {
            continue;
        }

        vec2 n_uv = (vec2(n_texel) + 0.5) * target_pixel_size;
        ivec2 n_full = clamp(ivec2(n_uv * view_res), ivec2(0), ivec2(view_res) - 1);
        float n_depth = texelFetch(depthtex1, n_full, 0).x;
        if (n_depth >= 1.0) {
            continue;
        }

        vec3 n_view_pos = screen_to_view_space(vec3(taau_projection_uv(n_uv), n_depth), true);
        vec3 pos_diff = n_view_pos - center_view_pos;
        if (dot(pos_diff, pos_diff) > max_chord_sq) {
            continue;
        }

        #if RTGI_FILTER_PASS == 2
        vec4 n_sample = texelFetch(rtgi_filter_inter_sampler, n_texel, 0);
        #else
        vec4 n_sample = texelFetch(rtgi_filtered_sampler, n_texel, 0);
        #endif
        vec4 n_aux = sample_aux_data(n_texel, ping);
        vec3 n_world_norm = unpack_oct_normal_f16(n_aux.b);
        vec3 n_view_norm = mat3(gbufferModelView) * n_world_norm;

        float plane_dist = max(
            abs(dot(pos_diff, center_view_normal)),
            abs(dot(pos_diff, n_view_norm))
        );
        float w_depth = exp(-sqr(plane_dist / sigma_z));
        float w_norm = pow(max(0.0, dot(center_world_normal, n_world_norm)), sigma_n);

        float n_m1 = dot(n_sample.rgb, luminance_weights);
        float w_luma = exp(-abs(n_m1 - center_m1) / sigma_l);
        float w_kernel = exp(-0.85 * dot(vogel_disk_12[k], vogel_disk_12[k]));

        float w = w_kernel * w_depth * w_norm * w_luma;
        sum_rad += n_sample.rgb * w;
        sum_se += max(n_sample.a, 0.0) * w * 0.85;
        sum_w += w;
    }

    float inv_w = 1.0 / max(sum_w, 1e-4);
    store_filter_output(texel, vec4(sum_rad * inv_w, sum_se * inv_w));

#else
    // -------------------------------------------------------------------------
    // Pass 4: Dense 3x3 Stride-1 Bilateral Polish (removes disk rotation grain)
    // -------------------------------------------------------------------------
    vec4 center_sample = texelFetch(rtgi_filter_inter_sampler, texel, 0);
    vec3 center_radiance = center_sample.rgb;
    float center_se = max(center_sample.a, 0.0);
    float center_m1 = dot(center_radiance, luminance_weights);
    float relative_se = center_se / (center_m1 + 0.04);

    float sigma_n = mix(128.0, 16.0, clamp01(relative_se));
    float sigma_l = PHI_COLOUR * (center_se + max(0.05 * center_m1, 0.02) + 1e-4);
    float max_chord_sq = sqr(3.0 * texel_world_size / cos_view + 0.25);

    vec3 sum_rad = vec3(0.0);
    float sum_se = 0.0;
    float sum_w = 0.0;

    for (int dy = -1; dy <= 1; ++dy) {
        float wy = gauss3_kernel[dy + 1];
        for (int dx = -1; dx <= 1; ++dx) {
            float w_spatial = gauss3_kernel[dx + 1] * wy;
            ivec2 n_texel = clamp(texel + ivec2(dx, dy), ivec2(0), active_res - 1);
            vec2 n_uv = (vec2(n_texel) + 0.5) * target_pixel_size;
            if (taau_uv_outside_viewport(n_uv)) {
                continue;
            }
            ivec2 n_full = ivec2(n_uv * view_res);
            float n_depth = texelFetch(depthtex1, n_full, 0).x;
            if (n_depth >= 1.0) {
                continue;
            }

            vec3 n_view_pos = screen_to_view_space(vec3(taau_projection_uv(n_uv), n_depth), true);
            vec3 pos_diff = n_view_pos - center_view_pos;
            if (dot(pos_diff, pos_diff) > max_chord_sq) {
                continue;
            }

            vec4 n_sample = texelFetch(rtgi_filter_inter_sampler, n_texel, 0);
            vec4 n_aux = sample_aux_data(n_texel, ping);
            vec3 n_world_norm = unpack_oct_normal_f16(n_aux.b);
            vec3 n_view_norm = mat3(gbufferModelView) * n_world_norm;

            float plane_dist = max(
                abs(dot(pos_diff, center_view_normal)),
                abs(dot(pos_diff, n_view_norm))
            );
            float w_depth = exp(-sqr(plane_dist / sigma_z));
            float w_norm = pow(max(0.0, dot(center_world_normal, n_world_norm)), sigma_n);

            float n_m1 = dot(n_sample.rgb, luminance_weights);
            float w_luma = exp(-abs(n_m1 - center_m1) / sigma_l);

            float w = w_spatial * w_depth * w_norm * w_luma;
            sum_rad += n_sample.rgb * w;
            sum_se += max(n_sample.a, 0.0) * w;
            sum_w += w;
        }
    }

    vec3 filtered = (sum_w > 1e-4) ? (sum_rad / sum_w) : center_radiance;
    float filtered_se = (sum_w > 1e-4) ? (sum_se / sum_w) : center_se;
    store_filter_output(texel, vec4(filtered, filtered_se));
#endif
}
