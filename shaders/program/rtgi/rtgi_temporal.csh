// SVGF Temporal Reconstruction with Disocclusion & Moment Tracking

#include "/include/global.glsl"

#if RTGI_RESOLUTION == 0
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif RTGI_RESOLUTION == 2
const vec2 workGroupsRender = vec2(1.0, 1.0);
#else
const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba16f) writeonly uniform image2D rtgi_history_img;
layout(rgba16f) writeonly uniform image2D rtgi_accum_img;
layout(rgba16f) writeonly uniform image2D rtgi_sky_history_img;
layout(rgba16f) writeonly uniform image2D rtgi_sky_accum_img;

layout(rgba16f) writeonly uniform image2D rtgi_state_img;

uniform sampler2D rtgi_raw_sampler;
uniform sampler2D rtgi_history_sampler;
uniform sampler2D rtgi_accum_sampler;
uniform sampler2D rtgi_sky_history_sampler;
uniform sampler2D rtgi_sky_accum_sampler;
uniform sampler2D rtgi_state_sampler;

// Declared only when the diagnostic view is active (see rtgi_trace.csh).
#if GI_DEBUG == GI_DEBUG_REJECT
layout(rgba8) writeonly uniform image2D rtgi_reject_img;
#endif
uniform sampler2D depthtex1;
uniform sampler2D colortex1; // gbuffer0

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform int frameCounter;

uniform vec2 view_res;
uniform vec2 view_pixel_size;

uniform float near;
uniform float far;
uniform vec2 taa_offset;

#define TEMPORAL_REPROJECTION
#include "/include/utility/encoding.glsl"
#include "/include/utility/space_conversion.glsl"
#include "/include/utility/color.glsl"
#include "/include/lighting/rt/pt_state.glsl"

// Every exit path writes the state image: a pixel that leaves the accumulation
// must still say why, or the debug view reads whatever the previous frame left
// in a texel nobody owns.
void write_state(ivec2 texel, float reason, float reactive, float valid_fraction, float converged) {
    imageStore(
        rtgi_state_img,
        texel,
        pt_state_pack(reason, reactive, valid_fraction, converged)
    );
}

void write_zero(ivec2 texel, bool ping, float reason) {
    if (ping) {
        imageStore(rtgi_accum_img, texel, vec4(0.0));
        imageStore(rtgi_sky_accum_img, texel, vec4(0.0));
    } else {
        imageStore(rtgi_history_img, texel, vec4(0.0));
        imageStore(rtgi_sky_history_img, texel, vec4(0.0));
    }
    write_state(texel, reason, 1.0, 0.0, 0.0);
}

// Spatial preaccumulation fallback for disoccluded pixels:
// Tight coplanar 5x5 tangent-plane gather capped at PT_SPATIAL_MAX_WEIGHT (3.0)
// so newly revealed strips lose raw 1-SPP speckle without dragging distant light
// across shadow boundaries or starving the spatial denoiser.
void pt_spatial_fallback(
    ivec2 texel,
    ivec2 target_res,
    ivec2 active_res,
    float current_dist,
    vec3 current_normal,
    vec3 current_view_pos,
    out vec4 radiance_and_hit,
    out float second_moment,
    out float weight
) {
    vec4 sum = vec4(0.0);
    float sum_lum2 = 0.0;
    float sum_weight = 0.0;

    vec3 current_view_norm = mat3(gbufferModelView) * current_normal;
    float plane_tolerance = max(current_dist * 0.025, 0.08);
    float max_chord = max(current_dist * 0.12, 0.45) / max(abs(current_view_norm.z), 0.15);

    for (int sy = -2; sy <= 2; ++sy) {
        for (int sx = -2; sx <= 2; ++sx) {
            if (sx == 0 && sy == 0) {
                continue;
            }
            ivec2 n_coord = clamp(texel + ivec2(sx, sy), ivec2(0), active_res - 1);
            vec2 n_uv = (vec2(n_coord) + 0.5) / vec2(target_res);
            if (taau_uv_outside_viewport(n_uv)) {
                continue;
            }
            ivec2 n_full = ivec2(n_uv * view_res);

            float n_depth = texelFetch(depthtex1, n_full, 0).x;
            if (n_depth >= 1.0) {
                continue;
            }

            vec3 n_view_pos = screen_to_view_space(
                vec3(taau_projection_uv(n_uv), n_depth), true
            );

            vec3 n_normal = decode_unit_vector(
                unpack_unorm_2x8(texelFetch(colortex1, n_full, 0).z)
            );
            float w_normal = pow(max(dot(n_normal, current_normal), 0.0), 16.0);

            vec3 delta_view = n_view_pos - current_view_pos;
            float plane_dist = abs(dot(current_view_norm, delta_view));
            if (dot(delta_view, delta_view) > max_chord * max_chord) {
                continue;
            }
            float w_plane = exp(-sqr(plane_dist / plane_tolerance));
            float w_kernel = exp(-0.35 * float(sx * sx + sy * sy));

            float w = w_kernel * w_normal * w_plane;
            if (w < 1e-3) {
                continue;
            }

            vec4 n_raw = texelFetch(rtgi_raw_sampler, n_coord, 0);
            float n_lum = dot(n_raw.rgb, luminance_weights);
            sum += n_raw * w;
            sum_lum2 += n_lum * n_lum * w;
            sum_weight += w;
        }
    }

    float inv = 1.0 / max(sum_weight, 1e-4);
    radiance_and_hit = sum * inv;
    second_moment = sum_lum2 * inv;
    weight = sum_weight;
}

// Reconstruct previous-camera view-space position from history UV and metric depth.
vec3 reconstruct_prev_view_pos(vec2 fb_uv, float prev_dist) {
    vec2 prev_ndc = taau_projection_uv(fb_uv) * 2.0 - 1.0;
    return vec3(
        (prev_ndc.x + gbufferPreviousProjection[2][0]) * prev_dist
            / max(gbufferPreviousProjection[0][0], 1e-4),
        (prev_ndc.y + gbufferPreviousProjection[2][1]) * prev_dist
            / max(gbufferPreviousProjection[1][1], 1e-4),
        -prev_dist
    );
}

// Evaluate geometric consistency of a single integer history texel using 3D
// symmetric tangent-plane distance and bounded chord distance.
float evaluate_history_tap_geometry(
    ivec2 c,
    vec2 target_res_f,
    bool ping,
    vec3 current_world_normal,
    vec3 prev_camera_view_pos,
    vec3 prev_view_normal,
    float plane_tol,
    float max_chord_sq,
    out vec4 tap_sample,
    out vec4 tap_aux,
    out bool depth_ok,
    out bool normal_ok
) {
    tap_sample = ping ? texelFetch(rtgi_history_sampler, c, 0)
                      : texelFetch(rtgi_accum_sampler, c, 0);
    tap_aux = ping ? texelFetch(rtgi_sky_history_sampler, c, 0)
                   : texelFetch(rtgi_sky_accum_sampler, c, 0);

    float tap_dist = tap_sample.a;
    depth_ok = false;
    normal_ok = false;
    if (tap_dist <= 0.05) {
        return 0.0;
    }

    vec3 tap_world_normal = unpack_oct_normal_f16(tap_aux.b);
    float normal_dot = dot(current_world_normal, tap_world_normal);
    normal_ok = normal_dot > 0.82;

    vec2 tap_fb_uv = (vec2(c) + 0.5) / target_res_f;
    vec3 tap_prev_view = reconstruct_prev_view_pos(tap_fb_uv, tap_dist);
    vec3 tap_prev_view_norm = mat3(gbufferPreviousModelView) * tap_world_normal;
    vec3 delta_prev = tap_prev_view - prev_camera_view_pos;

    float plane_dist = max(
        abs(dot(delta_prev, prev_view_normal)),
        abs(dot(delta_prev, tap_prev_view_norm))
    );
    float chord_sq = dot(delta_prev, delta_prev);
    depth_ok = (plane_dist < plane_tol * 2.0) && (chord_sq < max_chord_sq);

    if (!depth_ok || !normal_ok) {
        return 0.0;
    }

    float w_plane = clamp01(1.0 - max(0.0, plane_dist - plane_tol * 0.75) / (plane_tol * 1.25));
    float w_norm = smoothstep(0.80, 0.96, normal_dot);
    return w_plane * w_norm;
}

// Catmull-Rom resampling weights for taps at -1, 0, +1, +2 relative to the
// sample point
vec4 catmull_rom_weights(float f) {
    float f2 = f * f;
    float f3 = f2 * f;
    return vec4(
        -0.5 * f3 + f2 - 0.5 * f,
         1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
         0.5 * f3 - 0.5 * f2
    );
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
    ivec2 active_res = max(ivec2(1), ivec2(vec2(target_res) * taau_render_scale));

    if (texel.x >= target_res.x || texel.y >= target_res.y) {
        return;
    }

    // uv is a full-framebuffer UV; see rtgi_trace.csh.
    vec2 uv = (vec2(texel) + 0.5) / vec2(target_res);
    bool ping = (frameCounter & 1) == 0;

    if (taau_uv_outside_viewport(uv)) {
        write_zero(texel, ping, PT_RESET_NO_SURFACE);
        return;
    }

    ivec2 full_texel = ivec2(uv * view_res);

    float depth = texelFetch(depthtex1, full_texel, 0).x;

    if (depth >= 1.0) {
        write_zero(texel, ping, PT_RESET_NO_SURFACE);
        return;
    }

    // The trace pass wrote the valid-path fraction of this frame's rays into
    // the state image's .b channel; carry it through and replace the other
    // three channels with this pass's verdict.
    float valid_path_fraction = pt_state_valid_fraction(
        texelFetch(rtgi_state_sampler, texel, 0)
    );

    // Current frame raw sample (.rgb = radiance, .a = normalized primary hit distance)
    vec4 raw_sample = texelFetch(rtgi_raw_sampler, texel, 0);
    vec3 current_radiance = raw_sample.rgb;
    float current_hit_norm = clamp01(raw_sample.a);
    float current_lum = dot(current_radiance, luminance_weights);

    // Current surface geometric data from G-buffer
    vec4 gbuf = texelFetch(colortex1, full_texel, 0);
    vec3 current_world_normal = decode_unit_vector(unpack_unorm_2x8(gbuf.z));

    // Extreme-outlier & finite sample guards
    const float absolute_radiance_cap = 32.0;
    if (current_lum > absolute_radiance_cap) {
        current_radiance *= absolute_radiance_cap / current_lum;
        current_lum = absolute_radiance_cap;
    }

    bool raw_finite = all(equal(current_radiance, current_radiance))
        && all(greaterThanEqual(current_radiance, vec3(0.0)))
        && (current_hit_norm == current_hit_norm);

    if (!raw_finite) {
        current_radiance = vec3(0.0);
        current_hit_norm = 1.0;
        current_lum = 0.0;
    }

    // Static-state detector
    PtStaticState static_state = pt_static_state(
        cameraPosition,
        previousCameraPosition,
        gbufferModelView,
        gbufferPreviousModelView,
        gbufferProjection,
        gbufferPreviousProjection
    );

    // Reconstruct position in view and scene space (handle_jitter = true because depth buffer is jittered)
    vec3 screen_pos = vec3(taau_projection_uv(uv), depth);
    vec3 view_pos = screen_to_view_space(screen_pos, true);
    float current_dist = -view_pos.z;
    vec3 scene_pos = view_to_scene_space(view_pos);
    vec3 current_view_norm = mat3(gbufferModelView) * current_world_normal;
    float cos_curr_view = max(abs(current_view_norm.z), 0.15);

    // Bilateral 3x3 raw neighbourhood statistics (ignores sky and cross-corner edges)
    vec3 raw_m1 = vec3(0.0);
    vec3 raw_m2 = vec3(0.0);
    float raw_wsum = 0.0;
    float raw_depth_tol = max(current_dist * 0.03, 0.08) / cos_curr_view;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            ivec2 n_coord = clamp(texel + ivec2(dx, dy), ivec2(0), active_res - 1);
            float w_nb = 1.0;
            if (dx != 0 || dy != 0) {
                vec2 n_uv = (vec2(n_coord) + 0.5) / vec2(target_res);
                if (taau_uv_outside_viewport(n_uv)) {
                    continue;
                }
                ivec2 n_full = ivec2(n_uv * view_res);
                float n_depth = texelFetch(depthtex1, n_full, 0).x;
                if (n_depth >= 1.0) {
                    continue;
                }
                float n_dist = screen_to_view_space_depth(gbufferProjectionInverse, n_depth);
                vec3 n_norm = decode_unit_vector(unpack_unorm_2x8(texelFetch(colortex1, n_full, 0).z));
                w_nb = exp(-abs(n_dist - current_dist) / raw_depth_tol)
                    * pow(max(dot(n_norm, current_world_normal), 0.0), 8.0);
                if (w_nb < 1e-2) {
                    continue;
                }
            }
            vec3 n_val = texelFetch(rtgi_raw_sampler, n_coord, 0).rgb;
            raw_m1 += n_val * w_nb;
            raw_m2 += n_val * n_val * w_nb;
            raw_wsum += w_nb;
        }
    }
    raw_m1 /= max(raw_wsum, 1e-4);
    raw_m2 /= max(raw_wsum, 1e-4);
    vec3 raw_sigma = sqrt(max(vec3(0.0), raw_m2 - raw_m1 * raw_m1));

    vec3 prev_camera_view_pos = transform(
        gbufferPreviousModelView,
        scene_pos + (cameraPosition - previousCameraPosition)
    );
    vec3 prev_view_normal = mat3(gbufferPreviousModelView) * current_world_normal;

    // Reproject to previous frame projection space
    vec3 prev_screen_pos = reproject_scene_space(scene_pos, false, false);
    vec2 prev_framebuffer_uv = taau_framebuffer_uv(prev_screen_pos.xy);
    bool inside_prev_screen = clamp01(prev_screen_pos.xy) == prev_screen_pos.xy
        && !taau_uv_outside_viewport(prev_framebuffer_uv);

#if GI_DEBUG == GI_DEBUG_REJECT
    float history_accepted = 0.0;
    float history_available = inside_prev_screen ? 1.0 : 0.0;
#endif

    float current_spp = float(RTGI_RAY_COUNT);
    float sample_count = current_spp;
    float alpha = 1.0;
    float reset_reason = PT_RESET_FIRST;
    float rejection = 0.0;
    vec3 accumulated_radiance = current_radiance;
    float accumulated_hit = current_hit_norm;
    float fast_m1 = current_lum;
    float m2 = current_lum * current_lum
        + dot(raw_sigma * raw_sigma, luminance_weights);

    if (!inside_prev_screen) {
        reset_reason = PT_RESET_OFFSCREEN;
    }

    vec2 target_res_f = vec2(target_res);
    float texel_world_size = (2.0 * current_dist)
        / max(gbufferProjection[1][1] * float(active_res.y), 1.0);
    float plane_tol = max(0.008 * current_dist, 0.05) + 0.5 * texel_world_size;
    float max_chord = max(0.15 * current_dist, 0.45)
        + 2.5 * texel_world_size / max(abs(prev_view_normal.z), 0.12);
    float max_chord_sq = max_chord * max_chord;

    bool history_valid = false;
    float total_confidence = 0.0;
    vec3 prev_rad = vec3(0.0);
    float prev_hit = current_hit_norm;
    float prev_count = 0.0;
    float prev_var = 0.0;
    float prev_fast_m1 = current_lum;

    if (inside_prev_screen) {
        vec2 hist_px = prev_framebuffer_uv * target_res_f - 0.5;
        ivec2 hist_base = ivec2(floor(hist_px));
        vec2 hist_f = fract(hist_px);

        // 1. Evaluate 2x2 subpixel bilateral bilinear footprint first (zero UV drift)
        vec3 bilin_rad = vec3(0.0);
        float bilin_hit = 0.0;
        float bilin_count = 0.0;
        float bilin_var = 0.0;
        float bilin_fast = 0.0;
        float bilin_conf_sum = 0.0;
        float bilin_wsum = 0.0;
        vec3 valid_min = vec3(1e6);
        vec3 valid_max = vec3(-1e6);
        int core_valid_taps = 0;
        bool any_depth_ok = false;

        for (int jy = 0; jy <= 1; ++jy) {
            for (int ix = 0; ix <= 1; ++ix) {
                ivec2 c = clamp(hist_base + ivec2(ix, jy), ivec2(0), active_res - 1);
                float wb = (ix == 0 ? 1.0 - hist_f.x : hist_f.x)
                         * (jy == 0 ? 1.0 - hist_f.y : hist_f.y);
                vec4 tap_s, tap_a;
                bool d_ok, n_ok;
                float geom_conf = evaluate_history_tap_geometry(
                    c, target_res_f, ping, current_world_normal,
                    prev_camera_view_pos, prev_view_normal,
                    plane_tol, max_chord_sq, tap_s, tap_a, d_ok, n_ok
                );
                any_depth_ok = any_depth_ok || d_ok;
                if (geom_conf > 0.0) {
                    core_valid_taps += 1;
                    float w = max(wb, 0.02) * geom_conf;
                    vec2 tap_hc = unpack_hit_count(tap_a.g);
                    float tap_m1 = dot(tap_s.rgb, luminance_weights);
                    float tap_v = max(tap_a.r - tap_m1 * tap_m1, 0.0);

                    bilin_rad += tap_s.rgb * w;
                    bilin_hit += tap_hc.x * w;
                    bilin_count += tap_hc.y * w;
                    bilin_var += tap_v * w;
                    bilin_fast += max(tap_a.a, 0.0) * w;
                    bilin_conf_sum += geom_conf * w;
                    bilin_wsum += w;
                    valid_min = min(valid_min, tap_s.rgb);
                    valid_max = max(valid_max, tap_s.rgb);
                }
            }
        }

        // 2. Symmetric 4-tap cross fallback ONLY when all 4 core bilinear taps straddled a silhouette
        if (bilin_wsum <= 1e-4) {
            const ivec2 cross_offsets[4] = ivec2[4](
                ivec2(-1, 0), ivec2(2, 0), ivec2(0, -1), ivec2(0, 2)
            );
            for (int k = 0; k < 4; ++k) {
                ivec2 c = clamp(hist_base + cross_offsets[k], ivec2(0), active_res - 1);
                vec4 tap_s, tap_a;
                bool d_ok, n_ok;
                float geom_conf = evaluate_history_tap_geometry(
                    c, target_res_f, ping, current_world_normal,
                    prev_camera_view_pos, prev_view_normal,
                    plane_tol, max_chord_sq, tap_s, tap_a, d_ok, n_ok
                ) * 0.5;
                any_depth_ok = any_depth_ok || d_ok;
                if (geom_conf > 0.0) {
                    vec2 tap_hc = unpack_hit_count(tap_a.g);
                    float tap_m1 = dot(tap_s.rgb, luminance_weights);
                    float tap_v = max(tap_a.r - tap_m1 * tap_m1, 0.0);

                    bilin_rad += tap_s.rgb * geom_conf;
                    bilin_hit += tap_hc.x * geom_conf;
                    bilin_count += tap_hc.y * geom_conf;
                    bilin_var += tap_v * geom_conf;
                    bilin_fast += max(tap_a.a, 0.0) * geom_conf;
                    bilin_conf_sum += geom_conf * geom_conf;
                    bilin_wsum += geom_conf;
                }
            }
        }

        reset_reason = any_depth_ok ? PT_RESET_NORMAL : PT_RESET_DEPTH;

        if (bilin_wsum > 1e-4) {
            history_valid = true;
            float inv_w = 1.0 / bilin_wsum;
            prev_rad = bilin_rad * inv_w;
            prev_hit = clamp01(bilin_hit * inv_w);
            prev_count = bilin_count * inv_w;
            prev_var = max(bilin_var * inv_w, 0.0);
            prev_fast_m1 = bilin_fast * inv_w;
            total_confidence = clamp01(bilin_conf_sum * inv_w);

            // 3. Upgrade interior radiance to 2x2-AABB-clamped 4x4 Catmull-Rom
            // when all 16 stencil taps lie on the same continuous surface.
            if (core_valid_taps == 4) {
                vec4 hist_wx = catmull_rom_weights(hist_f.x);
                vec4 hist_wy = catmull_rom_weights(hist_f.y);
                vec3 bicubic_rad = vec3(0.0);
                float bicubic_wsum = 0.0;
                bool all_16_valid = true;

                for (int oy = -1; oy <= 2 && all_16_valid; ++oy) {
                    float wy = hist_wy[oy + 1];
                    for (int ox = -1; ox <= 2; ++ox) {
                        ivec2 c = clamp(hist_base + ivec2(ox, oy), ivec2(0), active_res - 1);
                        vec4 tap_s, tap_a;
                        bool d_ok, n_ok;
                        float geom_conf = evaluate_history_tap_geometry(
                            c, target_res_f, ping, current_world_normal,
                            prev_camera_view_pos, prev_view_normal,
                            plane_tol, max_chord_sq, tap_s, tap_a, d_ok, n_ok
                        );
                        if (geom_conf <= 0.0) {
                            all_16_valid = false;
                            break;
                        }
                        float w = hist_wx[ox + 1] * wy;
                        bicubic_rad += tap_s.rgb * w;
                        bicubic_wsum += w;
                    }
                }

                if (all_16_valid && bicubic_wsum > 0.25) {
                    prev_rad = clamp(
                        max(bicubic_rad / bicubic_wsum, vec3(0.0)),
                        valid_min,
                        valid_max
                    );
                }
            }
        }
    }

    if (history_valid) {
        float prev_m1 = dot(prev_rad, luminance_weights);
        float prev_m2 = prev_m1 * prev_m1 + prev_var;
        float spatial_lum = dot(raw_m1, luminance_weights);

        // Variance-adapted dual-history reactivity (A-SVGF / ReBLUR):
        // Pool current 3x3 bilateral spatial_lum into fast_m1 (~9x lower variance than 1-pixel raw)
        // and adapt the EMA horizon to the heavy-tail ratio (rel_std), eliminating stationary
        // torch dimming (0 false rejections) while flushing broken lights during both motion & rest.
        float rel_std = sqrt(prev_var) / (prev_m1 + 0.05);
        float alpha_fast = clamp(
            (0.18 * current_spp) / (1.0 + 0.45 * rel_std),
            0.05,
            0.35
        );
        fast_m1 = raw_finite ? mix(prev_fast_m1, spatial_lum, alpha_fast) : prev_fast_m1;

        float rel_diff = abs(fast_m1 - prev_m1)
            / (max(fast_m1, prev_m1) + RTGI_ADAPT_SIGMA_EPS);
        float adapt_lo = RTGI_ADAPT_REL_BASE + 0.06 * min(rel_std, 3.0);
        float adapt_hi = adapt_lo + RTGI_ADAPT_REL_SPAN;

        bool adapt_allowed = raw_finite
            && total_confidence >= RTGI_ADAPT_MIN_CONFIDENCE
            && prev_count >= RTGI_ADAPT_MIN_SAMPLES;

        rejection = adapt_allowed
            ? smoothstep(adapt_lo, adapt_hi, rel_diff) * RTGI_ADAPT_STRENGTH
            : 0.0;
        bool rejected = rejection > 0.05;

        // Accumulate sample count weighted by reprojection confidence, and shorten
        // stale history when dual-history detects a genuine lighting change.
        sample_count = min(
            prev_count + current_spp * total_confidence,
            float(RTGI_SPP_CAP)
        );
        if (rejected) {
            sample_count = mix(sample_count, min(sample_count, 4.0 * current_spp), rejection);
        }

        float alpha_temporal = current_spp / max(sample_count, current_spp);
        float min_alpha = 0.01 * RTGI_MIN_ACCUM;
        alpha_temporal = max(alpha_temporal, min_alpha);
        alpha_temporal = max(alpha_temporal, rejection);

        bool accumulating = static_state.camera_static
            && static_state.projection_static
            && raw_finite;

        reset_reason = !accumulating
            ? (static_state.projection_static ? PT_RESET_CAMERA : PT_RESET_PROJECTION)
            : (rejected ? PT_RESET_ADAPTIVE : PT_RESET_NONE);

        float conf_weight = smoothstep(0.0, 0.75, total_confidence);
        alpha = clamp(mix(0.20, alpha_temporal, conf_weight), min_alpha, 1.0);

        accumulated_radiance = mix(prev_rad, current_radiance, alpha);
        accumulated_hit = mix(prev_hit, current_hit_norm, alpha);

        if (!raw_finite) {
            accumulated_radiance = prev_rad;
            accumulated_hit = prev_hit;
            reset_reason = PT_RESET_NON_FINITE;
        }

        m2 = raw_finite
            ? mix(prev_m2, current_lum * current_lum, alpha)
            : prev_m2;

#if GI_DEBUG == GI_DEBUG_REJECT
        history_accepted = 1.0;
#endif
    } else {
#if RTGI_SPATIAL_FALLBACK != 0
        vec4 fb_radiance_hit = vec4(0.0);
        float fb_second_moment = 0.0;
        float fb_weight = 0.0;
        pt_spatial_fallback(
            texel,
            target_res,
            active_res,
            current_dist,
            current_world_normal,
            view_pos,
            fb_radiance_hit,
            fb_second_moment,
            fb_weight
        );

        float spatial_samples = min(fb_weight, PT_SPATIAL_MAX_WEIGHT);
        bool spatial_ok = spatial_samples >= 0.75;
#else
        vec4 fb_radiance_hit = vec4(0.0);
        float fb_second_moment = 0.0;
        float spatial_samples = 0.0;
        bool spatial_ok = false;
#endif

        if (spatial_ok) {
            float total_samples = spatial_samples + current_spp;
            float inv_total = 1.0 / total_samples;
            accumulated_radiance
                = (fb_radiance_hit.rgb * spatial_samples + current_radiance * current_spp)
                * inv_total;
            accumulated_hit
                = (fb_radiance_hit.a * spatial_samples + current_hit_norm * current_spp)
                * inv_total;
            sample_count = total_samples;
            alpha = current_spp * inv_total;
            float fb_lum = dot(fb_radiance_hit.rgb, luminance_weights);
            float fb_var = max(fb_second_moment - fb_lum * fb_lum, 0.0);
            float estimate_lum = dot(accumulated_radiance, luminance_weights);
            // Keep per-sample variance un-divided so the spatial filter knows this pixel is still noisy.
            m2 = estimate_lum * estimate_lum
                + max(fb_var, dot(raw_sigma * raw_sigma, luminance_weights));
            fast_m1 = estimate_lum;
            reset_reason = PT_RESET_SPATIAL;
        } else {
            accumulated_radiance = current_radiance;
            accumulated_hit = current_hit_norm;
            sample_count = current_spp;
            alpha = 1.0;
            fast_m1 = current_lum;
            reset_reason = inside_prev_screen ? reset_reason : PT_RESET_OFFSCREEN;
        }
    }

#if GI_DEBUG == GI_DEBUG_REJECT
    // Red: history rejected. Yellow: history accepted. Black: no history available.
    imageStore(rtgi_reject_img, texel, vec4(1.0 - history_accepted, history_available, 0.0, 1.0));
#endif

    // Auxiliary buffer (see /include/lighting/rt/pt_state.glsl for the contract):
    // .r = mean squared luminance M2
    // .g = exact packed normalized hit distance (4 bits) + sample count (8 bits)
    // .b = exact packed 63x63 octahedral world normal
    // .a = fast-history luminance EMA
    vec4 aux_data = vec4(
        m2,
        pack_hit_count(accumulated_hit, sample_count),
        pack_oct_normal_f16(current_world_normal),
        fast_m1
    );

    vec4 accum_data = vec4(accumulated_radiance, current_dist);

    if (ping) {
        imageStore(rtgi_accum_img, texel, accum_data);
        imageStore(rtgi_sky_accum_img, texel, aux_data);
    } else {
        imageStore(rtgi_history_img, texel, accum_data);
        imageStore(rtgi_sky_history_img, texel, aux_data);
    }

    // Reactive mask: fraction of pixel history discarded this frame.
    float reactive = 1.0 - clamp01(
        (sample_count - current_spp) / max(float(RTGI_SPP_CAP) - current_spp, 1.0)
    );
    float converged = step(float(RTGI_SPP_CAP) - 0.5, sample_count)
        * (1.0 - step(0.05, rejection));
    write_state(texel, reset_reason, reactive, valid_path_fraction, converged);
}
