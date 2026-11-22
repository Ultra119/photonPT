#if !defined INCLUDE_LIGHTING_SHADOWS
#define INCLUDE_LIGHTING_SHADOWS

#if defined SHADOW && (defined WORLD_OVERWORLD || defined WORLD_END)

#include "/include/lighting/shadows/common.glsl"
#include "/include/lighting/shadows/distortion.glsl"
#include "/include/utility/color.glsl"
#include "/include/utility/dithering.glsl"
#include "/include/utility/random.glsl"
#include "/include/utility/rotation.glsl"
#include "/include/utility/sampling.glsl"

const ivec2[9] blur_kernel_offsets_3x3 = ivec2[9](
    ivec2(-1, -1),
    ivec2(0, -1),
    ivec2(1, -1),
    ivec2(-1, 0),
    ivec2(0, 0),
    ivec2(1, 0),
    ivec2(-1, 1),
    ivec2(0, 1),
    ivec2(1, 1)
);

const int shadow_map_res = int(float(shadowMapResolution) * MC_SHADOW_QUALITY);
const float shadow_map_pixel_size = rcp(float(shadow_map_res));

vec2 blocker_search(
    vec3 scene_pos,
    vec2 receiver_plane_slope,
    float dither,
    bool has_sss
) {
    int step_count = has_sss ? SSS_STEPS : 8;

    vec3 shadow_view_pos = transform(shadowModelView, scene_pos);
    vec3 shadow_clip_pos = project_ortho(shadowProjection, shadow_view_pos);
    float ref_z = shadow_clip_pos.z * (SHADOW_DEPTH_SCALE * 0.5) + 0.5;

    float radius = SHADOW_BLOCKER_SEARCH_RADIUS * shadowProjection[0].x
        * (0.5 + 0.5 * linear_step(0.2, 0.4, light_dir.y));

    float depth_sum = 0.0;
    float weight_sum = 0.0;
    float depth_sum_sss = 0.0;

    for (int i = 0; i < step_count; ++i) {
        vec2 offset = vogel_disc_sample(i, step_count, dither * tau) * radius;
        vec2 uv = shadow_clip_pos.xy + offset;
        uv /= get_distortion_factor(uv);
        uv = uv * 0.5 + 0.5;
        if (clamp01(uv) != uv) {
            continue;
        }

        ivec2 texel = clamp(
            ivec2(uv * shadow_map_res),
            ivec2(0),
            ivec2(shadow_map_res - 1)
        );
        float depth = texelFetch(shadowtex0, texel, 0).x;
        float ref_z_tap = ref_z + clamp(dot(receiver_plane_slope, offset), -0.015, 0.0);
        float weight = step(depth, ref_z_tap);

        depth_sum += weight * depth;
        weight_sum += weight;
        depth_sum_sss += max0(ref_z - depth);
    }

    float blocker_depth = weight_sum == 0.0 ? 0.0 : depth_sum / weight_sum;
    float sss_depth = -shadowProjectionInverse[2].z * depth_sum_sss
        * rcp(SHADOW_DEPTH_SCALE * float(step_count));

    return vec2(blocker_depth, sss_depth);
}

vec3 shadow_basic(vec3 shadow_screen_pos) {
    if (clamp01(shadow_screen_pos.xy) != shadow_screen_pos.xy) {
        return vec3(1.0);
    }

    float shadow = texture(shadowtex1, shadow_screen_pos);

#ifdef SHADOW_COLOR
    ivec2 max_texel = ivec2(shadow_map_res - 1);
    ivec2 texel = clamp(
        ivec2(shadow_screen_pos.xy * shadow_map_res),
        ivec2(0),
        max_texel
    );

    float depth = texelFetch(shadowtex0, texel, 0).x;
    vec3 color = texelFetch(shadowcolor0, texel, 0).rgb * 4.0;
    float weight = step(depth, shadow_screen_pos.z);

    vec3 averageColor = vec3(0.0);
    for (int i = 0; i < 9; ++i) {
        ivec2 neighbor = clamp(
            texel + blur_kernel_offsets_3x3[i],
            ivec2(0),
            max_texel
        );
        averageColor += texelFetch(shadowcolor0, neighbor, 0).rgb;
    }
    // hide sunlight seams between colored shadow and opaque shadows
    weight *= step(eps, max_of(averageColor));

    color = color * weight + (1.0 - weight);

    return shadow * color;
#else
    return vec3(shadow);
#endif
}

vec3 shadow_pcf(
    vec3 shadow_screen_pos,
    vec3 shadow_clip_pos,
#ifdef SHADOW_COLOR
    vec3 shadow_screen_pos_translucent,
    vec3 shadow_clip_pos_translucent,
#endif
    vec2 receiver_plane_slope,
    float penumbra_size,
    float dither
) {
    // penumbra_size > max_filter_radius: blur
    // penumbra_size < min_filter_radius: anti-alias (blur then sharpen)
    float distortion_factor = get_distortion_factor(shadow_clip_pos.xy);
    float min_filter_radius = 2.0 * shadow_map_pixel_size * distortion_factor;

    float filter_radius = max(penumbra_size, min_filter_radius);
    float filter_scale = sqr(filter_radius / min_filter_radius);

    int step_count
        = int(SHADOW_PCF_STEPS_MIN + SHADOW_PCF_STEPS_SCALE * filter_scale);
    step_count = min(step_count, SHADOW_PCF_STEPS_MAX);

    float opaque_sum = 0.0;
    vec3 shadow_sum = vec3(0.0);

    // Combine opaque visibility and translucent transmittance per tap so
    // colored penumbra edges evaluate avg(V * T) across the full kernel.
    // Apply uphill receiver-plane depth bias (clamped <= 0) so wide PCF discs
    // on grazing vertical walls follow the wall plane uphill instead of
    // self-shadowing into horizontal shadow-map texel stripes.
    for (int i = 0; i < step_count; ++i) {
        vec2 offset
            = vogel_disc_sample(i, step_count, dither * tau) * filter_radius;

        vec2 uv = shadow_clip_pos.xy + offset;
        uv /= get_distortion_factor(uv);
        uv = uv * 0.5 + 0.5;

        float dz_tap = clamp(dot(receiver_plane_slope, offset), -0.015, 0.0);
        float vis = (clamp01(uv) == uv)
            ? texture(shadowtex1, vec3(uv, shadow_screen_pos.z + dz_tap))
            : 1.0;
        opaque_sum += vis;

#ifdef SHADOW_COLOR
        vec3 color = vec3(1.0);
        if (vis > eps) {
            vec2 uv_t = shadow_clip_pos_translucent.xy + offset;
            uv_t /= get_distortion_factor(uv_t);
            uv_t = uv_t * 0.5 + 0.5;

            if (clamp01(uv_t) == uv_t) {
                ivec2 texel = clamp(
                    ivec2(uv_t * shadow_map_res),
                    ivec2(0),
                    ivec2(shadow_map_res - 1)
                );

                float depth = texelFetch(shadowtex0, texel, 0).x;
                vec3 sample_color = texelFetch(shadowcolor0, texel, 0).rgb;
                color = mix(
                    vec3(1.0),
                    4.0 * sample_color,
                    step(depth, shadow_screen_pos_translucent.z + dz_tap)
                );
            }
        }
        shadow_sum += vis * color;
#else
        shadow_sum += vec3(vis);
#endif
    }

    float rcp_steps = rcp(float(step_count));
    float avg_opaque = opaque_sum * rcp_steps;
    if (avg_opaque <= eps) {
        return vec3(0.0);
    }

    vec3 avg_shadow = shadow_sum * rcp_steps;

    // sharpening for small penumbra sizes
    float sharpening_threshold
        = 0.4 * max0((min_filter_radius - penumbra_size) / min_filter_radius);
    float sharpened = linear_step(
        sharpening_threshold,
        1.0 - sharpening_threshold,
        avg_opaque
    );

    return avg_shadow * (sharpened / avg_opaque);
}

vec3 get_filtered_shadows(
    vec3 scene_pos,
    vec3 flat_normal,
    float skylight,
    float cloud_shadows,
    float sss_amount,
    inout float distance_fade,
    inout float sss_depth
) {
    sss_depth = 0.0;

    float NoL = dot(flat_normal, light_dir);

    vec3 bias = get_shadow_bias(scene_pos, flat_normal, NoL, skylight);

#ifdef PIXELATED_SHADOWS
    // Snap position to the nearest block texel
    const float pixel_scale = float(PIXELATED_SHADOWS_RESOLUTION);
    scene_pos = scene_pos + cameraPosition;
    scene_pos = floor(scene_pos * pixel_scale + 0.01) * rcp(pixel_scale)
        + (0.5 / pixel_scale);
    scene_pos = scene_pos - cameraPosition;
#endif

    vec3 shadow_view_pos
        = transform(shadowModelView, scene_pos + bias);
    vec3 shadow_clip_pos = project_ortho(shadowProjection, shadow_view_pos);
    vec3 shadow_screen_pos = distort_shadow_space(shadow_clip_pos) * 0.5 + 0.5;

    distance_fade = get_shadow_distance_fade(scene_pos, shadow_screen_pos);

    if (distance_fade >= 1.0) {
        return vec3(1.0);
    }

    float dither = texelFetch(noisetex, ivec2(gl_FragCoord.xy) & 511, 0).b;
    dither = r1(frameCounter, dither);

    // Exact shadow-screen depth slope dz/d(shadow_clip_xy) along the receiver
    // plane so uphill PCF / blocker taps never self-shadow grazing walls.
    vec3 shadow_normal = mat3(shadowModelView) * flat_normal;
    float nz_safe = shadow_normal.z >= 0.0
        ? max(shadow_normal.z, 0.05)
        : min(shadow_normal.z, -0.05);
    vec2 receiver_plane_slope
        = -(shadowProjection[2].z * (SHADOW_DEPTH_SCALE * 0.5) / nz_safe)
        * vec2(shadow_normal.x / shadowProjection[0].x,
               shadow_normal.y / shadowProjection[1].y);

#ifdef SHADOW_VPS
    // Bias blocker search so grazing walls do not self-occlude into penumbra
    // strings along narrow skylight beams.
    vec2 blocker_search_result = blocker_search(
        scene_pos + bias, receiver_plane_slope, dither, sss_amount > eps
    );

    sss_depth = blocker_search_result.y;

    if (NoL < 1e-3) {
        return vec3(0.0); // now we can exit early for SSS blocks
    }
    if (blocker_search_result.x < eps) {
        return vec3(1.0); // blocker search empty handed => no occluders
    }

    float penumbra_size = 16.0 * SHADOW_PENUMBRA_SCALE
        * (shadow_screen_pos.z - blocker_search_result.x)
        / blocker_search_result.x;
    penumbra_size
        *= 5.0 - 4.0 * cloud_shadows; // Increase penumbra radius inside cloud
                                      // shadows, nice overcast look
    penumbra_size = min(penumbra_size, SHADOW_BLOCKER_SEARCH_RADIUS);
    penumbra_size *= shadowProjection[0].x;
#else
    float penumbra_size
        = sqrt(0.5) * shadow_map_pixel_size * SHADOW_PENUMBRA_SCALE;

    // Increase blur radius to approximate subsurface scattering
    penumbra_size *= 1.0 + 7.0 * sss_amount;
#endif

#ifdef SHADOW_COLOR
    // Calculate position without light leaking fix applied, for colored shadow
    // Applying light leaking fix to translucent shadows causes artifacts on
    // water caustics
    vec3 shadow_view_pos_translucent
        = transform(shadowModelView, scene_pos + bias);
    vec3 shadow_clip_pos_translucent
        = project_ortho(shadowProjection, shadow_view_pos_translucent);
    vec3 shadow_screen_pos_translucent
        = distort_shadow_space(shadow_clip_pos_translucent) * 0.5 + 0.5;
#endif

#ifdef SHADOW_PCF
    vec3 shadow = shadow_pcf(
        shadow_screen_pos,
        shadow_clip_pos,
#ifdef SHADOW_COLOR
        shadow_screen_pos_translucent,
        shadow_clip_pos_translucent,
#endif
        receiver_plane_slope,
        penumbra_size,
        dither
    );
#else
    vec3 shadow = shadow_basic(shadow_screen_pos);
#endif

    return shadow;
}
#endif

#endif // INCLUDE_LIGHTING_SHADOWS
