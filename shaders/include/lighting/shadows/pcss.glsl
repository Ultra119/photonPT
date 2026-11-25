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
    vec3 shadow_clip_pos,
    vec2 receiver_plane_slope,
    vec2 slope_dir,
    float slope_compress,
    float max_slope_dz,
    float search_radius_world,
    float sss_radius_world,
    float dither,
    bool has_sss
) {
#if INFO <= 1
    const int base_blocker_steps = 5;
#elif INFO == 2
    const int base_blocker_steps = 6;
#else
    const int base_blocker_steps = 8;
#endif
    int step_count = has_sss ? SSS_STEPS : base_blocker_steps;

    float ref_z = shadow_clip_pos.z * (SHADOW_DEPTH_SCALE * 0.5) + 0.5;

    float radius = (has_sss ? min(search_radius_world, sss_radius_world) : search_radius_world)
        * shadowProjection[0].x;

    float depth_sum = 0.0;
    float weight_sum = 0.0;
    float depth_sum_sss = 0.0;

    for (int i = 0; i < step_count; ++i) {
        vec2 offset = vogel_disc_sample(i, step_count, dither * tau) * radius;
        // Compress along the receiver's uphill gradient so grazing walls sample
        // a world-isotropic disc instead of stretching 1/NoL across 1-block roofs.
        offset -= slope_dir * (dot(offset, slope_dir) * (1.0 - slope_compress));
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
        float ref_z_tap = ref_z + clamp(dot(receiver_plane_slope, offset), -max_slope_dz, 0.0);
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
    vec2 receiver_plane_slope,
    vec2 slope_dir,
    float slope_compress,
    float max_slope_dz,
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
    // Compress the PCF kernel along the uphill receiver gradient by |shadow_normal.z|
    // and cap uphill depth bias strictly below 1-block roof thickness so wide cloud
    // penumbras never stretch or punch sunlight through ceilings at wall-ceiling joints.
    for (int i = 0; i < step_count; ++i) {
        vec2 offset
            = vogel_disc_sample(i, step_count, dither * tau) * filter_radius;
        offset -= slope_dir * (dot(offset, slope_dir) * (1.0 - slope_compress));

        vec2 uv = shadow_clip_pos.xy + offset;
        uv /= get_distortion_factor(uv);
        uv = uv * 0.5 + 0.5;

        bool in_bounds = clamp01(uv) == uv;
        float tap_z = shadow_screen_pos.z + clamp(dot(receiver_plane_slope, offset), -max_slope_dz, 0.0);
        float vis = in_bounds ? texture(shadowtex1, vec3(uv, tap_z)) : 1.0;
        opaque_sum += vis;

#ifdef SHADOW_COLOR
        vec3 color = vec3(1.0);
        if (vis > eps && in_bounds) {
            ivec2 texel = clamp(
                ivec2(uv * shadow_map_res),
                ivec2(0),
                ivec2(shadow_map_res - 1)
            );

            float depth = texelFetch(shadowtex0, texel, 0).x;
            if (depth < tap_z) {
                color = 4.0 * texelFetch(shadowcolor0, texel, 0).rgb;
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
    float slope_len_sq = dot(receiver_plane_slope, receiver_plane_slope);
    vec2 slope_dir = slope_len_sq > 1e-12
        ? receiver_plane_slope * inversesqrt(slope_len_sq)
        : vec2(0.0);
    // Pre-compress PCF/blocker discs along the uphill gradient by |shadow_normal.z|
    // so grazing walls project a world-isotropic disc instead of stretching by 1/NoL.
    float slope_compress = clamp(abs(shadow_normal.z), 0.15, 1.0);
    // Hard world-space cap on uphill depth bias (< 0.35 * light_dir.y blocks) so
    // wide cloud penumbras can never push tap_z above a 1-block-thick solid roof.
    float depth_to_world
        = 2.0 * abs(shadowProjectionInverse[2].z) * rcp(SHADOW_DEPTH_SCALE);
    float max_slope_dz = (0.35 * max(light_dir.y, 0.25)) / max(depth_to_world, 1.0);

    float cloud_scatter = clamp01(1.0 - cloud_shadows);

#ifdef SHADOW_VPS
#if INFO <= 1
    const float cloud_search_mult = 1.8;
#elif INFO == 2
    const float cloud_search_mult = 2.6;
#else
    const float cloud_search_mult = 3.2;
#endif
    float elevation_scale = 0.5 + 0.5 * linear_step(0.2, 0.4, light_dir.y);
    float max_penumbra_world
        = SHADOW_BLOCKER_SEARCH_RADIUS * mix(1.0, cloud_search_mult, cloud_scatter);
    float search_radius_world = max_penumbra_world * elevation_scale;
    float sss_radius_world = SHADOW_BLOCKER_SEARCH_RADIUS * 1.5 * elevation_scale;

    // Bias blocker search so grazing walls do not self-occlude into penumbra
    // strings along narrow skylight beams.
    vec2 blocker_search_result = blocker_search(
        shadow_clip_pos,
        receiver_plane_slope,
        slope_dir,
        slope_compress,
        max_slope_dz,
        search_radius_world,
        sss_radius_world,
        dither,
        sss_amount > eps
    );

    sss_depth = blocker_search_result.y;

    if (NoL < 1e-3) {
        return vec3(0.0); // now we can exit early for SSS blocks
    }
    if (blocker_search_result.x < eps) {
        return vec3(1.0); // blocker search empty handed => no occluders
    }

    // Exact orthographic world-space blocker distance and physical sun/moon
    // angular penumbra, widened by forward scattering through cloud droplets.
    float blocker_distance
        = max0(shadow_screen_pos.z - blocker_search_result.x) * depth_to_world;
#if defined WORLD_OVERWORLD
    float light_angular_radius
        = (sunAngle < 0.5) ? (SUN_ANGULAR_RADIUS * degree)
                           : (MOON_ANGULAR_RADIUS * degree);
#else
    const float light_angular_radius = SUN_ANGULAR_RADIUS * degree;
#endif
    float tan_angle = (tan(light_angular_radius) + 0.14 * sqr(cloud_scatter))
        * SHADOW_PENUMBRA_SCALE;
    float penumbra_world = min(blocker_distance * tan_angle, max_penumbra_world);
    float penumbra_size = penumbra_world * shadowProjection[0].x;
#else
    float penumbra_size
        = sqrt(0.5) * shadow_map_pixel_size * SHADOW_PENUMBRA_SCALE;
    penumbra_size *= 1.0 + 3.0 * sqr(cloud_scatter);

    // Increase blur radius to approximate subsurface scattering
    penumbra_size *= 1.0 + 7.0 * sss_amount;
#endif

#ifdef SHADOW_PCF
    vec3 shadow = shadow_pcf(
        shadow_screen_pos,
        shadow_clip_pos,
        receiver_plane_slope,
        slope_dir,
        slope_compress,
        max_slope_dz,
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
