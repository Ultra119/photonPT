#if !defined INCLUDE_LIGHTING_SHADOWS_SSRT_SHADOWS
#define INCLUDE_LIGHTING_SHADOWS_SSRT_SHADOWS

#include "/include/lighting/shadows/common.glsl"
#include "/include/utility/random.glsl"
#include "/include/utility/sampling.glsl"
#include "/include/utility/space_conversion.glsl"

bool raymarch_shadow(
    sampler2D depth_sampler,
    mat4 projection_matrix,
    mat4 projection_matrix_inverse,
    vec3 ray_origin_screen,
    vec3 ray_origin_view,
    vec3 ray_dir_view,
    bool has_sss,
    float dither,
    out float sss_depth
) {
    const uint step_count = uint(SHADOW_SSRT_STEPS);
    const float step_ratio = 2.0; // geometric sample distribution
    const float z_tolerance = 10.0; // assumed minimum thickness in blocks

    // Project a distance-scaled endpoint and interpolate 1/z linearly in screen
    // space to avoid float32 catastrophic cancellation near depth = 1.0.
    float z0 = max(-ray_origin_view.z, eps);
    float max_probe = ray_dir_view.z > eps
        ? max(0.05, (z0 - max(2.0 * near, 0.05)) * rcp(ray_dir_view.z))
        : 1e6;
    float probe_dist = min(max(4.0, 0.5 * z0), max_probe);
    vec3 ray_end_view = ray_origin_view + probe_dist * ray_dir_view;
    float z1 = max(-ray_end_view.z, eps);

    vec2 ray_end_xy
        = view_to_screen_space(projection_matrix, ray_end_view, true).xy;
    vec2 delta_xy = ray_end_xy - ray_origin_screen.xy;
    float delta_xy_len = length(delta_xy);
    if (delta_xy_len < eps) {
        sss_depth = 0.0;
        return false;
    }

    vec2 dir_xy = delta_xy * rcp(delta_xy_len);
    float inv_z0 = rcp(z0);
    float inv_z1 = rcp(z1);
    float d_inv_z_per_xy = (inv_z1 - inv_z0) * rcp(delta_xy_len);

    // Parametric exit distance in screen XY [0, 1]^2
    vec2 box_target = step(vec2(0.0), dir_xy);
    bvec2 active_axis = greaterThan(abs(dir_xy), vec2(eps));
    vec2 safe_dir = mix(vec2(1.0), dir_xy, active_axis);
    vec2 box_dist
        = mix(vec2(1e6), (box_target - ray_origin_screen.xy) / safe_dir, active_axis);
    float ray_length = max0(min_of(box_dist));
    ray_length = min(
        ray_length,
        max(0.1, exp(-max0(length(ray_origin_view) * 0.025 - 1.0)))
    );
    if (d_inv_z_per_xy < -eps) {
        ray_length = min(ray_length, (inv_z0 * 0.999) * rcp(-d_inv_z_per_xy));
    } else if (d_inv_z_per_xy > eps) {
        float inv_z_near = rcp(max(near, 0.01));
        ray_length = min(
            ray_length,
            max0(inv_z_near - inv_z0) * rcp(d_inv_z_per_xy)
        );
    }

    const float initial_step_scale = step_ratio == 1.0
        ? rcp(float(step_count))
        : (step_ratio - 1.0) / (pow(step_ratio, float(step_count)) - 1.0);
    float step_length = ray_length * initial_step_scale;
    float s = length(view_pixel_size);

    ivec2 depth_size = textureSize(depth_sampler, 0);
    ivec2 max_texel = clamp(
        ivec2(ceil(vec2(depth_size) * taau_render_scale - 0.5)),
        ivec2(1),
        depth_size
    ) - 1;
    vec2 active_extent = vec2(depth_size) * taau_render_scale;

    bool hit = false;
    bool hit_after_sss = false;
    bool sss_raymarch = has_sss;
    bool found_exit = false;
    float exit_s = 0.0;
    float exit_z = z0;

    for (int i = 0; i < step_count; ++i) {
        float s_dithered = s + dither * step_length;
        s += step_length;
        step_length *= step_ratio;

        vec2 sample_xy = ray_origin_screen.xy + s_dithered * dir_xy;
        if (clamp01(sample_xy) != sample_xy) {
            break;
        }

        float inv_z_ray = inv_z0 + s_dithered * d_inv_z_per_xy;
        if (inv_z_ray <= eps) {
            break;
        }
        float z_ray = rcp(inv_z_ray);

        ivec2 sample_texel = clamp(
            ivec2(sample_xy * active_extent),
            ivec2(0),
            max_texel
        );
        float depth = texelFetch(depth_sampler, sample_texel, 0).x;

        bool inside = false;
        if (depth > 0.0 && depth < 1.0) {
            float z_sample
                = screen_to_view_space_depth(projection_matrix_inverse, depth);

            float dz = z_ray - z_sample;
            // Cover 24-bit depth quantization and 1-pixel slope steps on
            // distant terrain so surfaces never self-shadow in depth bands.
            float z_bias = 0.03 + 0.002 * z_ray;
            float z_tol = max(z_tolerance, 0.05 * z_ray);
            inside = dz > z_bias && dz < z_tol;
        }
        hit = inside || hit;

        if (sss_raymarch) {
            if (!inside) {
                exit_s = s_dithered;
                exit_z = z_ray;
                found_exit = true;
                sss_raymarch = false;
            }
        } else {
            hit_after_sss = inside || hit_after_sss;
            if (hit) {
                break;
            }
        }
    }

    float exit_dist = found_exit
        ? probe_dist * (exit_s * rcp(delta_xy_len)) * (exit_z * inv_z1)
        : 0.0;
    sss_depth = hit_after_sss ? -1.0 : max0(exit_dist * 0.2);

    return hit;
}

float get_screen_space_shadows(
    vec2 position_screen_xy,
    vec3 position_view,
    float depth,
#ifdef LOD_MOD_ACTIVE
    float depth_lod,
#endif
    float skylight,
    bool has_sss,
    inout float sss_depth
) {
    // Dithering for ray offset
    float dither = texelFetch(noisetex, ivec2(gl_FragCoord.xy) & 511, 0).b;
    dither = r1(frameCounter, dither);

    // Slightly randomise ray direction to create soft shadows
    vec2 hash = hash2(gl_FragCoord.xy);
    vec3 ray_dir
        = normalize(view_light_dir + 0.03 * uniform_sphere_sample(hash));

#ifdef LOD_MOD_ACTIVE
    // Which depth map to raymarch depends on distance
    // Closer fragments: use combined depth texture (so MC terrain can cast)
    // Further fragments: use LoD depth texture (maximise precision)

    bool raymarch_combined_depth = length_squared(position_view)
        < sqr(far + 64.0); // heuristic of 4 chunks overlap
    bool hit;

    /*
    #ifdef MC_GL_KHR_shader_subgroup
    // Using subgroup ops, we make sure that if any fragments in a warp are
    raymarching the
    // combined depth buffer, they all do, to avoid divergent branches.
    raymarch_combined_depth = subgroupAny(raymarch_combined_depth);
    #endif
    */

    if (raymarch_combined_depth) {
        hit = raymarch_shadow(
            combined_depth_tex,
            combined_projection_matrix,
            combined_projection_matrix_inverse,
            vec3(position_screen_xy, depth),
            position_view,
            ray_dir,
            has_sss,
            dither,
            sss_depth
        );
    } else {
        hit = raymarch_shadow(
            lod_depth_tex,
            lod_projection_matrix,
            lod_projection_matrix_inverse,
            vec3(position_screen_xy, depth_lod),
            screen_to_view_space(
                lod_projection_matrix_inverse,
                vec3(position_screen_xy, depth_lod),
                true
            ),
            ray_dir,
            has_sss,
            dither,
            sss_depth
        );
    }
#else
    bool hit = raymarch_shadow(
        depthtex1,
        gbufferProjection,
        gbufferProjectionInverse,
        vec3(position_screen_xy, depth),
        position_view,
        view_light_dir,
        has_sss,
        dither,
        sss_depth
    );
#endif

    return float(!hit) * get_lightmap_light_leak_prevention(skylight);
}

#endif // INCLUDE_LIGHTING_SHADOWS_SSRT_SHADOWS
