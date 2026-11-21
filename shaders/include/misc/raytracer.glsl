#if !defined INCLUDE_MISC_RAYTRACER
#define INCLUDE_MISC_RAYTRACER

#include "/include/misc/lod_mod_support.glsl"
#include "/include/utility/geometry.glsl"
#include "/include/utility/space_conversion.glsl"

#if defined SSRT_LOD
#define SSRT_DEPTH_SAMPLER lod_depth_tex_shading
#define SSRT_PROJECTION_MATRIX lod_projection_matrix
#define SSRT_PROJECTION_MATRIX_INVERSE lod_projection_matrix_inverse
#else
#define SSRT_DEPTH_SAMPLER combined_depth_tex
#define SSRT_PROJECTION_MATRIX combined_projection_matrix
#define SSRT_PROJECTION_MATRIX_INVERSE combined_projection_matrix_inverse
#endif

bool raymarch_depth_buffer(
    vec3 screen_pos,
    vec3 view_pos,
    vec3 view_dir,
    float dither,
    uint intersection_step_count,
    uint refinement_step_count,
    out vec3 ray_pos
) {
    if (view_dir.z > 0.0 && view_dir.z >= -view_pos.z) {
        return false;
    }

    vec3 screen_dir = normalize(
        view_to_screen_space(SSRT_PROJECTION_MATRIX, view_pos + view_dir, true)
        - screen_pos
    );

    float ray_length = ray_box_exit_unit(screen_pos, screen_dir);

    float step_length = ray_length * rcp(float(intersection_step_count));

    vec3 ray_step = screen_dir * step_length;
    ray_pos
        = screen_pos + dither * ray_step + length(view_pixel_size) * screen_dir;

    float depth_tolerance = max(
        abs(ray_step.z) * 3.0,
        0.02 / max(sqr(view_pos.z), 1e-4)
    ); // from DrDesten <3

    ivec2 depth_size = textureSize(SSRT_DEPTH_SAMPLER, 0);
    ivec2 max_texel = clamp(
        ivec2(ceil(vec2(depth_size) * taau_render_scale - 0.5)),
        ivec2(1),
        depth_size
    ) - 1;
    vec2 active_extent = vec2(depth_size) * taau_render_scale;

    bool hit = false;
    float final_depth = 1.0;

    // Intersection loop

    for (int i = 0; i < intersection_step_count; ++i, ray_pos += ray_step) {
#ifdef LOD_MOD_ACTIVE
        if (ray_pos.z < 0.0) {
            continue;
        }
#endif
        if (clamp01(ray_pos) != ray_pos) {
            return false;
        }

        ivec2 sample_texel = clamp(
            ivec2(ray_pos.xy * active_extent),
            ivec2(0),
            max_texel
        );
        float depth = texelFetch(SSRT_DEPTH_SAMPLER, sample_texel, 0).x;

        if (depth < ray_pos.z
            && abs(depth_tolerance - (ray_pos.z - depth)) < depth_tolerance) {
            hit = true;
            final_depth = depth;
            break;
        }
    }

    if (!hit) {
        return false;
    }

    // Refinement loop

    for (int i = 0; i < refinement_step_count; ++i) {
        ray_step *= 0.5;
        if (clamp01(ray_pos) != ray_pos) {
            break;
        }

        ivec2 sample_texel = clamp(
            ivec2(ray_pos.xy * active_extent),
            ivec2(0),
            max_texel
        );
        float depth = texelFetch(SSRT_DEPTH_SAMPLER, sample_texel, 0).x;

        if (depth < ray_pos.z
            && abs(depth_tolerance - (ray_pos.z - depth)) < depth_tolerance) {
            ray_pos -= ray_step;
            final_depth = depth;
        } else {
            ray_pos += ray_step;
        }
    }

    // No intersection if the binary search landed on a sky pixel or a hand
    // pixel
    return clamp01(ray_pos) == ray_pos && hand_depth < final_depth
        && final_depth < 1.0;
}

#endif // INCLUDE_MISC_RAYTRACER
