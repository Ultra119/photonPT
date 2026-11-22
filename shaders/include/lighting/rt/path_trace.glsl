#if !defined INCLUDE_LIGHTING_RT_PATH_TRACE
#define INCLUDE_LIGHTING_RT_PATH_TRACE

// Multi-bounce diffuse path tracer for voxel scene geometry.

#include "/include/utility/color.glsl"
#include "/include/utility/geometry.glsl"
#include "/include/utility/random.glsl"
#include "/include/sky/projection.glsl"
#include "voxel_dda.glsl"

#ifndef PT_BOUNCES
#define PT_BOUNCES 2
#endif

#ifndef RTGI_HIT_NONE
#define RTGI_HIT_NONE  0
#define RTGI_HIT_VOXEL 1
#define RTGI_HIT_SKY   2
#endif

#if !defined SAMPLE_COSINE_HEMISPHERE_FN
#define SAMPLE_COSINE_HEMISPHERE_FN
vec3 sample_cosine_hemisphere(vec3 normal, vec2 u) {
    float r = sqrt(u.x);
    float phi = 6.28318530718 * u.y;

    vec3 local_dir = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
    mat3 tbn_basis = build_orthonormal_basis(normal);
    return normalize(tbn_basis * local_dir);
}
#endif

// GGX Visible Normal Distribution Function (VNDF) sampling (Heitz 2018 / Zombye 2023)
vec3 sample_ggx_vndf_local(vec3 viewer_dir, vec2 alpha, vec2 hash) {
    viewer_dir = normalize(vec3(alpha * viewer_dir.xy, viewer_dir.z));
    const float tau = 6.28318530718;
    float phi = tau * hash.x;
    float cos_theta = fma(1.0 - hash.y, 1.0 + viewer_dir.z, -viewer_dir.z);
    float sin_theta = sqrt(clamp(1.0 - cos_theta * cos_theta, 0.0, 1.0));
    vec3 reflected = vec3(vec2(cos(phi), sin(phi)) * sin_theta, cos_theta);
    vec3 halfway = reflected + viewer_dir;
    return normalize(vec3(alpha * halfway.xy, halfway.z));
}

vec3 sample_specular_direction(vec3 normal, vec3 viewer_dir, float roughness, vec2 hash) {
    mat3 tbn = build_orthonormal_basis(normal);
    vec3 local_v = transpose(tbn) * viewer_dir;
    float alpha = max(roughness * roughness, 0.002);
    vec3 local_m = sample_ggx_vndf_local(local_v, vec2(alpha), hash);
    vec3 m = tbn * local_m;
    return reflect(-viewer_dir, m);
}

// Evaluates explicit path tracing for one sample ray.
// Returns vec4(radiance, norm_hit_dist) and writes hit_source (RTGI_HIT_*).
vec4 evaluate_path_trace(
    vec3 scene_pos,
    vec3 world_normal,
    vec3 mapped_normal,
    vec2 dither,
    float receiver_skylight,
    usampler3D voxel_grid,
    sampler2D sky_map,
    vec3 direct_sun_color,
    vec3 l_dir,
    int ray_index,
    out int hit_source
) {
    hit_source = RTGI_HIT_NONE;

    vec3 L = vec3(0.0);
    vec3 throughput = vec3(1.0);
    float primary_hit_dist = 32.0;

    vec2 u0 = r2(ray_index * PT_BOUNCES, dither);

    // Initial ray direction sampled from primary shading normal
    vec3 current_dir = sample_cosine_hemisphere(mapped_normal, u0);
    if (dot(current_dir, world_normal) <= 0.0) {
        current_dir = reflect(current_dir, world_normal);
    }

    vec3 ray_origin = scene_pos;
    vec3 surf_normal = world_normal;
    float current_dither = dither.x;

    for (int b = 0; b < PT_BOUNCES; ++b) {
        float bounce_max_dist = (b == 0) ? rt_primary_max_distance : rt_bounce_max_distance;
        VoxelHit hit = trace_voxel_dda(
            voxel_grid,
            ray_origin,
            surf_normal,
            current_dir,
            current_dither,
            direct_sun_color,
            l_dir,
            bounce_max_dist
        );

        // Thin-emitter pass-through gathered along this ray segment (torches, lanterns)
        L += hit.passed_emission * throughput;

        if (!hit.hit) {
#if defined WORLD_OVERWORLD
            // Only suppress escaped sky dome on subterranean out-of-volume misses
            float exit_world_y = hit.hit_pos.y + cameraPosition.y;
            bool subterranean_void_exit = (receiver_skylight < 0.01)
                && (eye_skylight < 0.01)
                && (exit_world_y < SEA_LEVEL - 4.0)
                && (current_dir.y < 0.15);
            float escaped_sky_gate = subterranean_void_exit ? 0.0 : 1.0;
#else
            const float escaped_sky_gate = 1.0;
#endif
            vec3 sky_radiance = texture(sky_map, project_sky(current_dir)).rgb
                * (hit.transmittance * escaped_sky_gate);

            L += sky_radiance * throughput;

            if (b == 0) {
                hit_source = (escaped_sky_gate > 0.0) ? RTGI_HIT_SKY : RTGI_HIT_NONE;
                primary_hit_dist = 32.0;
            }
            break;
        }

        if (b == 0) {
            hit_source = RTGI_HIT_VOXEL;
            primary_hit_dist = max(hit.distance, 0.0);
        }

        // Direct hit radiance: direct sun (with voxel shadow visibility) + solid block emission
        vec3 direct_radiance = hit.radiance * hit.transmittance;
        L += direct_radiance * throughput;

        // Path termination at maximum bounces
        if (b == PT_BOUNCES - 1) {
            break;
        }

        // Physical Lambertian diffuse continuation throughput
        throughput *= hit.albedo * hit.transmittance;

        // Russian roulette path termination for dark/low-throughput bounces
        float q = clamp(max(throughput.r, max(throughput.g, throughput.b)) * 1.6, 0.15, 1.0);
        if (q < 0.999) {
            float rr_rand = hash1(vec3(dither, float(ray_index * PT_BOUNCES + b + 17)));
            if (rr_rand > q) {
                break;
            }
            throughput /= q;
        }

        // Next bounce origin and geometric normal
        ray_origin = hit.hit_pos;
        surf_normal = hit.hit_normal;

        // Generate decorrelated direction for next bounce
        vec2 bounce_seed = hash2(vec3(dither, float(ray_index * PT_BOUNCES + b + 1)));
        vec2 next_u = r2(ray_index * PT_BOUNCES + (b + 1), bounce_seed);

#if defined PT_GLOSSY_TRANSPORT
        // Optional glossy bounce on smooth materials using an independent lobe selector
        float lobe_rand = hash1(vec3(dither.xy, float(ray_index * PT_BOUNCES + b + 31)));
        if (hit.roughness < 0.45 && lobe_rand < 0.2) {
            vec3 viewer_dir = -current_dir;
            current_dir = sample_specular_direction(surf_normal, viewer_dir, hit.roughness, next_u);
        } else {
            current_dir = sample_cosine_hemisphere(surf_normal, next_u);
        }
#else
        current_dir = sample_cosine_hemisphere(surf_normal, next_u);
#endif

        if (dot(current_dir, surf_normal) <= 0.0) {
            current_dir = reflect(current_dir, surf_normal);
        }
        current_dither = hash1(vec3(dither.yx, float(ray_index * PT_BOUNCES + b + 1)));
    }

    // Numerical safety guard: reject non-finite or negative values
    if (!all(equal(L, L)) || any(lessThan(L, vec3(0.0)))) {
        return vec4(0.0);
    }

    // Saturating normalized primary hit distance (ReBLUR-style contact guide):
    // ~0.05 at 0.1m contact corners, 0.50 at 2m, ~0.94 on far/sky rays.
    float norm_hit_dist = primary_hit_dist / (primary_hit_dist + 2.0);

    return vec4(L, norm_hit_dist);
}

#endif // INCLUDE_LIGHTING_RT_PATH_TRACE
