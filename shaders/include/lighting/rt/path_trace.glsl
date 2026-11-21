#ifndef INCLUDE_LIGHTING_RT_PATH_TRACE
#define INCLUDE_LIGHTING_RT_PATH_TRACE

#include "/include/utility/geometry.glsl"
#include "/include/utility/random.glsl"
#include "/include/sky/projection.glsl"
#include "voxel_dda.glsl"

#ifdef VOXEL_DDA_USE_SHADOWMAP
#include "/include/lighting/shadows/distortion.glsl"
#endif
#ifdef RTGI_EMITTER_SAMPLING
#include "rtgi_emitters.glsl"
#endif

const int rtgi_hit_none = 0;
const int rtgi_hit_voxel = 1;
const int rtgi_hit_volume_exit = 2;
const int rtgi_hit_distance_limit = 3;
const int rtgi_hit_invalid = 4;

vec3 sample_cosine_hemisphere(vec3 normal, vec2 u) {
    float r = sqrt(u.x);
    float phi = tau * u.y;
    vec3 local_dir = vec3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x));
    return normalize(build_orthonormal_basis(normal) * local_dir);
}

vec3 voxel_sun_visibility(usampler3D grid, vec3 origin, vec3 direction) {
    float reach = 32.0;
    float macro_visibility = 1.0;
    vec3 shadow_tint = vec3(1.0);
#ifdef VOXEL_DDA_USE_SHADOWMAP
    vec3 near_scene = voxel_to_scene_space(origin) + direction * 0.45;
    vec3 shadow_clip = project_ortho(shadowProjection, transform(shadowModelView, near_scene));
    if (all(lessThan(abs(shadow_clip), vec3(0.99)))) {
        vec3 screen = distort_shadow_space(shadow_clip) * 0.5 + 0.5;
        macro_visibility = texture(shadowtex1, vec3(screen.xy, screen.z - 0.0025));
        if (macro_visibility < 0.01) {
            return vec3(0.0);
        }
#ifdef SHADOW_COLOR
        ivec2 shadow_size = textureSize(shadowtex0, 0);
        ivec2 shadow_texel = clamp(ivec2(screen.xy * vec2(shadow_size)), ivec2(0), shadow_size - 1);
        if (texelFetch(shadowtex0, shadow_texel, 0).x < screen.z - 0.0025) {
            vec3 sc = texelFetch(shadowcolor0, shadow_texel, 0).rgb * 4.0;
            if (max_of(sc) > eps) {
                shadow_tint = clamp(sc, vec3(0.0), vec3(1.0));
            }
        }
#endif
        reach = 4.5;
    }
#endif
    VoxelHit blocker = trace_voxel_segment(grid, origin, direction, reach, voxel_ray_visibility);
    if (blocker.status != voxel_volume_exit && blocker.status != voxel_distance_limit) {
        return vec3(0.0);
    }
    // Prefer exact near-field voxel glass transmittance when traversed so glass
    // within `reach` is never double-tinted by shadowcolor0; fall back to
    // shadowcolor0 for distant stained glass casters beyond `reach`.
    vec3 transmittance = any(lessThan(blocker.transmittance, vec3(0.999)))
        ? blocker.transmittance
        : shadow_tint;
    return macro_visibility * transmittance;
}

vec3 voxel_environment(
    sampler2D sky_map, vec3 direction, vec3 endpoint, float receiver_skylight
) {
    // The sky map models incoming radiance beyond the represented segment.
    // Inside a subterranean volume, avoid exposing the sky below the surface.
#if defined WORLD_OVERWORLD
    float exit_world_y = voxel_to_scene_space(endpoint).y + cameraPosition.y;
    if (receiver_skylight < 0.01 && eye_skylight < 0.01
        && exit_world_y < SEA_LEVEL - 4.0 && direction.y < 0.15) {
        return vec3(0.0);
    }
#endif
    vec3 sky = texture(sky_map, project_sky(direction)).rgb;
    return (!any(isnan(sky)) && !any(isinf(sky))) ? max0(sky) : vec3(0.0);
}

// RGB is demodulated diffuse radiance. A is the distance from the receiver
// anchor to the first opaque hit. A miss has distance -1.
vec4 evaluate_path_trace(
    vec3 receiver, vec3 receiver_origin, vec3 receiver_normal, float receiver_skylight,
    usampler3D grid, sampler2D sky_map, vec3 direct_sun_color, vec3 light_direction,
    uint seed, bool sample_emitters, out int hit_source
) {
    hit_source = rtgi_hit_none;
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    vec3 origin = receiver_origin;
    vec3 normal = receiver_normal;
    float primary_distance = -1.0;

    for (int b = 0; b < PT_BOUNCES; ++b) {
        vec2 u = vec2(random_dimension(seed, 3 * b), random_dimension(seed, 3 * b + 1));
        vec3 direction = sample_cosine_hemisphere(normal, u);
        float reach = b == 0 ? rt_primary_max_distance : rt_bounce_max_distance;
        VoxelHit hit = trace_voxel_segment(grid, origin, direction, reach, voxel_ray_path);
        radiance += hit.passed_emission * throughput;
        vec3 endpoint = origin + hit.distance * direction;

        if (hit.status != voxel_hit_surface) {
            if (hit.status == voxel_step_limit || hit.status == voxel_invalid_ray
                || (b == 0 && hit.status == voxel_inside_solid)) {
                hit_source = rtgi_hit_invalid;
            } else if (hit.status != voxel_inside_solid) {
                vec3 environment = voxel_environment(sky_map, direction, endpoint, receiver_skylight);
                radiance += environment * hit.transmittance * throughput;
                if (b == 0) {
                    hit_source = hit.status == voxel_volume_exit ? rtgi_hit_volume_exit : rtgi_hit_distance_limit;
                }
            }
            break;
        }

        if (b == 0) {
            hit_source = rtgi_hit_voxel;
            primary_distance = length(endpoint - receiver);
        }

        VoxelSample sample_data = unpack_voxel_data(hit.payload);
        VoxelMaterial material = get_voxel_material(sample_data.block_id, sample_data.color);
        vec3 emission = material.emission;
#ifdef RTGI_EMITTER_SAMPLING
        if (b == 0 && sample_emitters && max_of(emission) > 0.0) {
            float light_pdf = primary_emitter_pdf(hit, origin, normal, direction);
            if (light_pdf > 0.0) {
                float bsdf_pdf = max0(dot(normal, direction)) * rcp_pi;
                emission *= power_mis(float(RTGI_RAY_COUNT) * bsdf_pdf, light_pdf);
            }
        }
#endif

        vec3 next_origin = endpoint + voxel_bounce_bias * hit.normal;
        vec3 sun = vec3(0.0);
        float sun_cosine = dot(hit.normal, light_direction);
        vec3 safe_sun_color = max0(direct_sun_color);
        if (sun_cosine > 0.0 && max_of(safe_sun_color) > 0.0) {
            vec3 visibility = voxel_sun_visibility(grid, next_origin, light_direction);
            // Cap artistic sunlight bounce boost per hit so high-albedo surfaces
            // never exceed unit reflectance and create energy in 90-degree corners,
            // while darker stone/foliage retains full cave-bounce reach and hue.
            float albedo_peak = max_of(material.albedo);
            float sun_boost = albedo_peak > 0.0
                ? min(SUN_INDIRECT_BOOST, max(1.0, 0.92 / albedo_peak))
                : 1.0;
            sun = material.albedo * safe_sun_color * visibility
                * (sun_boost * rcp_pi * sun_cosine);
        }

        throughput *= hit.transmittance;
        radiance += (emission + sun) * throughput;
        if (b == PT_BOUNCES - 1) {
            break;
        }

        throughput *= material.albedo;
        if (b >= 1) {
            float survival = clamp(1.6 * max_of(throughput), 0.35, 1.0);
            if (random_dimension(seed, 3 * b + 2) >= survival) {
                break;
            }
            throughput /= survival;
        }
        origin = next_origin;
        normal = hit.normal;
    }

    return vec4(radiance, primary_distance);
}

#endif
