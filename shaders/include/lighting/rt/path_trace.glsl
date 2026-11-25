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
    vec3 near_scene = voxel_to_scene_space(origin) + direction * 2.5;
    vec3 shadow_clip = project_ortho(shadowProjection, transform(shadowModelView, near_scene));
    if (all(lessThan(abs(shadow_clip), vec3(0.99)))) {
        vec3 screen = distort_shadow_space(shadow_clip) * 0.5 + 0.5;
        float depth_bias = 0.5 * abs(shadowProjection[2].z) * (SHADOW_DEPTH_SCALE * 0.5);
        macro_visibility = texture(shadowtex1, vec3(screen.xy, screen.z - depth_bias));
        if (macro_visibility < 0.01) {
            return vec3(0.0);
        }
#ifdef SHADOW_COLOR
        ivec2 shadow_size = textureSize(shadowtex0, 0);
        ivec2 shadow_texel = clamp(ivec2(screen.xy * vec2(shadow_size)), ivec2(0), shadow_size - 1);
        if (texelFetch(shadowtex0, shadow_texel, 0).x < screen.z - depth_bias) {
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
    // Prefer voxel glass transmittance within `reach` to avoid double-tinting;
    // fall back to shadowcolor0 for distant stained glass beyond `reach`.
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
    sky = (!any(isnan(sky)) && !any(isinf(sky))) ? max0(sky) : vec3(0.0);
#if defined WORLD_OVERWORLD
    // Wire SKYLIGHT_I into traced sky GI and slightly tame/neutralize the cool
    // night sky dome so night terrain matches stock Photon's deeper contrast.
    float night = clamp01(time_midnight);
    if (night > 0.0) {
        float sky_luma = dot(sky, luminance_weights_rec2020);
        vec3 night_sky = mix(sky, vec3(sky_luma), 0.30) * 0.72;
        sky = mix(sky, night_sky, night);
    }
    sky *= SKYLIGHT_I;
#endif
    return sky;
}

struct PathTraceSample {
    vec3 direct_rgb;
    vec3 indirect_radiance;
    vec3 primary_transmittance;
    vec3 first_pos_or_dir;
    vec3 first_normal;
    float primary_distance;
    int hit_source;
    ivec3 bounce_emitter_cell;
};

vec3 clamp_bounce_radiance(vec3 rad, int bounce) {
    if (RTGI_MAX_RADIANCE <= 0.0) {
        return rad;
    }
    // Progressive indirect luminance cap (path regularization) so rare bounce-2/3
    // outliers cannot hijack ReSTIR GI, while leaving direct light unconstrained.
    float cap = bounce == 0 ? 0.25 * RTGI_MAX_RADIANCE
              : bounce == 1 ? 0.14 * RTGI_MAX_RADIANCE
                            : 0.07 * RTGI_MAX_RADIANCE;
    float lum = dot(rad, luminance_weights);
    return lum > cap ? rad * (cap / lum) : rad;
}

// Traces one primary cosine-hemisphere path, splitting direct-emitter MIS
// (`direct_rgb`) from un-tinted first-bounce indirect radiance for ReSTIR GI.
PathTraceSample evaluate_path_trace(
    vec3 receiver, vec3 receiver_origin, vec3 receiver_normal, float receiver_skylight,
    usampler3D grid, sampler2D sky_map, vec3 direct_sun_color, vec3 light_direction,
    uint seed, vec2 primary_u, bool sample_emitters, bool receiver_two_sided, int max_bounces
) {
    PathTraceSample result = PathTraceSample(
        vec3(0.0), vec3(0.0), vec3(1.0), vec3(0.0), vec3(0.0), -1.0, rtgi_hit_none, ivec3(-1)
    );
    vec3 throughput = vec3(1.0);
    vec3 origin = receiver_origin;
    vec3 normal = receiver_normal;

    for (int b = 0; b < max_bounces; ++b) {
        vec2 u = b == 0 ? primary_u
                        : vec2(random_dimension(seed, 3 * b), random_dimension(seed, 3 * b + 1));
        vec3 hemi_normal = normal;
        if (b == 0 && receiver_two_sided
            && random_dimension(seed, 12) >= rtgi_leaf_front_prob) {
            hemi_normal = -normal;
        }
        vec3 direction = sample_cosine_hemisphere(hemi_normal, u);
        float reach = b == 0 ? rt_primary_max_distance : rt_bounce_max_distance;
        VoxelHit hit = trace_voxel_segment(grid, origin, direction, reach, voxel_ray_path);
        if (b == 0) {
            result.direct_rgb += hit.passed_emission;
        } else {
            result.indirect_radiance += clamp_bounce_radiance(hit.passed_emission * throughput, b);
        }
        vec3 endpoint = origin + hit.distance * direction;

        if (hit.status != voxel_hit_surface) {
            if (hit.status == voxel_step_limit || hit.status == voxel_invalid_ray
                || (b == 0 && hit.status == voxel_inside_solid)) {
                result.hit_source = rtgi_hit_invalid;
            } else if (hit.status != voxel_inside_solid) {
                vec3 environment = voxel_environment(sky_map, direction, endpoint, receiver_skylight);
                if (b == 0) {
                    result.primary_transmittance = hit.transmittance;
                    result.indirect_radiance = environment;
                    result.first_pos_or_dir = direction;
                    result.hit_source = hit.status == voxel_volume_exit
                        ? rtgi_hit_volume_exit : rtgi_hit_distance_limit;
                } else {
                    result.indirect_radiance += clamp_bounce_radiance(
                        environment * hit.transmittance * throughput, b
                    );
                }
            }
            break;
        }

        if (b == 0) {
            result.hit_source = rtgi_hit_voxel;
            result.primary_distance = length(endpoint - receiver);
            result.primary_transmittance = hit.transmittance;
            result.first_pos_or_dir = endpoint;
            result.first_normal = hit.normal;
        }

        VoxelSample sample_data = unpack_voxel_data(hit.payload);
        VoxelMaterial material = get_voxel_material(
            sample_data.block_id, sample_data.color, sample_data.emitter_box
        );
        vec3 emission = material.emission;
#ifdef RTGI_EMITTER_SAMPLING
        if (sample_emitters && max_of(emission) > 0.0
            && (b == 0 || hit.distance < rt_bounce_max_distance)) {
            bool ts_bounce = (b == 0) && receiver_two_sided;
            float light_pdf = primary_emitter_pdf(
                grid, hit, origin, ts_bounce ? vec3(0.0) : normal, direction, reach
            );
            if (light_pdf > 0.0) {
                float bsdf_pdf = rtgi_receiver_cos(normal, direction, ts_bounce) * rcp_pi;
                float mis_rays = b == 0 ? float(RTGI_RAY_COUNT) : 1.0;
                emission *= power_mis(mis_rays * bsdf_pdf, emitter_nee_mis_bias * light_pdf);
            }
        }
        if ((b >= 1 || !sample_emitters) && max_of(emission) > 0.0 && emitter_index_member(hit.payload)
            && result.bounce_emitter_cell.x < 0) {
            result.bounce_emitter_cell = hit.cell;
        }
#endif

        vec3 next_origin = endpoint + voxel_bounce_bias * hit.normal;
        vec3 sun = vec3(0.0);
        float sun_cosine = dot(hit.normal, light_direction);
        vec3 safe_sun_color = max0(direct_sun_color);
        if (sun_cosine > 0.0 && max_of(safe_sun_color) > 0.0) {
#if defined WORLD_OVERWORLD && defined CLOUD_SHADOWS && defined RTGI_CLOUD_SHADOWS
            float cloud_vis
                = get_cloud_shadows_fast(colortex8, voxel_to_scene_space(next_origin));
#else
            const float cloud_vis = 1.0;
#endif
            if (cloud_vis > 0.005) {
                vec3 sun_ray_dir = light_direction;
#if INFO >= 3
                if (b == 0) {
                    float cloud_scatter = clamp01(1.0 - cloud_vis);
                    float sun_half_angle
                        = (radians(SUN_ANGULAR_RADIUS) + 0.14 * sqr(cloud_scatter))
                        * max(SHADOW_PENUMBRA_SCALE, 0.0);
                    if (sun_half_angle > 1e-4) {
                        float r_u = sqrt(random_dimension(seed, 16 + 2 * b));
                        float phi_u = tau * random_dimension(seed, 17 + 2 * b);
                        mat3 sun_basis = build_orthonormal_basis(light_direction);
                        float tan_r = tan(min(sun_half_angle, 0.35)) * r_u;
                        vec3 cand_dir = normalize(
                            light_direction
                            + tan_r * (cos(phi_u) * sun_basis[0] + sin(phi_u) * sun_basis[1])
                        );
                        if (dot(hit.normal, cand_dir) > 0.0) {
                            sun_ray_dir = cand_dir;
                        }
                    }
                }
#endif
                vec3 visibility
                    = voxel_sun_visibility(grid, next_origin, sun_ray_dir) * cloud_vis;
                // Cap sunlight bounce boost so bright surfaces never exceed unit
                // reflectance in corners while dark stone/foliage keeps cave reach.
                float albedo_peak = max_of(material.albedo);
                float sun_boost = albedo_peak > 0.0
                    ? min(SUN_INDIRECT_BOOST, max(1.0, 0.92 / albedo_peak))
                    : 1.0;
                sun = material.albedo * safe_sun_color * visibility
                    * (sun_boost * rcp_pi * sun_cosine);
            }
        }

        // Non-terminal bounce emitter NEE paired with BSDF MIS on the next bounce
        // to eliminate un-MIS'd emitter fireflies at PT_BOUNCES >= 2.
        vec3 bounce_emitter = vec3(0.0);
#ifdef RTGI_EMITTER_SAMPLING
        if (b < max_bounces - 1 && sample_emitters && max_of(material.albedo) > 0.0) {
            ivec3 b_cell;
            uint nee_seed = lowbias32(seed ^ (0x9e3779b9u + uint(b) * 0x85ebca6bu));
            vec4 b_sample = sample_direct_emitter_ex(
                grid, next_origin, hit.normal,
                nee_seed,
                1.0, rt_bounce_max_distance, b_cell
            );
            if (b_sample.a > 1.5) {
                bounce_emitter = clamp_bounce_radiance(material.albedo * b_sample.rgb, b);
                if (result.bounce_emitter_cell.x < 0) {
                    result.bounce_emitter_cell = b_cell;
                }
            }
        }
#endif

        if (b == 0) {
            if (sample_emitters) {
                // Keep primary MIS-weighted direct emission out of ReSTIR GI
                // so ReSTIR DI and ReSTIR GI remain cleanly separated and unbiased.
                result.direct_rgb += emission * hit.transmittance;
                result.indirect_radiance += sun + bounce_emitter;
            } else {
                result.indirect_radiance += emission + sun + bounce_emitter;
            }
        } else {
            throughput *= hit.transmittance;
            result.indirect_radiance += clamp_bounce_radiance(
                (emission + sun + bounce_emitter) * throughput, b
            );
        }
        if (b == max_bounces - 1) {
#ifdef RTGI_WORLD_CACHE
            // Query face-oriented world-space residual radiance cache (D_block +
            // D_indirect, excluding sun) for multi-bounce GI without double-counting.
            vec3 cached_d_gi;
            if (world_cache_query(world_cache_cell_from_voxel(hit.cell), hit.normal, cached_d_gi)) {
                vec3 safe_albedo = min(material.albedo, vec3(0.88));
                result.indirect_radiance += safe_albedo * cached_d_gi * throughput;
            }
#endif
            break;
        }

        throughput *= material.albedo;
        if (max_of(throughput) < 0.01) {
            break;
        }
        origin = next_origin;
        normal = hit.normal;
    }

    return result;
}

#endif
