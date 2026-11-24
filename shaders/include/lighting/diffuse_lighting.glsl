#if !defined INCLUDE_LIGHTING_DIFFUSE_LIGHTING
#define INCLUDE_LIGHTING_DIFFUSE_LIGHTING

#include "/include/lighting/bsdf.glsl"
#include "/include/lighting/colors/blocklight_color.glsl"
#include "/include/misc/end_lighting_fix.glsl"
#include "/include/surface/material.glsl"
#include "/include/utility/fast_math.glsl"
#include "/include/utility/phase_functions.glsl"


#if !defined WORLD_OVERWORLD
#undef CLOUD_SHADOWS
#endif

#if defined PROGRAM_DEFERRED4
#define DIFFUSE_USE_RTGI
uniform sampler2D rtgi_filtered_sampler;
uniform usampler2D rtgi_guide_even_sampler;
uniform usampler2D rtgi_guide_odd_sampler;
#include "/include/lighting/rt/rtgi_surface.glsl"
#include "/include/lighting/voxel/voxel_constants.glsl"
#endif

const float sss_density = 6.5;
const float sss_scale = 3.2 * SSS_INTENSITY;
const float night_vision_scale = 1.5;
const float metal_diffuse_amount
    = 0.5; // Scales diffuse lighting on metals, ideally this would be zero but
           // purely specular metals don't play well with SSR

#ifdef SHADOW_VPS
vec3 sss_approx(
    vec3 albedo,
    float sss_amount,
    float sheen_amount,
    float sss_depth,
    float LoV,
    float NoL,
    float shadow
) {
    // Transmittance-based SSS
    if (sss_amount < eps) {
        return vec3(0.0);
    }

    // Normalize by peak channel so dark green foliage and bright cherry petals
    // share smooth multi-block canopy penetration without an abrupt cliff.
    float albedo_peak = max(max_of(albedo), eps);
    vec3 coeff = 0.65 * clamp01(albedo / albedo_peak);
    coeff = (1.0 - coeff) * sss_density / sss_amount;

    float front_lit = clamp01(shadow * smoothstep(0.0, 0.35, NoL));
    float phase
        = mix(isotropic_phase, henyey_greenstein_phase(-LoV, 0.28), 0.22);

    vec3 transmittance = exp2(-coeff * sss_depth);
    vec3 sss = sss_scale * phase * transmittance * dampen(sss_amount) * pi
        * mix(1.0, 0.20, front_lit);

#ifdef SSS_SHEEN
    vec3 sheen = (0.22 * SSS_INTENSITY) * rcp(max(albedo, vec3(0.12)))
        * transmittance * henyey_greenstein_phase(-LoV, 0.30)
        * linear_step(-0.6, 0.4, -LoV);
    sss += sheen * sheen_amount * mix(1.0, 0.4, front_lit * (1.0 - sheen_amount));
#endif

    return sss;
}
#else
vec3 sss_approx(
    vec3 albedo,
    float sss_amount,
    float sheen_amount,
    float sss_depth,
    float LoV,
    float NoL,
    float shadow
) {
    // Blur-based SSS
    float front_lit = clamp01(shadow * smoothstep(0.0, 0.35, NoL));
    float sss = 0.06 * sss_scale * pi * mix(1.0, 0.25, front_lit);
    vec3 sheen = 0.8 * rcp(albedo + eps) * henyey_greenstein_phase(-LoV, 0.5)
        * linear_step(-0.8, -0.2, -LoV) * shadow;

    return sss + sheen * sheen_amount;
}
#endif

vec3 rtgi_fallback_radiance(vec2 light_levels, vec3 flat_normal, float cloud_shadows) {
    // Reuse the underlying lightmap falloff only where traced diffuse has no
    // compatible support (or for forward translucent callers). Keep the
    // fallback deterministic and in E/pi units.
    float blocklight = light_levels.x;
    float falloff = pow8(blocklight) + 0.18 * sqr(blocklight) + 0.16 * dampen(blocklight);
    vec3 block = (rcp_pi * blocklight_scale * falloff) * blocklight_color;

#if defined PROGRAM_DEFERRED4
#if defined WORLD_OVERWORLD
    // Match path_trace.glsl's sky-map cosine-hemisphere integral on open
    // surfaces and estimate warm sunlit ground bounce on vertical/underside
    // faces so the RT coverage boundary has no brightness drop or blue shift.
    float sky_vis = sqr(light_levels.y);
    vec3 sky_up = sky_fallback_color * SKYLIGHT_I;
    vec3 ground_bounce = 0.28 * (sky_up + max0(light_color) * (cloud_shadows * max0(light_dir.y) * rcp_pi));
    float up_factor = 0.5 + 0.5 * clamp(flat_normal.y, -1.0, 1.0);
    vec3 environment = mix(ground_bounce, sky_up, up_factor) * sky_vis;
#elif defined WORLD_NETHER
    vec3 environment = ambient_color;
#else
    vec3 environment = sky_fallback_color;
#endif
#else
#if defined WORLD_OVERWORLD
    vec3 environment = ambient_color * (rcp_pi * SKYLIGHT_I * sqr(light_levels.y));
#elif defined WORLD_NETHER
    vec3 environment = ambient_color;
#else
    vec3 environment = 0.1 * ambient_color;
#endif
#endif
    return block + environment;
}

vec3 rtgi_fallback_radiance(vec2 light_levels) {
    return rtgi_fallback_radiance(light_levels, vec3(0.0, 1.0, 0.0), 1.0);
}

#ifdef DIFFUSE_USE_RTGI
float rtgi_upsample_weight(
    ivec2 tap, RtgiGrid grid, vec2 jitter, bool even, vec3 view_pos,
    vec3 normal, uint material, float footprint, float max_chord
) {
    if (!rtgi_in_bounds(tap, grid.gi_active)) {
        return 0.0;
    }

    uvec4 guide = even ? texelFetch(rtgi_guide_even_sampler, tap, 0)
                       : texelFetch(rtgi_guide_odd_sampler, tap, 0);
    vec3 tap_view;
    if (!rtgi_guide_matches(guide, material)
        || !rtgi_view_from_guide(guide, grid.full_size, jitter, gbufferProjection, tap_view)) {
        return 0.0;
    }
    vec3 tap_normal = mat3(gbufferModelView) * rtgi_guide_normal(guide);
    return rtgi_geometry_weight(tap_view - view_pos, normal, tap_normal, footprint, max_chord);
}

vec3 sample_rtgi_diffuse(
    ivec2 pixel, ivec2 full_size, vec3 view_pos, vec3 scene_pos,
    vec3 flat_normal, uint material, uint kind, vec2 light_levels,
    float cloud_shadows
) {
    vec3 fallback = rtgi_fallback_radiance(light_levels, flat_normal, cloud_shadows);
    if (kind != rtgi_world || !rtgi_finite(vec4(scene_pos, view_pos.z)) || view_pos.z >= 0.0) {
        return fallback;
    }

    vec3 local = scene_to_voxel_space(scene_pos);
    vec3 edge = min(local, rt_voxel_size_f - local);
    float coverage = smoothstep(0.0, 8.0, min(edge.x, min(edge.y, edge.z)));
    RtgiGrid grid = rtgi_grid(full_size, textureSize(rtgi_filtered_sampler, 0));
    if (coverage <= 0.0 || !rtgi_in_bounds(pixel, grid.full_active)
        || any(equal(grid.gi_active, ivec2(0)))) {
        return fallback;
    }

    vec2 jitter = rtgi_raster_jitter(taa_offset, full_size);
    vec3 normal = mat3(gbufferModelView) * flat_normal;
    float footprint = rtgi_view_footprint(-view_pos.z, grid.gi_active, gbufferProjection);
    float grazing = max(abs(dot(normal, normalize(view_pos))), 0.02);
    float max_chord = min(6.0, 0.05 + 2.5 * footprint / grazing);
    bool even = (frameCounter & 1) == 0;

#if RTGI_RESOLUTION == 2
    float weight = rtgi_upsample_weight(pixel, grid, jitter, even, view_pos,
                                        normal, material, footprint, max_chord);
    if (weight > 0.0) {
        vec4 sample_value = texelFetch(rtgi_filtered_sampler, pixel, 0);
        if (sample_value.a >= 0.0) {
            return mix(fallback, sample_value.rgb, coverage);
        }
    }
#else
    vec2 center = vec2(pixel) + 0.5;
    ivec2 base;
    vec2 fraction;
    rtgi_footprint(center, grid, base, fraction);

    vec3 sum = vec3(0.0);
    float sum_weight = 0.0;
    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            ivec2 offset = ivec2(x, y);
            ivec2 tap = base + offset;
            float weight = rtgi_bilinear_weight(offset, fraction);
            if (weight <= 0.0) {
                continue;
            }
            weight *= rtgi_upsample_weight(tap, grid, jitter, even, view_pos,
                                           normal, material, footprint, max_chord);
            if (weight > 0.0) {
                vec4 sample_value = texelFetch(rtgi_filtered_sampler, tap, 0);
                if (sample_value.a >= 0.0) {
                    sum += weight * sample_value.rgb;
                    sum_weight += weight;
                }
            }
        }
    }
    if (sum_weight > 1e-4) {
        return mix(fallback, sum / sum_weight, coverage);
    }

    // A thin receiver may have no compatible anchor in the primary four.
    // Try the nearby independent anchors with the same plane and chord gates.
    ivec2 cell = rtgi_cell(pixel, grid);
    vec3 nearest = fallback;
    float best_score = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 tap = cell + ivec2(x, y);
            ivec2 offset = tap - base;
            if (all(greaterThanEqual(offset, ivec2(0))) && all(lessThanEqual(offset, ivec2(1)))
                && rtgi_bilinear_weight(offset, fraction) > 0.0) {
                continue;
            }
            float weight = rtgi_upsample_weight(tap, grid, jitter, even, view_pos,
                                                normal, material, footprint, max_chord);
            if (weight <= 0.0) {
                continue;
            }
            vec2 delta = vec2(rtgi_anchor(tap, grid) - pixel);
            float score = weight / (1.0 + dot(delta, delta));
            if (score > best_score) {
                vec4 sample_value = texelFetch(rtgi_filtered_sampler, tap, 0);
                if (sample_value.a >= 0.0) {
                    nearest = sample_value.rgb;
                    best_score = score;
                }
            }
        }
    }
    return mix(fallback, nearest, coverage);
#endif
    return fallback;
}
#endif


vec3 get_diffuse_lighting(
    Material material,
    vec3 scene_pos,
    vec3 normal,
    vec3 flat_normal,
    vec3 bent_normal,
    vec3 shadows,
    vec2 light_levels,
    float ao,
    float sss_depth,
#ifdef CLOUD_SHADOWS
    float cloud_shadows,
#endif
    float shadow_distance_fade,
    float NoL,
    float NoV,
    float NoH,
    float LoV,
    vec3 indirect_radiance
) {
#if defined PROGRAM_GBUFFERS_WATER
    // Small optimization, don't calculate diffuse lighting when albedo and
    // emission are both 0 (eg water)
    if (max_of(material.albedo) < eps && max_of(material.emission) < eps) {
        return vec3(0.0);
    }
#endif

    vec3 direct_lighting = vec3(0.0);

    // Arbitrary directional shading to make faces easier to distinguish
    float directional_lighting
        = (0.9 + 0.1 * normal.x) * (0.8 + 0.2 * abs(flat_normal.y));

    // Negative SSS depth => SSS blocked by occluder (SSRT SSS)
    bool sss_blocked = sss_depth < 0.0 || light_levels.y < 0.1;
    sss_depth = max0(sss_depth);

#if defined WORLD_OVERWORLD || defined WORLD_END

    // Sunlight/moonlight

#if defined SHADOW || defined SHADOW_SSRT
    vec3 diffuse = vec3(
        lift(max0(NoL), 0.25 * rcp(SHADING_STRENGTH))
        * (1.0 - 0.20 * material.sss_amount)
    );

    vec3 sss = sss_approx(
                   material.albedo,
                   material.sss_amount,
                   material.sheen_amount,
                   mix(sss_depth, 0.0, shadow_distance_fade),
                   LoV,
                   NoL,
                   shadows.x
               )
        * float(!sss_blocked);

    // Adjust SSS outside of shadow distance
    // RT contact AO may soften distant direct SSS, but never supplies an
    // ambient irradiance term.
    sss *= mix(1.0, ao * (clamp01(NoL) * 0.8 + 0.2), clamp01(shadow_distance_fade));

#ifdef RT_AO_IN_DIRECT
    diffuse *= sqr(ao);
#endif

#ifdef SHADOW_VPS
    // Add SSS and diffuse
    direct_lighting += diffuse * shadows + sss;
#else
    // Blend SSS and diffuse
    direct_lighting += mix(diffuse, sss, material.sss_amount) * shadows;
#endif
#else
    // Simple shading for when shadows are disabled
    vec3 sss = 0.08 * sss_scale * pi
        + 0.5 * material.sheen_amount * rcp(material.albedo + eps)
            * henyey_greenstein_phase(-LoV, 0.5)
            * linear_step(-0.8, -0.2, -LoV);

    vec3 diffuse
        = vec3(lift(max0(NoL), 0.5 * rcp(SHADING_STRENGTH)) * 0.6 + 0.4)
        * (shadows * 0.8 + 0.2);
    diffuse = diffuse + max0(sss * lift(material.sss_amount, 5.0));
    diffuse *= 1.0 * (0.9 + 0.1 * normal.x) * (0.8 + 0.2 * abs(flat_normal.y));
    diffuse *= ao * pow4(light_levels.y) * (dampen(light_dir.y) * 0.5 + 0.5);

    direct_lighting += diffuse;
#endif

    direct_lighting *= light_color;

#ifdef CLOUD_SHADOWS
    direct_lighting *= cloud_shadows;
#endif
#endif

#if defined WORLD_OVERWORLD
    // Cave/indoor ambient
    direct_lighting += 0.25 * CAVE_LIGHTING_I * directional_lighting * ao
        * (1.0 - light_levels.y * light_levels.y)
        * (1.0 - 0.7 * darknessFactor);
    direct_lighting += nightVision * night_vision_scale * directional_lighting * ao;
#endif

#if defined PROGRAM_DEFERRED4 && defined WORLD_OVERWORLD
    // Gentle overhead skylight transmission through translucent foliage so
    // shaded tree backsides and undersides stay smooth like stock Photon
    // without extra ray traces (naturally masked out indoors by light_levels.y).
    if (material.sss_amount > eps) {
        vec3 foliage_sky_fill = sky_fallback_color
            * (0.30 * SKYLIGHT_I * material.sss_amount * sqr(light_levels.y) * ao);
        indirect_radiance = mix(
            indirect_radiance,
            max(indirect_radiance, foliage_sky_fill),
            0.50 * material.sss_amount
        );
    }
#endif

    // The raster term is irradiance, and RTGI already includes the diffuse 1/pi.
    // Outgoing surface emission Le is already tinted in linear Rec.2020 and
    // must not be multiplied by diffuse albedo or metallic attenuation again.
    vec3 outgoing_emission = max0(material.emission) * (emission_scale * rcp_pi);
    vec3 final_diffuse = (max0(direct_lighting) * rcp_pi + indirect_radiance)
        * material.albedo * mix(1.0, metal_diffuse_amount, float(material.is_metal));
    return conserve_diffuse_energy(final_diffuse, material.f0, NoV)
        + outgoing_emission;
}

// Forward callers have no opaque RTGI receiver and use the shared coverage fallback.
vec3 get_diffuse_lighting(
    Material material,
    vec3 scene_pos,
    vec3 normal,
    vec3 flat_normal,
    vec3 bent_normal,
    vec3 shadows,
    vec2 light_levels,
    float ao,
    float sss_depth,
#ifdef CLOUD_SHADOWS
    float cloud_shadows,
#endif
    float shadow_distance_fade,
    float NoL,
    float NoV,
    float NoH,
    float LoV
) {
    return get_diffuse_lighting(
        material,
        scene_pos,
        normal,
        flat_normal,
        bent_normal,
        shadows,
        light_levels,
        ao,
        sss_depth,
#ifdef CLOUD_SHADOWS
        cloud_shadows,
#endif
        shadow_distance_fade,
        NoL,
        NoV,
        NoH,
        LoV,
        rtgi_fallback_radiance(light_levels)
    );
}

#ifdef DIFFUSE_USE_RTGI
#undef DIFFUSE_USE_RTGI
#endif

#endif // INCLUDE_LIGHTING_DIFFUSE_LIGHTING
