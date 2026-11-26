#if !defined INCLUDE_WEATHER_CLOUDS
#define INCLUDE_WEATHER_CLOUDS

#include "/include/sky/clouds/constants.glsl"
#include "/include/sky/clouds/parameters.glsl"
#include "/include/utility/color.glsl"
#include "/include/weather/core.glsl"

float clouds_cumulus_congestus_blend(Weather weather, vec2 l0_coverage) {
    float temperature_weight = linear_step(0.5, 1.0, weather.temperature);
    float humidity_weight = linear_step(0.3, 0.9, weather.humidity);
    float wind_weight = sqr(weather.wind);
    float l0_high_coverage
        = linear_step(0.45, 0.5, dot(l0_coverage, vec2(0.66, 0.33)));

    return clamp01(
        1.5 * dampen(dampen(temperature_weight * humidity_weight * wind_weight))
        * (1.0 - l0_high_coverage)
    );
}

float clouds_l0_cumulus_stratus_blend(Weather weather) {
    float temperature_weight
        = dampen(linear_step(0.5, 1.0, 1.0 - weather.temperature));
    float wind_weight = dampen(1.0 - weather.wind);

    return clamp01(temperature_weight * wind_weight);
}

vec2 clouds_l0_coverage(Weather weather, float cumulus_congestus_blend) {
    // very high temperature -> lower coverage
    // higher humidity -> higher coverage
    float temperature_weight
        = 1.0 - 0.15 * linear_step(0.6, 1.0, weather.temperature);
    float humidity_weight
        = 0.4 * weather.humidity + 0.5 * sqr(weather.humidity);
    float stratus_sheet = sqr(clouds_l0_cumulus_stratus_blend(weather));
    vec2 local_variation = vec2(-0.1, 1.0) * (0.1 + 0.1 * weather.wind);

    return clamp01(
               temperature_weight * humidity_weight + local_variation
               + 0.3 * stratus_sheet
           )
        * CLOUDS_CUMULUS_COVERAGE;
}

vec2 clouds_l0_detail_weights(Weather weather, float cumulus_stratus_blend) {
    float wind_torn_factor
        = linear_step(0.66, 0.9, weather.wind) * (1.0 - cumulus_stratus_blend);

    return mix(vec2(0.33, 0.40) * (1.0 + 0.5 * wind_torn_factor),
               vec2(0.07, 0.10),
               vec2(sqr(cumulus_stratus_blend), cumulus_stratus_blend))
        * CLOUDS_CUMULUS_DETAIL_STRENGTH;
}

vec2 clouds_l0_edge_sharpening(Weather weather, float cumulus_stratus_blend) {
    return mix(vec2(3.0, 12.0), vec2(2.0, 7.0), sqr(cumulus_stratus_blend));
}

float clouds_l0_altitude_scale(Weather weather, vec2 coverage) {
    float dynamic_thickness
        = mix(0.5, 1.0, smoothstep(0.4, 0.6, dot(coverage, vec2(0.25, 0.75))));
    return 0.8 * rcp(dynamic_thickness * clouds_cumulus_thickness);
}

float clouds_l1_cumulus_stratus_blend(Weather weather) {
    float temperature_weight
        = linear_step(0.5, 0.66, 1.0 - weather.temperature);

    return cubic_smooth(temperature_weight);
}

vec2 clouds_l1_coverage(Weather weather, float cumulus_stratus_blend) {
    // Altocumulus: high humidity, high wind, not too high temperature
    vec2 coverage_ac = (linear_step(0.4, 1.0, weather.humidity) * 0.5 + 0.5)
        * linear_step(0.4, 0.66, weather.wind)
        * (linear_step(1.0, 0.8, weather.temperature)) * vec2(0.5, 1.5);

    // Altostratus: high wind, high humidity
    vec2 coverage_as
        = vec2(linear_step(0.25, 0.45, weather.wind * weather.humidity));

    return clamp01(mix(coverage_ac, coverage_as, cumulus_stratus_blend))
        * CLOUDS_ALTOCUMULUS_COVERAGE;
}

float clouds_cirrus_amount(Weather weather) {
    float temperature_weight = 0.6
        - 0.4 * sqr(linear_step(0.5, 0.9, weather.temperature))
        + 0.4 * (1.0 - linear_step(0.0, 0.2, weather.temperature));
    float humidity_weight
        = 1.0 - 0.5 * linear_step(0.5, 0.75, weather.humidity);

    return clamp01(
               0.5 * temperature_weight * humidity_weight + 0.5 * rainStrength
           )
        * CLOUDS_CIRRUS_COVERAGE;
}

float clouds_cirrocumulus_amount(Weather weather) {
    float temperature_weight
        = 1.0 - 0.3 * linear_step(0.5, 1.0, weather.temperature);
    float humidity_weight = 0.5 + 0.5 * linear_step(0.4, 1.0, weather.humidity);
    float wind_weight = pow1d5(weather.wind);

    return 0.8 * dampen(temperature_weight * humidity_weight * wind_weight)
        * CLOUDS_CIRROCUMULUS_COVERAGE;
}

float clouds_noctilucent_amount() {
    float intensity = hash1(fract(float(worldDay) * golden_ratio));
    intensity = linear_step(CLOUDS_NOCTILUCENT_RARITY, 1.0, intensity);

    return dampen(intensity) * CLOUDS_NOCTILUCENT_INTENSITY;
}

CloudsParameters get_clouds_parameters(Weather weather) {
    CloudsParameters params;

    // Shaping parameters

    // Volumetric layer 0 - cumulus/stratocumulus/stratus
    params.l0_cumulus_stratus_blend = clouds_l0_cumulus_stratus_blend(weather);
    params.l0_coverage
        = clouds_l0_coverage(weather, params.cumulus_congestus_blend);
    params.l0_detail_weights
        = clouds_l0_detail_weights(weather, params.l0_cumulus_stratus_blend);
    params.l0_edge_sharpening
        = clouds_l0_edge_sharpening(weather, params.l0_cumulus_stratus_blend);
    params.l0_altitude_scale
        = clouds_l0_altitude_scale(weather, params.l0_coverage);

    // Volumetric layer 1 - altocumulus/altostratus/undulatus
    params.l1_cumulus_stratus_blend = clouds_l1_cumulus_stratus_blend(weather);
    params.l1_coverage
        = clouds_l1_coverage(weather, params.l1_cumulus_stratus_blend);

    // Planar clouds
    params.cirrus_amount = clouds_cirrus_amount(weather);
    params.cirrocumulus_amount = clouds_cirrocumulus_amount(weather);
    params.noctilucent_amount = clouds_noctilucent_amount();

    params.cumulus_congestus_blend
        = clouds_cumulus_congestus_blend(weather, params.l0_coverage);

    // Lighting parameters

    params.l0_shadow
        = linear_step(
              0.7,
              1.0,
              dot(params.l1_coverage, vec2(0.25, 0.75))
                  * (1.0 + params.l1_cumulus_stratus_blend)
          )
        * dampen(day_factor);
    params.l0_extinction_coeff
        = mix(0.05, 0.1, smoothstep(0.0, 0.3, abs(sun_dir.y)))
        * (1.0 - 0.33 * rainStrength) * (1.0 - 0.6 * params.l0_shadow)
        * CLOUDS_CUMULUS_DENSITY;
    params.l0_extinction_coeff *= 1.0
        - 0.4
            * linear_step(
                0.8,
                1.0,
                params.l0_coverage.y
                    - params.l0_coverage.y * params.l0_cumulus_stratus_blend
            );
    params.l0_extinction_coeff *= 1.0 - 0.5 * params.l0_cumulus_stratus_blend;
    params.l0_scattering_coeff
        = params.l0_extinction_coeff * mix(1.00, 0.66, rainStrength);

    // Crepuscular rays

    params.crepuscular_rays_amount = dampen(clamp01(2.0 * weather.humidity))
        * (linear_step(
            0.45,
            0.7,
            (1.0 + 0.2 * sqr(params.l0_cumulus_stratus_blend))
                    * dot(params.l0_coverage, vec2(0.66, 0.33))
                + 0.09 * params.l1_coverage.y
        ));

    return params;
}

// Macro sky overcast factor in [0, 1] driven by dynamic cloud deck coverage
// and rainStrength. Stays near 0 on clear/scattered cumulus days, ramps
// smoothly to ~0.85 on heavy dry stratus/altostratus overcast, and reaches
// 1.0 in rain.
float get_overcast_factor(CloudsParameters params) {
    float l0_cover = 0.0;
#ifdef CLOUDS_CUMULUS
    l0_cover = clamp01(
        dot(params.l0_coverage, vec2(0.55, 0.45))
        * (1.0 + 0.30 * params.l0_cumulus_stratus_blend)
    );
#endif
    float l1_cover = 0.0;
#ifdef CLOUDS_ALTOCUMULUS
    l1_cover = clamp01(
        dot(params.l1_coverage, vec2(0.55, 0.45))
        * (1.0 + 0.25 * params.l1_cumulus_stratus_blend)
    );
#endif
    float combined_cover = 1.0 - (1.0 - l0_cover) * (1.0 - 0.65 * l1_cover);
    float dry_overcast = smoothstep(0.50, 0.90, combined_cover);
    return clamp01(max(dry_overcast * 0.85, rainStrength));
}

// Macro direct sun/moon attenuation from overcast haze & rain (applies to
// both sun and moon). When CLOUD_SHADOWS is enabled, per-pixel colortex8
// handles local cloud cores and clear breaks while this provides macro haze
// and full rain extinction.
float get_overcast_direct_attenuation(float overcast) {
#ifdef CLOUD_SHADOWS
    float dry_haze = mix(1.0, 0.70, smoothstep(0.30, 0.95, overcast));
#else
    float dry_haze = mix(1.0, 0.16, smoothstep(0.15, 0.90, overcast));
#endif
    float rain_atten = 1.0 - 0.94 * smoothstep(0.0, 0.85, rainStrength);
    return dry_haze * rain_atten;
}

// Scales cloud/atmosphere direct sunlight & moonlight scattering during heavy
// dry overcast and rain so dry overcast stays luminous silver-gray (~35-45%
// dimmer than clear sun) while rain/storms step noticeably darker (~65-75%).
float get_overcast_cloud_light_scale(float overcast) {
    return mix(1.0, 0.86, smoothstep(0.25, 0.95, overcast))
        * mix(1.0, 0.54, smoothstep(0.0, 0.90, rainStrength));
}

// Blends and dims overhead sky/cloud ambient color during dry overcast & rain
// so RTGI sky illumination cools and drops ~35-50% on heavy dry overcast and
// ~60-75% in rain.
vec3 apply_overcast_sky_color(
    vec3 clear_sky_color,
    vec3 weather_sky_color,
    float overcast
) {
    // Physically, an overcast cloud deck cannot emit light: during twilight
    // and night, keep the weather target luminance bounded by clear-sky
    // irradiance (preserving weather chromaticity and moon-phase scaling) so
    // overcast clouds and fog never glow brighter than a clear night sky.
    float clear_luma = dot(clear_sky_color, luminance_weights_rec2020);
    float weather_luma = dot(weather_sky_color, luminance_weights_rec2020);
    float night_cap = smoothstep(0.05, -0.05, sun_dir.y);
    float capped_scale = min(weather_luma, clear_luma) / max(weather_luma, eps);
    weather_sky_color *= mix(1.0, capped_scale, night_cap);

    float blend = clamp01(
        max(0.45 * smoothstep(0.20, 0.90, overcast), rainStrength)
    );
    vec3 blended = mix(clear_sky_color, weather_sky_color, blend);
    float dim = mix(1.0, 0.90, smoothstep(0.30, 0.95, overcast))
        * mix(1.0, 0.60, smoothstep(0.0, 0.90, rainStrength));
    return blended * dim;
}

#endif // INCLUDE_WEATHER_CLOUDS
