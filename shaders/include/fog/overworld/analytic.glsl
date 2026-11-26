#if !defined INCLUDE_FOG_AIR_FOG_ANALYTIC
#define INCLUDE_FOG_AIR_FOG_ANALYTIC

#include "/include/fog/overworld/constants.glsl"
#include "/include/misc/lod_mod_support.glsl"
#include "/include/sky/atmosphere.glsl"
#include "/include/utility/phase_functions.glsl"

vec2 air_fog_analytic_airmass(
    vec3 ray_origin_world,
    vec3 ray_direction_world,
    float ray_length
) {
    // Exact piecewise integral of raymarched.glsl's height-clamped density
    // exp2(min((y - falloff_start) / -half_life, 0.0)) along [0, ray_length]
    // so analytic reflection/sky-map fog never spikes above VL near sea level.

    const vec2 mul = -rcp(air_fog_falloff_half_life);
    const vec2 add = -mul * air_fog_falloff_start;

    float y0 = ray_origin_world.y;
    float dy = ray_length * ray_direction_world.y;
    float y_lo = min(y0, y0 + dy);
    float y_hi = max(y0, y0 + dy);
    float abs_vy = abs(ray_direction_world.y);

    vec2 integral;
    if (abs_vy < 1e-5) {
        integral = ray_length * exp2(min(y0 * mul + add, 0.0));
    } else {
        float rcp_abs_vy = rcp(abs_vy);
        vec2 len_below
            = clamp(air_fog_falloff_start - y_lo, 0.0, y_hi - y_lo) * rcp_abs_vy;
        vec2 ya = max(vec2(y_lo), air_fog_falloff_start);
        vec2 yb = max(vec2(y_hi), air_fog_falloff_start);
        vec2 p_a = exp2(ya * mul + add);
        vec2 p_b = exp2(yb * mul + add);
        vec2 len_above = (p_a - p_b) * rcp(-log(2.0) * mul * abs_vy);
        integral = len_below + len_above;
    }

    return clamp(integral, 0.0, ray_length) * (0.5 * OVERWORLD_FOG_INTENSITY);
}

mat2x3 air_fog_analytic(
    vec3 ray_origin_world,
    vec3 ray_end_world,
    bool sky,
    float skylight,
    float shadow
) {
#ifdef LOD_MOD_ACTIVE
    float fog_end = float(lod_render_distance);
#else
    float fog_end = far;
#endif

    vec3 ray_direction_world;
    float ray_length;
    length_normalize(
        ray_end_world - ray_origin_world,
        ray_direction_world,
        ray_length
    );
    ray_length = sky ? fog_end : ray_length;

    vec2 airmass = air_fog_analytic_airmass(
        ray_origin_world,
        ray_direction_world,
        ray_length
    );
    vec3 optical_depth = fog_params.rayleigh_scattering_coeff * airmass.x
        + fog_params.mie_extinction_coeff * airmass.y;
    vec3 transmittance = exp(-optical_depth);
    vec3 scattering_integral = (1.0 - transmittance) / max(optical_depth, eps);

    vec3 rayleigh_scattering = scattering_integral * airmass.x
        * fog_params.rayleigh_scattering_coeff;
    vec3 mie_scattering
        = scattering_integral * airmass.y * fog_params.mie_scattering_coeff;

    float LoV = dot(ray_direction_world, light_dir);
    float mie_phase = 0.7 * henyey_greenstein_phase(LoV, 0.5)
        + 0.3 * henyey_greenstein_phase(LoV, -0.2);

    /*
    // Single scattering
    vec3 scattering  = light_color * (light_sun * vec2(isotropic_phase,
    mie_phase)); scattering += ambient_color * (light_sky *
    vec2(isotropic_phase));
    /*/
    // Multiple scattering
    vec3 scattering = vec3(0.0);
    float scatter_amount = 1.0;
    float anisotropy = 1.0;

    scattering += 2.0 * (rayleigh_scattering + mie_scattering) * isotropic_phase
        * ambient_color;

    for (int i = 0; i < 4; ++i) {
        float mie_phase = 0.7 * henyey_greenstein_phase(LoV, 0.5 * anisotropy)
            + 0.3 * henyey_greenstein_phase(LoV, -0.2 * anisotropy);

        scattering += scatter_amount
            * (rayleigh_scattering * isotropic_phase
               + mie_scattering * mie_phase)
            * light_color * shadow;

        scatter_amount *= 0.5;
        anisotropy *= 0.7;
    }
    //*/

    scattering *= max(skylight, eye_skylight);
    scattering *= clamp01(1.0 - blindness - darknessFactor);

    // Artifically brighten fog in the early morning and evening (looks nice)
    float evening_glow
        = 0.75 * linear_step(0.05, 1.0, exp(-300.0 * sqr(sun_dir.y + 0.02)));
    scattering += scattering * evening_glow;

    return mat2x3(max0(scattering), max0(transmittance));
}

#endif // INCLUDE_FOG_AIR_FOG_ANALYTIC
