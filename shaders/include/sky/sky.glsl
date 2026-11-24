#if !defined INCLUDE_SKY_SKYsky
#define INCLUDE_SKY_SKY

#include "/include/utility/color.glsl"
#include "/include/utility/dithering.glsl"
#include "/include/utility/fast_math.glsl"
#include "/include/utility/random.glsl"

//----------------------------------------------------------------------------//
#if defined WORLD_OVERWORLD

#include "/include/lighting/bsdf.glsl"
#include "/include/lighting/colors/light_color.glsl"
#include "/include/lighting/colors/weather_color.glsl"
#include "/include/misc/lightning_flash.glsl"
#include "/include/sky/atmosphere.glsl"
#include "/include/sky/projection.glsl"
#include "/include/sky/rainbow.glsl"
#include "/include/sky/stars.glsl"
#include "/include/utility/geometry.glsl"

#if defined PROGRAM_DEFERRED0
#include "/include/sky/clouds.glsl"

#if defined CREPUSCULAR_RAYS && !defined BLOCKY_CLOUDS
#include "/include/sky/crepuscular_rays.glsl"
#endif
#endif

const float sun_luminance = rcp(
    cone_angle_to_solid_angle(sun_angular_radius)
); // luminance of sun disk (solid-angle normalized for constant flux)
const float moon_ref_solid_angle = cone_angle_to_solid_angle(3.0 * degree);
const float moon_luminance = 10.0
    * clamp(
        sqrt(moon_ref_solid_angle * rcp(cone_angle_to_solid_angle(moon_angular_radius))),
        1.0,
        6.0
    ); // scales luminance gently with smaller moon disc while preserving crater detail

vec3 draw_sun(vec3 ray_dir) {
    float nu = clamp(dot(ray_dir, sun_dir), -1.0, 1.0);
    if (nu < cos(sun_angular_radius)) {
        return vec3(0.0);
    }
    float angle = fast_acos(nu);

    // Limb darkening model from
    // http://www.physics.hmc.edu/faculty/esin/a101/limbdarkening.pdf
    // Normalize radius to [0, 1] and conserve integrated disc flux across
    // all SUN_ANGULAR_RADIUS values (matching stock 2.0-deg integrated brightness).
    const vec3 alpha = vec3(0.429, 0.522, 0.614);
    float r_norm = clamp01(angle * rcp(sun_angular_radius));
    float mu_limb = sqrt(max0(1.0 - sqr(r_norm)));
    vec3 limb_darkening
        = pow(vec3(max(mu_limb, 1e-4)), alpha) * (0.50 * (1.0 + 0.5 * alpha));

    // Clamp below R11F_G11F_B10F max (64512) so tiny angular radii / blue hour
    // never overflow colortex0 to Inf.
    return min(
        sun_luminance * sun_color * limb_darkening,
        vec3(48000.0)
    );
}

vec4 draw_moon(vec3 ray_dir) {
    const vec3 lit_color
        = vec3(MOON_R, MOON_G <= 0.03 ? 0.0 : MOON_G - 0.03, MOON_B);
    const vec3 glow_color
        = vec3(MOON_R <= 0.05 ? 0.0 : MOON_R - 0.05, MOON_G, MOON_B);

    // Cut out the moon disc.
    float MoV = clamp(dot(ray_dir, moon_dir), -1.0, 1.0);
    if (MoV < cos(moon_angular_radius)) {
        return vec4(0.0);
    }

    // Find distance from center to edge.
    float dist = clamp01(fast_acos(MoV) * rcp(moon_angular_radius));

    // Transform the coordinate space such that z is parallel to moon_dir
    vec3 tangent = moon_dir.y == 1.0
        ? vec3(1.0, 0.0, 0.0)
        : normalize(cross(vec3(0.0, 1.0, 0.0), moon_dir));
    vec3 bitangent = normalize(cross(tangent, moon_dir));
    mat3 tbn = mat3(tangent, bitangent, moon_dir);

    // Vector from moon center scaled by angular radius so crater scale stays
    // consistent across MOON_ANGULAR_RADIUS settings.
    vec2 offset = ((ray_dir - moon_dir) * tbn).xy
        * (3.0 * degree * rcp(moon_angular_radius));
    offset = fract(offset + 0.5);

    vec3 noise = texture(noisetex, 2.0 * offset).xyz;
    float moon_texture
        = pow1d5(noise.x) * 0.75 + 0.6 * cube(noise.y) - 0.1 * noise.z;

    // Find the distance to the moon if it were 1 unit away, and its normal.
    float moon_dist
        = intersect_sphere(-moon_dir, ray_dir, moon_angular_radius).x;
    vec3 moon_normal = normalize(ray_dir * moon_dist - moon_dir);

    // Get light direction which orbits around moon
    float light_angle = 0.125 * tau * float(moonPhase);
    vec3 left_dir = normalize(cross(vec3(0.0, 1.0, 0.0), ray_dir));
    vec3 light_dir = cos(light_angle) * -ray_dir + sin(light_angle) * left_dir;
    float moon_shadow = dampen(max0(dot(moon_normal, light_dir)));

    float edge_glow = sqr(sqr(sqr(dist)));

    vec3 color = max(moon_shadow * lit_color * (1.0 + 4.0 * edge_glow),
                     0.5 * glow_color * (0.1 + 0.1 * edge_glow))
        * (0.2 + 0.8 * moon_texture);
    color = moon_luminance * sqr(color);
    return vec4(color, 1.0);
}

#ifdef GALAXY
vec3 draw_galaxy(vec3 ray_dir, out float galaxy_luminance) {
    const vec3 galaxy_tint = vec3(0.75, 0.66, 1.0) * GALAXY_INTENSITY;

    float galaxy_intensity = 0.05 + 1.0 * linear_step(-0.1, 0.25, -sun_dir.y);

    float lon = atan(ray_dir.x, ray_dir.z);
    float lat = fast_acos(-ray_dir.y);

    vec3 galaxy
        = texture(galaxy_sampler, vec2(lon * rcp(tau) + 0.5, lat * rcp(pi)))
              .rgb;

    galaxy = srgb_eotf_inv(galaxy) * rec709_to_working_color;

    galaxy *= galaxy_intensity * galaxy_tint;

    galaxy_luminance = dot(galaxy, luminance_weights_rec709);

    galaxy = mix(vec3(galaxy_luminance), galaxy, 2.0);

    return max0(galaxy);
}
#endif

vec3 draw_sky(
    vec3 ray_dir,
    vec3 atmosphere,
    vec4 clouds_and_aurora,
    float clouds_apparent_distance
) {
    vec3 sky = vec3(0.0);

#if defined SHADOW
    // Trick to make stars rotate with sun and moon
    mat3 rot = (sunAngle < 0.5)
        ? mat3(shadowModelViewInverse)
        : mat3(
              -shadowModelViewInverse[0].xyz,
              shadowModelViewInverse[1].xyz,
              -shadowModelViewInverse[2].xyz
          );

    vec3 celestial_dir = ray_dir * rot;
#else
    vec3 celestial_dir = ray_dir;
#endif

    // Galaxy

#ifdef GALAXY
    float galaxy_luminance;
    sky += draw_galaxy(celestial_dir, galaxy_luminance);
#else
    const float galaxy_luminance = 0.0;
#endif

    // Sun, moon stars

#if defined PROGRAM_DEFERRED4
    vec3 skytextured_output
        = texelFetch(colortex0, ivec2(gl_FragCoord.xy), 0).rgb;
    sky += texelFetch(colortex0, ivec2(gl_FragCoord.xy), 0).rgb;

#ifdef STARS
    // Stars
    float stars_visibility
        = clamp01(1.0 - dot(skytextured_output, vec3(0.33) * 256.0));
    sky += draw_stars(celestial_dir, galaxy_luminance) * stars_visibility;
#endif

#ifndef VANILLA_SUN
    // Shader sun
    sky += draw_sun(ray_dir);
#endif

#ifndef VANILLA_MOON
    // Shader moon
    vec4 moon = draw_moon(ray_dir);
    sky *= 1.0 - moon.a;
    sky += moon.rgb;
#endif
#endif

    // Atmosphere

    sky *= atmosphere_transmittance(ray_dir.y, planet_radius)
        * (1.0 - rainStrength);
    sky += atmosphere;

    // Clouds, aurora, crepuscular rays

    sky *= clouds_and_aurora.a; // Transmittance
    sky += clouds_and_aurora.rgb; // Scattering

    // Rainbow

    sky = draw_rainbows(
        sky,
        ray_dir,
        mix(clouds_apparent_distance,
            1e6,
            linear_step(1.0, 0.95, clouds_and_aurora.w))
    );

    // Cave sky fix

#if !defined PROGRAM_DEFERRED0
    // Fade lower part of sky into cave fog color when underground so that the
    // sky isn't visible beyond the render distance
    float underground_sky_fade
        = biome_cave * smoothstep(-0.1, 0.1, 0.4 - ray_dir.y);
    sky = mix(sky, vec3(0.0), underground_sky_fade);
#endif

    return sky;
}

#if defined PROGRAM_DEFERRED0
vec4 get_clouds_and_aurora(
    vec3 ray_dir,
    vec3 clear_sky,
    out float clouds_apparent_distance
) {
    clouds_apparent_distance = 1e6;

    ivec2 texel = ivec2(gl_FragCoord.xy);
    texel.x = texel.x % (sky_map_res.x - 4);

    float dither = interleaved_gradient_noise(vec2(texel));

    // Clouds

#ifndef BLOCKY_CLOUDS
    const vec3 air_viewer_pos = vec3(0.0, planet_radius, 0.0);
    CloudsResult result
        = draw_clouds(air_viewer_pos, ray_dir, clear_sky, -1.0, dither);

    // Lightning flash
    result.scattering.rgb += LIGHTNING_FLASH_UNIFORM * lightning_flash_intensity
        * result.scattering.a;
#else
    CloudsResult result = clouds_not_hit;
#endif

    clouds_apparent_distance = result.apparent_distance;

    // Aurora

    vec3 aurora = draw_aurora(ray_dir, dither);

    vec4 clouds_and_aurora = vec4(
        result.scattering.xyz + aurora * result.transmittance,
        result.transmittance
    );

    // Crepuscular rays

#if defined CREPUSCULAR_RAYS && !defined BLOCKY_CLOUDS
    vec4 crepuscular_rays
        = draw_crepuscular_rays(colortex8, ray_dir, false, 0.5);
    clouds_and_aurora *= crepuscular_rays.w;
    clouds_and_aurora.rgb += crepuscular_rays.xyz;
#endif

    return clouds_and_aurora;
}

vec3 draw_sky(vec3 ray_dir) {
    vec3 atmosphere = atmosphere_scattering(
        ray_dir,
        sun_color,
        sun_dir,
        moon_color,
        moon_dir,
        true
    );
    float clouds_apparent_distance;
    vec4 clouds_and_aurora
        = get_clouds_and_aurora(ray_dir, atmosphere, clouds_apparent_distance);
    return draw_sky(
        ray_dir,
        atmosphere,
        clouds_and_aurora,
        clouds_apparent_distance
    );
}
#endif

//----------------------------------------------------------------------------//
#elif defined WORLD_NETHER

vec3 draw_sky(vec3 ray_dir) { return ambient_color; }

//----------------------------------------------------------------------------//
#elif defined WORLD_END

#include "/include/misc/end_lighting_fix.glsl"
#include "/include/sky/atmosphere.glsl"
#include "/include/sky/stars.glsl"

const float sun_solid_angle = cone_angle_to_solid_angle(sun_angular_radius);
const vec3 end_sun_color = vec3(1.0, 0.5, 0.25);

vec3 draw_sun(vec3 ray_dir) {
    float nu = dot(ray_dir, sun_dir);
    float r = fast_acos(nu);

    // Sun disk

    const vec3 alpha = vec3(0.6, 0.5, 0.4);
    float center_to_edge = max0(sun_angular_radius - r);
    vec3 limb_darkening
        = pow(vec3(1.0 - sqr(1.0 - center_to_edge)), 0.5 * alpha);
    vec3 sun_disk = vec3(r < sun_angular_radius);

    // Solar flare effect

    // Transform the coordinate space such that z is parallel to sun_dir
    vec3 tangent = sun_dir.y == 1.0
        ? vec3(1.0, 0.0, 0.0)
        : normalize(cross(vec3(0.0, 1.0, 0.0), sun_dir));
    vec3 bitangent = normalize(cross(tangent, sun_dir));
    mat3 rot = mat3(tangent, bitangent, sun_dir);

    // Vector from ray dir to sun dir
    vec2 q = ((ray_dir - sun_dir) * rot).xy;

    float theta = fract(
        linear_step(-pi, pi, atan(q.y, q.x)) + 0.015 * frameTimeCounter
        - 0.33 * r
    );

    float flare
        = texture(noisetex, vec2(theta, r - 0.025 * frameTimeCounter)).x;
    flare = pow5(flare) * exp(-25.0 * (r - sun_angular_radius));
    flare = r < sun_angular_radius ? 0.0 : flare;

    return min(
        end_sun_color * rcp(sun_solid_angle) * max0(sun_disk + 0.1 * flare),
        vec3(48000.0)
    );
}

vec3 draw_sky(vec3 ray_dir) {
    // Sky gradient

    float up_gradient
        = linear_step(0.0, 0.4, ray_dir.y) + linear_step(0.1, 0.8, -ray_dir.y);
    vec3 sky = ambient_color * mix(0.1, 0.04, up_gradient);
    float mie_phase = cornette_shanks_phase(dot(ray_dir, sun_dir), 0.6);
    sky += 0.1 * (ambient_color + 0.5 * end_sun_color) * mie_phase;

#if defined PROGRAM_DEFERRED4
    // Sun

#ifdef END_SUN_EFFECT
    sky += draw_sun(ray_dir);
#endif

    // Stars

    vec3 stars_fade
        = exp2(-0.1 * max0(1.0 - ray_dir.y) / max(ambient_color, eps))
        * linear_step(-0.2, 0.0, ray_dir.y);
    sky += draw_stars(ray_dir, 0.0).xzy * stars_fade;
#endif

    return sky;
}

#endif

#endif // INCLUDE_SKY_SKY
