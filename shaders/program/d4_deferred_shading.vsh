/*
--------------------------------------------------------------------------------

  Photon Shader by SixthSurge

  program/d4_deferred_shading:
  Shade terrain and entities, draw sky

--------------------------------------------------------------------------------
*/

#include "/include/global.glsl"

out vec2 uv;

flat out vec3 ambient_color;
flat out vec3 light_color;
flat out vec3 sky_fallback_color;

#if defined WORLD_OVERWORLD
flat out vec3 sun_color;
flat out vec3 moon_color;

#include "/include/fog/overworld/parameters.glsl"
flat out OverworldFogParameters fog_params;

flat out float rainbow_amount;
#endif

// ------------
//   Uniforms
// ------------

uniform sampler3D depthtex0; // Atmosphere scattering LUT

uniform sampler2D colortex4; // Sky map and environment colors

uniform int worldTime;
uniform int worldDay;
uniform int moonPhase;
uniform float sunAngle;
uniform float rainStrength;
uniform float wetness;

uniform int frameCounter;
uniform float frameTimeCounter;

uniform int isEyeInWater;
uniform float blindness;
uniform float nightVision;
uniform float darknessFactor;

uniform vec3 light_dir;
uniform vec3 sun_dir;
uniform vec3 moon_dir;

uniform float biome_cave;
uniform float biome_may_rain;
uniform float biome_may_snow;
uniform float biome_snowy;
uniform float biome_temperate;
uniform float biome_arid;
uniform float biome_taiga;
uniform float biome_jungle;
uniform float biome_swamp;
uniform float biome_temperature;
uniform float biome_humidity;
uniform float desert_sandstorm;

uniform float world_age;
uniform float time_sunrise;
uniform float time_noon;
uniform float time_sunset;
uniform float time_midnight;

// ------------
//   Includes
// ------------

#define ATMOSPHERE_SCATTERING_LUT depthtex0
#define WEATHER_AURORA

#if defined WORLD_OVERWORLD
#include "/include/lighting/colors/light_color.glsl"
#include "/include/sky/atmosphere.glsl"
#include "/include/weather/fog.glsl"
#include "/include/weather/rainbow.glsl"
#endif

#include "/include/sky/projection.glsl"
#include "/include/utility/random.glsl"
#include "/include/utility/sampling.glsl"

#if defined WORLD_OVERWORLD || defined WORLD_END
// Cosine-weighted upper-hemisphere integral of colortex4 (sky_map) in E/pi
// radiance units, matching path_trace.glsl's voxel_environment() integration
// so terrain outside the RT voxel grid matches traced sky GI in brightness
// and color across clear, cloudy, foggy, rainy, and End skies.
vec3 integrate_sky_map_hemisphere() {
    const int samples = 32;
    const float golden_angle = 2.39996323;
    vec3 sum = vec3(0.0);
    for (int i = 0; i < samples; ++i) {
        float u = (float(i) + 0.5) / float(samples);
        float y = sqrt(u);
        float r = sqrt(max0(1.0 - u));
        float phi = float(i) * golden_angle;
        vec3 dir = vec3(r * cos(phi), y, r * sin(phi));
        vec3 s = textureLod(colortex4, project_sky(dir), 0.0).rgb;
        sum += (!any(isnan(s)) && !any(isinf(s))) ? max0(s) : vec3(0.0);
    }
    vec3 avg = sum / float(samples);
#if defined WORLD_OVERWORLD
    // Match voxel_environment() night skylight calibration so inside and
    // outside RT coverage stay seamless at night too.
    float night = clamp01(time_midnight);
    if (night > 0.0) {
        float sky_luma = dot(avg, luminance_weights_rec2020);
        vec3 night_sky = mix(avg, vec3(sky_luma), 0.30) * 0.72;
        avg = mix(avg, night_sky, night);
    }
#endif
    return avg;
}
#endif

void main() {
    uv = gl_MultiTexCoord0.xy;

    light_color = texelFetch(colortex4, ivec2(191, 0), 0).rgb;
    ambient_color = texelFetch(colortex4, ivec2(191, 1), 0).rgb;
#if defined WORLD_OVERWORLD || defined WORLD_END
    sky_fallback_color = integrate_sky_map_hemisphere();
#else
    sky_fallback_color = ambient_color;
#endif

#if defined WORLD_OVERWORLD
    Weather weather = get_weather();

    sun_color = get_sun_exposure() * get_sun_tint();
    moon_color = get_moon_exposure() * get_moon_tint();
    fog_params = get_fog_parameters(weather);

    rainbow_amount = get_rainbow_amount(weather);
#endif

    vec2 vertex_pos = gl_Vertex.xy * taau_render_scale;
    gl_Position = vec4(vertex_pos * 2.0 - 1.0, 0.0, 1.0);
}
