/*
--------------------------------------------------------------------------------

  Photon Shader by SixthSurge

  program/d0_sky_map:
  Render omnidirectional sky map for reflections and SH lighting

--------------------------------------------------------------------------------
*/

#include "/include/global.glsl"

layout(location = 0) out vec3 sky_map;

/* RENDERTARGETS: 4 */

in vec2 uv;

flat in vec3 ambient_color;
flat in vec3 light_color;

#if defined WORLD_OVERWORLD
flat in vec3 sun_color;
flat in vec3 moon_color;
flat in vec3 sky_color;

flat in mat2x4 aurora_data;

float aurora_amount;
mat2x3 aurora_colors;
float rainbow_amount;

#include "/include/sky/clouds/parameters.glsl"
flat in CloudsParameters clouds_params;

#include "/include/fog/overworld/parameters.glsl"
flat in OverworldFogParameters fog_params;
#endif

// ------------
//   Uniforms
// ------------

uniform sampler3D colortex6; // 3D bubbly worley noise
#define SAMPLER_WORLEY_BUBBLY colortex6
uniform sampler3D colortex7; // 3D swirley worley noise
#define SAMPLER_WORLEY_SWIRLEY colortex7
uniform sampler2D colortex4; // previous frame sky map for hemisphere average

#if defined WORLD_OVERWORLD && defined GALAXY
uniform sampler2D colortex13;
#define galaxy_sampler colortex13
#endif

uniform sampler3D depthtex0; // atmospheric scattering LUT
uniform sampler2D depthtex1;

uniform sampler2D colortex8; // cloud shadow map

uniform sampler2D noisetex;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform float near;
uniform float far;

uniform int worldTime;
uniform int moonPhase;
uniform float sunAngle;

uniform int frameCounter;
uniform float frameTimeCounter;

uniform int isEyeInWater;
uniform float eyeAltitude;
uniform float rainStrength;
uniform float blindness;
uniform float darknessFactor;

uniform vec3 light_dir;
uniform vec3 sun_dir;
uniform vec3 moon_dir;

uniform vec2 view_res;
uniform vec2 view_pixel_size;
uniform vec2 taa_offset;

uniform float world_age;
uniform float eye_skylight;

uniform float time_sunrise;
uniform float time_noon;
uniform float time_sunset;
uniform float time_midnight;

uniform float biome_cave;
uniform float biome_may_snow;

// ------------
//   Includes
// ------------

#define ATMOSPHERE_SCATTERING_LUT depthtex0
#define CLOUDS_USE_LOCAL_COVERAGE_MAP

#ifdef CLOUDS_CUMULUS_PRECOMPUTE_LOCAL_COVERAGE
#define CLOUDS_USE_LOCAL_COVERAGE_MAP
#endif

#if defined WORLD_OVERWORLD
#include "/include/fog/overworld/analytic.glsl"
#ifdef CLOUD_SHADOWS
#include "/include/lighting/cloud_shadows.glsl"
#endif
#include "/include/sky/aurora.glsl"
#endif

#include "/include/sky/projection.glsl"
#include "/include/sky/sky.glsl"

void main() {
#if defined WORLD_OVERWORLD
    aurora_amount = aurora_data[0].w;
    aurora_colors = mat2x3(aurora_data[0].xyz, aurora_data[1].xyz);
    rainbow_amount = aurora_data[1].w;
#endif

    ivec2 texel = ivec2(gl_FragCoord.xy);

    if (texel.x == sky_map_res.x) { // Store lighting colors
        sky_map = vec3(0.0);
        if (texel.y >= 4 && texel.y <= 100) {
#if defined WORLD_OVERWORLD
            // 96-segment (97 texel, wrapped endpoints) unfogged horizon strip
            // in column 191 so deferred4 BORDER_FOG samples it in 1 hardware
            // linear fetch without double-fogging.
            float az = (float(texel.y - 4) * rcp(96.0)) * tau - pi;
            vec3 h_dir = normalize(vec3(sin(az), -0.1, -cos(az)));
            sky_map = draw_sky(h_dir);
#endif
        } else {
            switch (texel.y) {
                case 0:
                    sky_map = light_color;
                    break;

                case 1:
                    sky_map = ambient_color;
                    break;

                case 2: {
                    // 32-sample cosine-weighted upper-hemisphere average of the
                    // previous frame's sky map (1 texel/frame = 32 fetches total).
                    // Supplies neutral sky+clouds ambient for water fog (matching
                    // stock Photon's SH skylight_up) without disturbing air fog.
#if defined WORLD_OVERWORLD || defined WORLD_END
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
                    float night = clamp01(time_midnight);
                    if (night > 0.0) {
                        float sky_luma = dot(avg, luminance_weights_rec2020);
                        vec3 night_sky = mix(avg, vec3(sky_luma), 0.30) * 0.72;
                        avg = mix(avg, night_sky, night);
                    }
#endif
                    sky_map = max_of(avg) > 1e-6 ? avg : ambient_color * rcp_pi;
#else
                    sky_map = ambient_color;
#endif
                    break;
                }
                case 3: {
#if defined WORLD_OVERWORLD
                    // Frame-constant overcast horizon mist (1 texel/frame) for
                    // deferred4 draw_sky so sky pixels need zero extra VGPRs.
                    sky_map = overcast_horizon_mist(
                        sky_color,
                        sun_color,
                        moon_color,
                        vec3(0.0, 1.0, 0.0)
                    );
#else
                    sky_map = ambient_color;
#endif
                    break;
                }
            }
        }
    } else { // Draw sky map (with cloud-shadowed analytic fog for cheap
             // reflections; BORDER_FOG reads the unfogged strip in col 191)
        vec3 ray_dir = unproject_sky(uv);

        sky_map = draw_sky(ray_dir);

#if defined WORLD_OVERWORLD
        vec3 fog_dir = normalize(vec3(ray_dir.x, max(ray_dir.y, 0.01), ray_dir.z));
#ifdef CLOUD_SHADOWS
#ifdef LOD_MOD_ACTIVE
        float fog_end = float(lod_render_distance);
#else
        float fog_end = far;
#endif
        float fog_shadow = (
            get_cloud_shadows_fast(colortex8, fog_dir * (0.20 * fog_end))
            + get_cloud_shadows_fast(colortex8, fog_dir * (0.50 * fog_end))
            + get_cloud_shadows_fast(colortex8, fog_dir * (0.80 * fog_end))
        ) * rcp(3.0);
#else
        const float fog_shadow = 1.0;
#endif
        mat2x3 fog = air_fog_analytic(
            cameraPosition,
            cameraPosition + fog_dir,
            true,
            eye_skylight,
            fog_shadow
        );
        sky_map = sky_map * fog[1] + fog[0];
#endif
    }
}
