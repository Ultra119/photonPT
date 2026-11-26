/*
--------------------------------------------------------------------------------

  Photon Shader by SixthSurge

  program/d4_deferred_shading:
  Shade terrain and entities, draw sky

--------------------------------------------------------------------------------
*/

#include "/include/global.glsl"

layout(location = 0) out vec3 fragment_color;
layout(location = 1) out vec4 colortex3_clear;

/* RENDERTARGETS: 0,3 */

in vec2 uv;

flat in vec3 ambient_color;
flat in vec3 light_color;
flat in vec3 sky_fallback_color;

#if defined WORLD_OVERWORLD
flat in vec3 sun_color;
flat in vec3 moon_color;

#include "/include/fog/overworld/parameters.glsl"
flat in OverworldFogParameters fog_params;


flat in float rainbow_amount;
flat in float sky_overcast;
#endif

// ------------
//   Uniforms
// ------------

uniform sampler2D noisetex;

uniform sampler2D colortex0; // skytextured output
uniform sampler2D colortex1; // gbuffer 0
uniform sampler2D colortex2; // gbuffer 1
uniform sampler2D colortex3; // OF damage overlay, armor glint
uniform sampler2D colortex4; // sky map
uniform sampler2D colortex5; // previous frame color
uniform sampler2D colortex6; // ambient lighting data
uniform sampler2D colortex7; // previous frame fog scattering
uniform sampler2D colortex11; // clouds history
uniform sampler2D colortex12; // clouds data

#if defined WORLD_OVERWORLD && defined GALAXY
uniform sampler2D colortex13;
#define galaxy_sampler colortex13
#endif

#ifdef CLOUD_SHADOWS
uniform sampler2D colortex8; // cloud shadow map
#endif

uniform sampler3D depthtex0; // atmosphere scattering LUT
uniform sampler2D depthtex1;


#ifdef BLOCKY_CLOUDS
uniform sampler2D blocky_clouds_tex;
#endif

#ifndef WORLD_NETHER
#ifdef SHADOW
uniform sampler2D shadowtex0;
uniform sampler2DShadow shadowtex1;

#ifdef SHADOW_COLOR
uniform sampler2D shadowcolor0;
#endif
#endif
#endif

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform float eyeAltitude;
uniform float near;
uniform float far;

uniform int worldTime;
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
uniform vec3 view_light_dir;

uniform vec2 view_res;
uniform vec2 view_pixel_size;
uniform vec2 taa_offset;

uniform float biome_cave;
uniform float biome_may_rain;
uniform float biome_may_snow;

uniform float time_sunrise;
uniform float time_noon;
uniform float time_sunset;
uniform float time_midnight;

uniform float world_age;
uniform float eye_skylight;

/*
const bool colortex5MipmapEnabled = true;
const bool colortex11MipmapEnabled = true;
*/

// ------------
//   Includes
// ------------

#define ATMOSPHERE_SCATTERING_LUT depthtex0
#define TEMPORAL_REPROJECTION


#include "/include/fog/simple_fog.glsl"
#include "/include/lighting/diffuse_lighting.glsl"
#include "/include/lighting/rt/rt_ao_history.glsl"
#include "/include/lighting/rt/rtgi_opaque_depth.glsl"
#include "/include/lighting/shadows/common.glsl"
#include "/include/lighting/shadows/pcss.glsl"
#include "/include/lighting/shadows/ssrt.glsl"
#include "/include/lighting/specular_lighting.glsl"
#include "/include/misc/lod_mod_support.glsl"
#include "/include/misc/purkinje_shift.glsl"
#include "/include/sky/sky.glsl"
#include "/include/surface/edge_highlight.glsl"
#include "/include/surface/material.glsl"
#include "/include/surface/rain_puddles.glsl"
#include "/include/utility/bicubic.glsl"
#include "/include/utility/color.glsl"
#include "/include/utility/encoding.glsl"
#include "/include/utility/space_conversion.glsl"

#if defined WORLD_OVERWORLD
#include "/include/sky/clouds/sampling.glsl"
#include "/include/sky/rainbow.glsl"
#if defined BLOCKY_CLOUDS
#include "/include/sky/blocky_clouds.glsl"
#endif
#endif

#if defined CLOUD_SHADOWS
#include "/include/lighting/cloud_shadows.glsl"
#endif

float sample_rt_ao(
    ivec2 pixel, ivec2 full_size, vec2 jitter, vec3 view_pos, vec3 world_normal,
    uint kind
) {
    if (RT_AO_STRENGTH <= 0.0 || kind != rtgi_world || view_pos.z >= 0.0) {
        return 1.0;
    }

    RtgiGrid grid = rt_ao_grid(full_size, textureSize(colortex6, 0));
    if (!rtgi_in_bounds(pixel, grid.full_active) || any(equal(grid.gi_active, ivec2(0)))) {
        return 1.0;
    }

    ivec2 base;
    vec2 fraction;
    rtgi_footprint(vec2(pixel) + 0.5, grid, base, fraction);
    mat3 rotation = mat3(gbufferModelView);
    vec3 normal = rotation * world_normal;
    float footprint = rtgi_view_footprint(-view_pos.z, grid.gi_active, gbufferProjection);
    float chord = rt_ao_chord(view_pos, normal, footprint);
    float sum = 0.0;
    float weight_sum = 0.0;

    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            ivec2 offset = ivec2(x, y);
            ivec2 tap = base + offset;
            float weight = rtgi_bilinear_weight(offset, fraction);
            if (weight <= 0.0 || !rtgi_in_bounds(tap, grid.gi_active)) {
                continue;
            }
            vec4 sample_value = texelFetch(colortex6, tap, 0);
            vec3 tap_view;
            if (!rt_ao_view_pos(sample_value, tap, grid, jitter, gbufferProjection, tap_view)) {
                continue;
            }
            vec3 tap_normal = rotation * decode_unit_vector(unpack_unorm_2x8(sample_value.a));
            weight *= rtgi_geometry_weight(tap_view - view_pos, normal, tap_normal, footprint, chord);
            sum += weight * sample_value.r;
            weight_sum += weight;
        }
    }
    return weight_sum > 1e-4 ? sum / weight_sum : 1.0;
}

void main() {
    colortex3_clear = vec4(0.0);

    ivec2 texel = ivec2(gl_FragCoord.xy);

    // Sample textures

    float depth = texelFetch(combined_depth_tex, texel, 0).x;
    vec4 gbuffer_data_0 = texelFetch(colortex1, texel, 0);
#if defined NORMAL_MAPPING || defined SPECULAR_MAPPING
    vec4 gbuffer_data_1 = texelFetch(colortex2, texel, 0);
#endif
    vec4 overlays = texelFetch(colortex3, texel, 0);

    RtgiOpaqueDepth surface = rtgi_opaque_depth(texel);
    bool is_hand = surface.kind == rtgi_hand;
    bool is_sky = surface.kind == rtgi_sky;
#ifdef LOD_MOD_ACTIVE
    bool is_lod = surface.kind == rtgi_lod;
    float depth_lod = is_lod ? surface.depth : rtgi_lod_depth(texel);
#endif

    // Space conversions

    ivec2 full_size = textureSize(depthtex1, 0);
    vec2 jitter = rtgi_raster_jitter(taa_offset, full_size);
    vec3 position_view = rtgi_opaque_view_pos(surface, gl_FragCoord.xy, full_size, jitter);
    vec3 position_scene = view_to_scene_space(position_view);
    vec3 position_world = position_scene + cameraPosition;
    vec3 direction_world
        = normalize(position_scene - gbufferModelViewInverse[3].xyz);

#if defined WORLD_OVERWORLD
    // Read clouds/aurora/crepuscular rays

    float clouds_apparent_distance;
    vec4 clouds_and_aurora
        = read_clouds_and_aurora(uv, clouds_apparent_distance);

    // Atmosphere is needed immediately for sky and blocky clouds, or lazily
    // for terrain pixels inside border fog.
#ifdef BLOCKY_CLOUDS
    vec3 atmosphere = atmosphere_scattering(
        direction_world,
        sun_color,
        sun_dir,
        moon_color,
        moon_dir,
        /* use_klein_nishina_phase */ is_sky
    );
#endif

    // Blocky clouds

#ifdef BLOCKY_CLOUDS
    vec3 world_start_pos = gbufferModelViewInverse[3].xyz + cameraPosition;
    vec3 world_end_pos = position_world;

    float dither = texelFetch(noisetex, texel & 511, 0).b;
    dither = r1(frameCounter, dither);

    vec4 blocky_clouds = raymarch_blocky_clouds(
        world_start_pos,
        world_end_pos,
        is_sky,
        blocky_clouds_altitude_l0,
        dither
    );

#ifdef BLOCKY_CLOUDS_LAYER_2
    float visibility = pow4(blocky_clouds.a);
    vec4 blocky_clouds_l2 = raymarch_blocky_clouds(
        world_start_pos,
        world_end_pos,
        is_sky,
        blocky_clouds_altitude_l1,
        dither
    );
    blocky_clouds.rgb += blocky_clouds_l2.xyz * visibility;
    blocky_clouds.a *= mix(1.0, blocky_clouds_l2.a, visibility);
#endif

    float new_alpha = sqr(sqr(blocky_clouds.a));
    blocky_clouds.rgb
        += atmosphere * (1.0 - new_alpha) * (blocky_clouds.a - new_alpha);
    blocky_clouds.a = new_alpha;
#endif
#endif

    if (is_sky) {
#if defined WORLD_OVERWORLD
#ifndef BLOCKY_CLOUDS
        vec3 atmosphere = atmosphere_scattering(
            direction_world,
            sun_color,
            sun_dir,
            moon_color,
            moon_dir,
            /* use_klein_nishina_phase */ true
        );
#endif
        fragment_color = draw_sky(
            direction_world,
            atmosphere,
            clouds_and_aurora,
            clouds_apparent_distance,
            sky_overcast
        );
#else
        fragment_color = draw_sky(direction_world);
#endif

        // Apply blocky clouds
#if defined WORLD_OVERWORLD && defined BLOCKY_CLOUDS
        fragment_color = fragment_color * blocky_clouds.w + blocky_clouds.xyz;
#endif

        // Apply common fog
        vec4 fog = common_fog(far, true);
        fragment_color = mix(fog.rgb, fragment_color.rgb, fog.a);

        // Apply purkinje shift
        fragment_color = purkinje_shift(fragment_color, vec2(0.0, 1.0));
    } else { // Terrain
        // Unpack gbuffer data

        mat4x2 data = mat4x2(
            unpack_unorm_2x8(gbuffer_data_0.x),
            unpack_unorm_2x8(gbuffer_data_0.y),
            unpack_unorm_2x8(gbuffer_data_0.z),
            unpack_unorm_2x8(gbuffer_data_0.w)
        );

        vec3 albedo = vec3(data[0], data[1].x);
        uint material_mask = uint(round(255.0 * data[1].y));
        vec3 flat_normal = decode_unit_vector(data[2]);
        vec2 light_levels = data[3];

        uint overlay_id = uint(255.0 * overlays.a);
        albedo = overlay_id == 0u ? albedo + overlays.rgb
                                  : albedo; // enchantment glint
        albedo = overlay_id == 1u && !is_hand
            ? 2.0 * albedo * overlays.rgb
            : albedo; // damage overlay

        // Get material and normal

        Material material = material_from(
            albedo,
            material_mask,
            position_world,
            flat_normal,
            light_levels
        );

        vec3 normal = flat_normal;
        bool parallax_shadow = false;

#ifdef LOD_MOD_ACTIVE
        if (!is_lod) {
#endif

#ifdef NORMAL_MAPPING
            normal = decode_unit_vector(gbuffer_data_1.xy);
#endif

#ifdef SPECULAR_MAPPING
            vec4 specular_map = vec4(
                unpack_unorm_2x8(gbuffer_data_1.z),
                unpack_unorm_2x8(gbuffer_data_1.w)
            );
            decode_specular_map(specular_map, material, parallax_shadow);
#elif defined NORMAL_MAPPING
        parallax_shadow = gbuffer_data_1.z >= 0.5;
#endif

#ifdef LOD_MOD_ACTIVE
        }
#endif

        // Rain puddles

#if defined WORLD_OVERWORLD && defined RAIN_PUDDLES
        if (wetness > eps && biome_may_rain > eps) {
            bool puddle = get_rain_puddles(
                position_world,
                flat_normal,
                light_levels,
                material.porosity,
                material_mask,
                normal,
                material.albedo,
                material.f0,
                material.roughness,
                material.ssr_multiplier
            );
        }
#endif

        float ao = sample_rt_ao(texel, full_size, jitter, position_view, flat_normal, surface.kind);

        // Calculate lighting dot products

        float NoL = dot(normal, light_dir);
        float NoV = clamp01(dot(normal, -direction_world));
        float LoV = dot(light_dir, -direction_world);
        float halfway_norm = inversesqrt(max(2.0 * LoV + 2.0, 1e-6));
        float NoH = (NoL + NoV) * halfway_norm;
        float LoH = LoV * halfway_norm + halfway_norm;

        // Cloud shadows

#if defined WORLD_OVERWORLD && defined CLOUD_SHADOWS
        float cloud_shadows = get_cloud_shadows(colortex8, position_scene);
#else
        const float cloud_shadows = 1.0;
#endif

        // Shadows

#if defined WORLD_OVERWORLD || defined WORLD_END
        vec3 shadows = vec3(0.0);
        float shadow_distance_fade = 1.0;
        float sss_depth = 0.0;
        bool has_direct_light = (cloud_shadows * max_of(light_color)) > 1e-4;

        if (has_direct_light && (NoL > 1e-3 || material.sss_amount > 1e-3)) {
            // Calculate near shadows
            vec3 shadow_near = vec3(0.0);
            float shadow_distant = 0.0;
            float sss_depth_near = 0.0;
            float sss_depth_distant = 0.0;

#ifdef SHADOW
            shadow_near = get_filtered_shadows(
                position_scene,
                flat_normal,
                light_levels.y,
                cloud_shadows,
                material.sss_amount,
                shadow_distance_fade,
                sss_depth_near
            );
#endif

            // Calculate distant shadows
            if (shadow_distance_fade >= eps) {
#ifdef SHADOW_SSRT
                shadow_distant = get_screen_space_shadows(
                    uv,
                    position_view,
                    depth,
#ifdef LOD_MOD_ACTIVE
                    depth_lod,
#endif
                    light_levels.y,
                    material.sss_amount > eps,
                    sss_depth_distant
                );
#else
                shadow_distant = get_lightmap_shadows(light_levels.y);
#endif
            }

            shadows = mix(
                shadow_near,
                vec3(shadow_distant),
                clamp01(shadow_distance_fade)
            );

            sss_depth = mix(
                sss_depth_near,
                sss_depth_distant,
                clamp01(shadow_distance_fade)
            );

            // Apply parallax shadow
#if defined POM && defined POM_SHADOW \
    && (defined SPECULAR_MAPPING || defined NORMAL_MAPPING)
            shadows *= float(!parallax_shadow);
#endif
        }
#else
        const vec3 shadows = vec3(1.0);
        const float shadow_distance_fade = 1.0;
        const float sss_depth = 0.0;
#endif

        // Diffuse lighting

        vec3 indirect_radiance = sample_rtgi_diffuse(
            texel, full_size, position_view, position_scene, flat_normal,
            material_mask, surface.kind, light_levels, cloud_shadows,
            max_of(shadows)
        ) * RTGI_INTENSITY;
        fragment_color = get_diffuse_lighting(
            material,
            position_scene,
            normal,
            flat_normal,
            normal, // Deferred RTGI does not use a bent-normal probe offset.
            shadows,
            light_levels,
            ao,
            sss_depth,
#ifdef CLOUD_SHADOWS
            cloud_shadows,
#endif
#ifdef SHADOW_SSRT
            0.0,
#else
            shadow_distance_fade,
#endif
            NoL,
            NoV,
            NoH,
            LoV,
            indirect_radiance
        );

        // Specular highlight

#if defined WORLD_OVERWORLD || defined WORLD_END
        if (has_direct_light && NoL > eps && max_of(shadows) > eps) {
            vec3 direct_specular = get_specular_highlight(material, NoL, NoV, NoH, LoV, LoH)
                * light_color * shadows * cloud_shadows;
#ifdef RT_AO_IN_DIRECT
            direct_specular *= ao;
#endif
            fragment_color += direct_specular;
        }
#endif

        // Specular reflections

#if defined ENVIRONMENT_REFLECTIONS || defined SKY_REFLECTIONS
        if (material.ssr_multiplier > eps) {
            mat3 tbn = get_tbn_matrix(normal);

            fragment_color += get_specular_reflections(
                material,
                tbn,
                vec3(uv, depth),
                position_view,
                position_world,
                normal,
                flat_normal,
                direction_world,
                direction_world * tbn,
                light_levels.y,
                false
            );
        }
#endif
        // Edge highlight

#ifdef EDGE_HIGHLIGHT
        fragment_color *= 1.0
            + 0.5
                * get_edge_highlight(
                    position_scene,
                    flat_normal,
                    depth,
                    material_mask
                );
#endif

        // Apply fog

        float view_distance = length(position_view);

#ifdef BORDER_FOG
        float border_fog = border_fog(position_scene, direction_world);
        if (border_fog < 0.9999) {
#if defined WORLD_OVERWORLD
#ifndef BLOCKY_CLOUDS
            vec3 atmosphere = atmosphere_scattering(
                direction_world,
                sun_color,
                sun_dir,
                moon_color,
                moon_dir,
                /* use_klein_nishina_phase */ false
            );
#endif
            vec3 horizon_dir = normalize(
                vec3(direction_world.xz, min(direction_world.y, -0.1)).xzy
            );
            vec3 horizon_color = texture(colortex4, project_sky(horizon_dir)).rgb;

            float horizon_factor
                = linear_step(0.1, 1.0, exp(-75.0 * sqr(sun_dir.y + 0.0496)));
            horizon_factor = clamp01(horizon_factor + step(0.01, rainStrength));
            horizon_factor = max(
                horizon_factor,
                dampen(linear_step(0.15, 0.05, direction_world.y))
            );

            vec3 border_fog_color
                = mix(atmosphere, horizon_color, sqr(horizon_factor))
                * (1.0 - biome_cave);
#else
            vec3 border_fog_color
                = texture(colortex4, project_sky(direction_world)).rgb;
#endif

            fragment_color = mix(border_fog_color, fragment_color, border_fog);
        }
#endif

        vec4 fog = common_fog(view_distance, false);
        fragment_color = fragment_color * fog.a + fog.rgb;

#if defined WORLD_OVERWORLD
        // Apply clouds in front of terrain

#ifdef BLOCKY_CLOUDS
        fragment_color = fragment_color * blocky_clouds.w + blocky_clouds.xyz;
#else
        if (sqr(clouds_apparent_distance) < length_squared(position_view)) {
            fragment_color
                = fragment_color * clouds_and_aurora.w + clouds_and_aurora.xyz;
        }
#endif

        // Apply rainbows in front of terrain

#ifdef RAINBOWS
        if (rainbow_amount > eps) {
            fragment_color = draw_rainbows(
                fragment_color,
                direction_world,
                min(view_distance,
                    mix(clouds_apparent_distance,
                        1e6,
                        linear_step(1.0, 0.95, clouds_and_aurora.w)))
            );
        }
#endif
#endif

        // Apply purkinje shift

        fragment_color = purkinje_shift(fragment_color, light_levels);
    }
}
