// Trace diffuse indirect GI rays (world-space voxel RT)

#include "/include/global.glsl"



#if RTGI_RESOLUTION == 0
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif RTGI_RESOLUTION == 2
const vec2 workGroupsRender = vec2(1.0, 1.0);
#else
const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba16f) writeonly uniform image2D rtgi_raw_img;
layout(rgba16f) writeonly uniform image2D rtgi_state_img;

#if GI_DEBUG == GI_DEBUG_HIT
layout(rgba8) writeonly uniform image2D rtgi_hit_img;
#endif

#include "/include/lighting/rt/voxel_samplers.glsl"
#if defined SHADOW && !defined WORLD_NETHER
uniform sampler2DShadow shadowtex1;
#define VOXEL_DDA_USE_SHADOWMAP
#endif
uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D colortex1; // gbuffer0
uniform sampler2D colortex2; // gbuffer1
uniform sampler2D colortex4; // sky map & light colors
uniform sampler2D noisetex;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjection;

uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform int frameCounter;

uniform vec2 view_res;
uniform vec2 view_pixel_size;
uniform vec3 light_dir;

uniform float near;
uniform float far;
uniform vec2 taa_offset;
uniform float eye_skylight;

#define TEMPORAL_REPROJECTION
#include "/include/utility/color.glsl"
#include "/include/utility/encoding.glsl"
#include "/include/utility/space_conversion.glsl"
#include "/include/utility/random.glsl"
#include "/include/lighting/rt/path_trace.glsl"
#include "/include/lighting/rt/pt_state.glsl"

void main() {
#if RTGI_RESOLUTION == 0
    const float render_scale = 0.25;
#elif RTGI_RESOLUTION == 2
    const float render_scale = 1.0;
#else
    const float render_scale = 0.5;
#endif

    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 target_res = ivec2(view_res * render_scale);

    if (texel.x >= target_res.x || texel.y >= target_res.y) {
        return;
    }

    // uv is a full-framebuffer UV. Under TAAU only the lower-left
    // taau_render_scale x taau_render_scale region contains scene data.
    vec2 uv = (vec2(texel) + 0.5) / vec2(target_res);

    if (taau_uv_outside_viewport(uv)) {
        imageStore(rtgi_raw_img, texel, vec4(0.0));
        imageStore(rtgi_state_img, texel, pt_state_pack(PT_RESET_NO_SURFACE, 1.0, 0.0, 0.0));
#if GI_DEBUG == GI_DEBUG_HIT
        imageStore(rtgi_hit_img, texel, vec4(0.0));
#endif
        return;
    }

    ivec2 full_texel = ivec2(uv * view_res);

    float depth = texelFetch(depthtex1, full_texel, 0).x;
    if (depth >= 1.0) {
        imageStore(rtgi_raw_img, texel, vec4(0.0));
        imageStore(rtgi_state_img, texel, pt_state_pack(PT_RESET_NO_SURFACE, 1.0, 0.0, 0.0));
#if GI_DEBUG == GI_DEBUG_HIT
        imageStore(rtgi_hit_img, texel, vec4(0.0));
#endif
        return;
    }

    // Unpack surface data
    vec4 gbuf0 = texelFetch(colortex1, full_texel, 0);
    vec3 world_normal = decode_unit_vector(unpack_unorm_2x8(gbuf0.z));
    float receiver_skylight = unpack_unorm_2x8(gbuf0.w).y;

#if defined NORMAL_MAPPING
    vec4 gbuf1 = texelFetch(colortex2, full_texel, 0);
    vec3 mapped_normal = decode_unit_vector(gbuf1.xy);
#else
    vec3 mapped_normal = world_normal;
#endif

    // Convert the framebuffer UV back to projection UV before inverting the
    // (jittered, TAAU-scaled) projection.
    vec3 screen_pos = vec3(taau_projection_uv(uv), depth);
    vec3 view_pos = screen_to_view_space(screen_pos, true);
    vec3 scene_pos = view_to_scene_space(view_pos);

    // Spatio-temporal blue-noise dither: wrap frameCounter to preserve fp32
    // mantissa precision across multi-hour play sessions.
    int frame_wrapped = frameCounter & 4095;
    ivec2 temporal_tile_shift = ivec2(frame_wrapped * 59, frame_wrapped * 97);
    vec2 dither = vec2(
        texelFetch(noisetex, (texel + temporal_tile_shift) & 511, 0).b,
        texelFetch(noisetex, (texel + temporal_tile_shift + ivec2(249, 137)) & 511, 0).b
    );
    dither = r2(frame_wrapped, dither);

    // Fast-path for distant surfaces outside the voxel volume: evaluate sky dome
    // directly in 0 DDA steps instead of marching backward into the player's local grid.
    if (!rt_voxel_contains_world(scene_pos + cameraPosition)) {
        vec3 sky_dir = sample_cosine_hemisphere(mapped_normal, dither);
        if (dot(sky_dir, world_normal) <= 0.0) {
            sky_dir = reflect(sky_dir, world_normal);
        }
#if defined WORLD_OVERWORLD
        float sky_vis = smoothstep(0.02, 0.25, receiver_skylight) * receiver_skylight;
#else
        const float sky_vis = 1.0;
#endif
        vec3 far_sky_rad = texture(colortex4, project_sky(sky_dir)).rgb * sky_vis;
        imageStore(rtgi_raw_img, texel, vec4(far_sky_rad, 32.0 / 34.0));
        imageStore(rtgi_state_img, texel, pt_state_pack(PT_RESET_NONE, 0.0, 1.0, 0.0));
#if GI_DEBUG == GI_DEBUG_HIT
        imageStore(rtgi_hit_img, texel, vec4(0.0, 0.0, float(sky_vis > 0.0), 1.0));
#endif
        return;
    }

    // Direct sun/moon color encoded in colortex4 column 191
    vec3 direct_light_col = texelFetch(colortex4, ivec2(191, 0), 0).rgb;

#ifndef RTGI_RAY_COUNT
#define RTGI_RAY_COUNT 2
#endif

    vec4 accum_radiance = vec4(0.0);
    float valid_paths = 0.0;

#if GI_DEBUG == GI_DEBUG_HIT
    vec3 hit_counts = vec3(0.0);
#endif

    for (int r = 0; r < RTGI_RAY_COUNT; ++r) {
        int hit_source = RTGI_HIT_NONE;
        vec4 radiance = evaluate_path_trace(
            scene_pos,
            world_normal,
            mapped_normal,
            dither,
            receiver_skylight,
            voxel_sampler,
            colortex4,
            direct_light_col,
            light_dir,
            r,
            hit_source
        );

#if GI_DEBUG == GI_DEBUG_HIT
        if (hit_source == RTGI_HIT_VOXEL) {
            hit_counts.g += 1.0;
        } else if (hit_source == RTGI_HIT_SKY) {
            hit_counts.b += 1.0;
        }
#endif

        // Firefly suppression
        float lum = dot(radiance.rgb, luminance_weights);
#ifndef RTGI_MAX_RADIANCE
#define RTGI_MAX_RADIANCE 24.0
#endif
        if (lum > RTGI_MAX_RADIANCE) {
            radiance.rgb *= RTGI_MAX_RADIANCE / lum;
        }

        // Exclude non-finite radiance
        bool path_valid = all(equal(radiance, radiance))
            && all(greaterThanEqual(radiance.rgb, vec3(0.0)));

        if (path_valid) {
            valid_paths += 1.0;
            accum_radiance += radiance;
        }
    }

    // Divide by valid contributing paths
    float path_divisor = max(valid_paths, 1.0);
    vec4 final_radiance = accum_radiance * (1.0 / path_divisor);
    final_radiance.a = clamp01(final_radiance.a);
    float valid_path_fraction = valid_paths / float(RTGI_RAY_COUNT);

    imageStore(rtgi_raw_img, texel, final_radiance);
    imageStore(
        rtgi_state_img,
        texel,
        pt_state_pack(PT_RESET_NONE, 0.0, valid_path_fraction, 0.0)
    );

#if GI_DEBUG == GI_DEBUG_HIT
    imageStore(rtgi_hit_img, texel, vec4(hit_counts / float(RTGI_RAY_COUNT), 1.0));
#endif
}
