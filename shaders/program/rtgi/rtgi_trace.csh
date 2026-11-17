// Trace one diffuse frame estimate at each registered opaque raster anchor.

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
layout(rgba32ui) writeonly uniform uimage2D rtgi_guide_even_img;
layout(rgba32ui) writeonly uniform uimage2D rtgi_guide_odd_img;
layout(rgba32ui) writeonly uniform uimage2D rtgi_frame_state_img;

#if GI_DEBUG == GI_DEBUG_HIT
layout(rgba8) writeonly uniform image2D rtgi_hit_img;
#endif

#include "/include/lighting/rt/voxel_samplers.glsl"
#if defined SHADOW && !defined WORLD_NETHER
uniform sampler2DShadow shadowtex1;
#define VOXEL_DDA_USE_SHADOWMAP
#endif
uniform sampler2D depthtex1;
uniform sampler2D colortex1;
uniform sampler2D colortex4; // sky map & light colors
uniform sampler2D noisetex;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

uniform vec3 cameraPosition;
uniform int frameCounter;

uniform vec3 light_dir;

uniform float near;
uniform float far;
uniform vec2 taa_offset;
uniform float eye_skylight;

#include "/include/lighting/rt/rtgi_opaque_depth.glsl"
#include "/include/lighting/rt/path_trace.glsl"
#include "/include/lighting/rt/pt_state.glsl"

void store_trace(ivec2 texel, vec4 raw, uvec4 guide, vec4 hits) {
    imageStore(rtgi_raw_img, texel, raw);
    if ((frameCounter & 1) == 0) {
        imageStore(rtgi_guide_even_img, texel, guide);
    } else {
        imageStore(rtgi_guide_odd_img, texel, guide);
    }
#if GI_DEBUG == GI_DEBUG_HIT
    imageStore(rtgi_hit_img, texel, hits);
#endif
}

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 target_res = imageSize(rtgi_raw_img);
    if (!rtgi_in_bounds(texel, target_res)) {
        return;
    }

    RtgiGrid grid = rtgi_grid(textureSize(depthtex1, 0), target_res);
    vec2 jitter = rtgi_raster_jitter(taa_offset, grid.full_size);
    if (texel == ivec2(0)) {
        // Publish a single record for the parity that owns this raster lattice.
        int bank = frameCounter & 1;
        imageStore(rtgi_frame_state_img, ivec2(0, bank),
                   uvec4(uint(frameCounter), rtgi_frame_tag, floatBitsToUint(jitter)));
        imageStore(rtgi_frame_state_img, ivec2(1, bank),
                   uvec4(grid.full_size, grid.full_active));
        imageStore(rtgi_frame_state_img, ivec2(2, bank),
                   uvec4(target_res, grid.gi_active));
    }

    if (!rtgi_in_bounds(texel, grid.gi_active)) {
        store_trace(texel, vec4(0.0), uvec4(0u), vec4(0.0));
        return;
    }

    ivec2 full_texel = rtgi_anchor(texel, grid);
    RtgiOpaqueDepth surface = rtgi_opaque_depth(full_texel);
    if (surface.kind != rtgi_world) {
        // Hand and LoD are shaded by the receiver-local fallback in D4.
        uvec4 guide = rtgi_pack_guide(full_texel, 0.0, vec2(0.0), 0u,
                                      surface.kind, 0u);
        store_trace(texel, vec4(0.0), guide, vec4(0.0));
        return;
    }

    vec3 view_pos = rtgi_opaque_view_pos(surface, vec2(full_texel) + 0.5, grid.full_size, jitter);
    vec3 scene_pos = transform(gbufferModelViewInverse, view_pos);
    vec4 gbuf0 = texelFetch(colortex1, full_texel, 0);
    vec2 normal_oct = unpack_unorm_2x8(gbuf0.z);
    vec3 world_normal = decode_unit_vector(normal_oct);
    uint material = uint(round(255.0 * unpack_unorm_2x8(gbuf0.y).y));
    float receiver_skylight = unpack_unorm_2x8(gbuf0.w).y;

    bool valid_position = rtgi_finite(vec4(scene_pos, view_pos.z)) && view_pos.z < 0.0;
    vec3 voxel_pos = scene_to_voxel_space(scene_pos);
    if (!valid_position || any(lessThan(voxel_pos, vec3(0.0)))
        || any(greaterThanEqual(voxel_pos, rt_voxel_size_f))) {
        uvec4 guide = rtgi_pack_guide(full_texel, 0.0, normal_oct, material,
                                      rtgi_world, 0u);
        store_trace(texel, vec4(0.0), guide, vec4(0.0));
        return;
    }

    // Spatio-temporal blue-noise dither: wrap frameCounter to preserve fp32
    // mantissa precision across multi-hour play sessions.
    int frame_wrapped = frameCounter & 4095;
    ivec2 temporal_tile_shift = ivec2(frame_wrapped * 59, frame_wrapped * 97);
    vec2 dither = vec2(
        texelFetch(noisetex, (texel + temporal_tile_shift) & 511, 0).b,
        texelFetch(noisetex, (texel + temporal_tile_shift + ivec2(249, 137)) & 511, 0).b
    );
    dither = r2(frame_wrapped, dither);

    // Direct sun/moon color encoded in colortex4 column 191
    vec3 direct_light_col = texelFetch(colortex4, ivec2(191, 0), 0).rgb;

    vec3 accum_radiance = vec3(0.0);
    float hit_distance_sum = 0.0;
    uint hit_count = 0u;
    uint sky_count = 0u;
    bool valid_batch = true;

    for (int r = 0; r < RTGI_RAY_COUNT; ++r) {
        int hit_source = RTGI_HIT_NONE;
        vec4 radiance = evaluate_path_trace(
            scene_pos,
            world_normal,
            dither,
            receiver_skylight,
            voxel_sampler,
            colortex4,
            direct_light_col,
            light_dir,
            r,
            hit_source
        );

        float lum = dot(radiance.rgb, luminance_weights);
        if (!rtgi_finite(radiance) || isinf(lum) || any(lessThan(radiance.rgb, vec3(0.0)))) {
            valid_batch = false;
            break;
        }
        // Keep the existing user-configured cap before conversion to FP16.
        accum_radiance += radiance.rgb * min(1.0, RTGI_MAX_RADIANCE / max(lum, eps));

        if (hit_source == RTGI_HIT_VOXEL) {
            hit_distance_sum += radiance.a;
            ++hit_count;
        } else if (hit_source == RTGI_HIT_SKY) {
            ++sky_count;
        }
    }

    float hit_distance = hit_count > 0u ? hit_distance_sum / float(hit_count) : -1.0;
    uint flags = rtgi_guide_covered | (valid_batch ? rtgi_guide_raw_valid : 0u);
    uvec4 guide = rtgi_pack_guide(full_texel, -view_pos.z, normal_oct, material,
                                 rtgi_world, flags);
    vec4 raw = valid_batch ? vec4(accum_radiance / float(RTGI_RAY_COUNT), hit_distance)
                           : vec4(0.0, 0.0, 0.0, -2.0);
    vec4 hits = valid_batch ? vec4(0.0, vec2(hit_count, sky_count) / float(RTGI_RAY_COUNT), 1.0)
                            : vec4(1.0, 0.0, 0.0, 1.0);
    store_trace(texel, raw, guide, hits);
}
